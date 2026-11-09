"""Restricted FP32 weight/LoRA bridge; no implicit fitting or model loading."""
import hashlib
import json
import numpy as np
from .wsp_artifact import TensorArtifact
from .wsp_editor import plan,apply
from .wsp_values import Limits

LAYERS=(26,27,28,29)
RANK=8
SCALE=1.0
TARGETS={f'W{i}':f'model.layers.{i}.self_attn.o_proj.weight' for i in LAYERS}


def tensor_hash(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode()+json.dumps(a.shape).encode()+a.tobytes()).hexdigest()


def check_base(base):
    if set(base)!=set(TARGETS):raise ValueError('exact target scope required')
    for value in base.values():
        if value.shape!=(576,576) or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('FP32 576x576 finite matrices required')


def initial_bank(base,mode):
    check_base(base)
    if mode=='direct':return {**{k:v.copy() for k,v in base.items()},**{f'P{i}':base[f'W{i}'].copy() for i in LAYERS}}
    if mode!='lora':raise ValueError('unknown mode')
    # Both factors start zero: direct program authoring, not gradient initialization.
    bank={}
    for i in LAYERS:
        bank[f'P{i}']=base[f'W{i}'].copy()
        bank[f'A{i}']=np.zeros((RANK,576),np.float32)
        bank[f'B{i}']=np.zeros((576,RANK),np.float32)
    return bank


def effective_weights(base,bank,mode,max_relative_norm=.1):
    expected=initial_bank(base,mode)
    if set(bank)!=set(expected):raise ValueError('unexpected tensors')
    for k,v in bank.items():
        if v.shape!=expected[k].shape or v.dtype!=np.float32 or not np.isfinite(v).all():raise ValueError('invalid tensor: '+k)
    result={};metrics={}
    for i in LAYERS:
        key=f'W{i}'
        if tensor_hash(bank[f'P{i}'])!=tensor_hash(base[key]):raise ValueError('read-only base matrix changed')
        if mode=='lora':
            with np.errstate(over='raise',invalid='raise'):
                delta=np.float32(SCALE)*(bank[f'B{i}']@bank[f'A{i}'])
                candidate=base[key]+delta
        else:candidate=bank[key].copy()
        if not np.isfinite(candidate).all():raise ValueError('nonfinite effective weights')
        actual=candidate.astype(np.float64)-base[key].astype(np.float64)
        norm=float(np.linalg.norm(actual));den=float(np.linalg.norm(base[key].astype(np.float64)))
        if norm>max_relative_norm*den:raise ValueError('effective update norm budget exceeded')
        result[key]=candidate;metrics[key]=dict(delta_norm=norm,base_norm=den,relative_norm=norm/den if den else 0.,changed=int(np.count_nonzero(actual)))
    return result,metrics


def edit_wspl(base,bank,mode,source):
    effective_weights(base,bank,mode)
    parent=TensorArtifact.from_arrays(bank,graph=dict(kind='opaque',label='SmolLM adapter bank; out_in; rank8 scale1'),max_elements=3_000_000)
    if '{{PARENT_SHA256}}' not in source:raise ValueError('parent placeholder required')
    source=source.replace('{{PARENT_SHA256}}',parent.sha256)
    staged=plan(source,parent=parent,limits=Limits(max_elements=3_000_000,max_operations=100_000_000,max_statements=100_000,max_loop_iterations=100_000))
    candidate=apply(staged,parent=parent,expected_parent_id=parent.sha256)
    if candidate.to_dict()['graph']!=parent.to_dict()['graph']:raise ValueError('fixed adapter schema required')
    arrays=candidate.arrays();effective,metrics=effective_weights(base,arrays,mode)
    return arrays,effective,dict(parent_sha256=parent.sha256,candidate_sha256=candidate.sha256,plan_sha256=staged.sha256,metrics=metrics)
