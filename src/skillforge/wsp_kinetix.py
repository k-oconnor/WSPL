"""Matched FP32 native/WSPL Kinetix construction. No simulation or model calls."""
from pathlib import Path
import re
import numpy as np
from .kinetix_replay import parse_parent,imported_artifact,ALIASES,LAYERS,SHAPES,reconstruct_native
from .wsp_breakout import fenced
from .native_weight_edits import statements
from .wsp_editor import plan,apply
from .wsp_syntax import parse
from .wsp_values import Limits

LIMITS=Limits(max_elements=171664,max_operations=100_000_000,max_statements=4096,max_loop_iterations=4096)


def native_source(body,parent):
    normalized=[];native=[]
    for row in statements(body):
        zero=re.fullmatch(r'ZERO\s+(embed|actor[0-4]|actor_head)\.(W|b)',row)
        if zero:
            key=zero[1]+'.'+zero[2]
            row=key+('[:, :]' if len(SHAPES[key])==2 else '[:]')+' = 0'
        match=re.match(r'^(embed|actor[0-4]|actor_head)\.(W|b)(\[.*)',row,re.S)
        if not match:raise ValueError('known tensor required')
        normalized.append(ALIASES[match[1]+'.'+match[2]]+match[3]+';');native.append(row)
    if not normalized or len(normalized)>4096:raise ValueError('1..4096 statements required')
    source='wspl 0.2; target base=artifact("parent",sha256="'+parent.sha256+'"); edit base as candidate mode=fixed numeric=native_numpy_v1 {\n'+'\n'.join(normalized)+'\n}'
    return source,'```edits\n'+'\n'.join(native)+'\n```'


def prompt(reference,guide,arm,parent_id):
    if arm not in ('native','wspl'):raise ValueError('unknown arm')
    # Keep original problem, parent failure trace and all rounded weights;
    # replace historical parsing instructions symmetrically.
    task=reference[reference.index('THE SCENE AND THE RULES'):reference.index('EDIT FORMAT')]
    weights=reference[reference.index('THE WEIGHTS (complete actor).'):]
    return ('Repair the supplied neural car controller. One fresh response, no tools or evaluation feedback. '
        'Explain your strategy briefly and return exactly one candidate fence.\n\n'+task+'\n'+guide.replace('{PARENT_SHA256}',parent_id)+'\n'+weights+'\nSELECTED ARM: '+arm+'\n')


class KinetixRepair:
    def __init__(self,reference_prompt):
        self.original={k:a.astype(np.float32) for k,a in parse_parent(Path(reference_prompt).read_text(encoding='utf-8')).items()}
        self.parent=imported_artifact(self.original);self.constructed=False

    def construct(self,arm,answer):
        if self.constructed:raise ValueError('one candidate per constructor')
        self.constructed=True
        if arm=='native':
            source,native=native_source(fenced(answer,'edits'),self.parent)
            independent,_=reconstruct_native(self.original,native)
        elif arm=='wspl':source=fenced(answer,'wspl');independent=None
        else:raise ValueError('unknown arm')
        tree=parse(source)
        if tree['edit'] is None or tree['edit']['mode']!='fixed' or tree['regions'] or tree['circuits'] or tree['descriptions']:
            raise ValueError('fixed tensor editing only')
        p=plan(source,parent=self.parent,limits=LIMITS)
        artifact=apply(p,parent=self.parent,expected_parent_id=self.parent.sha256)
        if artifact.to_dict()['graph']!=self.parent.to_dict()['graph']:raise ValueError('architecture changed')
        arrays=artifact.arrays();self.weights={k:arrays[v] for k,v in ALIASES.items()}
        for k,a in self.weights.items():
            if not np.isfinite(a).all() or (np.abs(a)>1e6).any():raise ValueError('weights outside finite bounds')
            if independent is not None and not np.array_equal(a.view('u4'),independent[k].view('u4')):raise ValueError('native/editor parity differs')
            a.flags.writeable=False
        return artifact,p,source

    def logits(self,observation):
        x=np.asarray(observation,dtype=np.float32)
        if x.shape!=(679,) or not np.isfinite(x).all():raise ValueError('679 finite observations required')
        for i,name in enumerate(LAYERS):
            x=x@self.weights[name+'.W']+self.weights[name+'.b']
            if i==0:x=np.maximum(x,np.float32(0))
            elif i<6:x=np.tanh(x)
        return x
