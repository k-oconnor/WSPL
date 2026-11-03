"""Empirical computational intervention logs for opaque PyTorch modules.

Tensor input/output and tensor-valued module sites; eval/stateless adapter scope.
No world-causal, semantic-grounding or universal verification claim. Models with
unregistered hidden state need a separate reset adapter; concurrent use unsupported.
"""
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import sys
import time


def _canonical(d):return json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def tensor_digest(tensor):
    import torch
    if not isinstance(tensor,torch.Tensor) or tensor.layout!=torch.strided or tensor.is_quantized:
        raise ValueError('dense non-quantized tensor required')
    t=tensor.detach().cpu().contiguous()
    raw=t.reshape(-1).view(torch.uint8).numpy().tobytes()
    header=_canonical(dict(shape=list(t.shape),dtype=str(t.dtype)))
    return hashlib.sha256(header+raw).hexdigest()


def _summary(tensor):
    import torch
    if not isinstance(tensor,torch.Tensor) or not tensor.is_floating_point() or not tensor.numel() or not torch.isfinite(tensor).all():
        raise ValueError('nonempty finite real floating output/site required')
    x=tensor.detach().cpu().double()
    return dict(shape=list(x.shape),dtype=str(tensor.dtype),device=str(tensor.device),sha256=tensor_digest(tensor),
                minimum=float(x.min()),maximum=float(x.max()),max_abs=float(x.abs().max()))


def model_identity(model,runtime_id):
    """Bind registered tensors and available module class source, not all code.

    runtime_id must identify the caller's external runtime/adapter. Python globals,
    external code and unregistered custom state are not completely fingerprinted.
    """
    raw=model.state_dict()
    state={'state:'+k:tensor_digest(v) for k,v in sorted(raw.items())}
    for key,value in model.named_buffers():
        if key not in raw:state['nonpersistent_buffer:'+key]=tensor_digest(value)
    modules=[]
    for path,module in model.named_modules():
        cls=type(module)
        try:source=hashlib.sha256(inspect.getsource(cls).encode('utf-8')).hexdigest()
        except (OSError,TypeError):source=None
        modules.append(dict(path=path,kind=cls.__module__+'.'+cls.__qualname__,configuration=module.extra_repr(),class_source_sha256=source))
    body=dict(runtime_id=runtime_id,state=state,modules=modules)
    return dict(sha256=hashlib.sha256(_canonical(body)).hexdigest(),registered_state_sha256=hashlib.sha256(_canonical(state)).hexdigest(),
                runtime_id=runtime_id,modules=modules,coverage='registered tensors including nonpersistent buffers/module configuration/classes/available class source; external code and hidden state not fully bound')


@dataclass(frozen=True)
class Intervention:
    module: str
    kind: str = 'zero'
    call_index: int = 0
    features: tuple | None = None
    value: object = None

    def __post_init__(self):
        if not isinstance(self.module,str) or self.kind not in ('zero','scale','replace') or type(self.call_index) is not int or self.call_index<0:
            raise ValueError('invalid intervention target/operator')
        if self.features is not None and (not isinstance(self.features,tuple) or not self.features or any(type(i) is not int or i<0 for i in self.features) or len(set(self.features))!=len(self.features)):
            raise ValueError('features must be unique nonnegative last-axis indices')
        if self.kind=='zero' and self.value is not None:raise ValueError('zero intervention has no replacement value')

    def record(self):
        import torch
        d=dict(module=self.module,kind=self.kind,call_index=self.call_index,features=None if self.features is None else list(self.features))
        if isinstance(self.value,torch.Tensor):d.update(value_kind='tensor',value_sha256=tensor_digest(self.value))
        elif self.kind!='zero':
            if type(self.value) not in (int,float) or not math.isfinite(self.value):raise ValueError('finite scalar or tensor intervention value required')
            if self.kind=='scale' and isinstance(self.value,torch.Tensor):raise ValueError('scale requires a scalar')
            d.update(value_kind='scalar',value=self.value)
        return d


def audit_intervention(model,inputs,intervention,*,runtime_id,declared_stateless,
                       seed=0,atol=0.,rtol=0.,output_kind='continuous',hypothesis=None):
    """Four same-input/same-RNG forwards: baseline, no-op, edit, baseline repeat.

    Model weights are not patched. Registered-state mutations invalidate evidence
    but are not automatically rolled back; use an owned/resettable stateless model.
    Input tensors are cloned per forward, hooks/modes/RNG are restored on exit.
    """
    import torch
    import numpy as np
    if (not isinstance(model,torch.nn.Module) or not isinstance(inputs,torch.Tensor) or not isinstance(intervention,Intervention)
        or not isinstance(runtime_id,str) or not runtime_id or declared_stateless is not True):
        raise ValueError('explicit stateless PyTorch tensor adapter/runtime identity required')
    if type(seed) is not int or seed<0 or output_kind not in ('continuous','logits'):
        raise ValueError('invalid RNG/output semantics')
    if hypothesis is not None and not isinstance(hypothesis,str):raise ValueError('hypothesis must be an explicitly unverified text label')
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in (atol,rtol)):raise ValueError('finite nonnegative control tolerances required')
    spec=intervention.record()
    if intervention.kind=='scale' and spec.get('value_kind')!='scalar':raise ValueError('scale requires scalar value')
    started=time.monotonic();modes=[(m,m.training) for m in model.modules()]
    cpu_rng=torch.get_rng_state();py_rng=random.getstate();np_rng=np.random.get_state()
    devices=sorted({t.device.index for t in list(model.parameters())+list(model.buffers())+[inputs] if t.is_cuda})
    cuda_rng={i:torch.cuda.get_rng_state(i) for i in devices}
    result=dict(format='wspl.intervention_evidence.v0',status='INVALID',evidence_tier='empirical_computational_intervention',
                causal_scope='specified tensor intervention in this computation on this input',
                universal_verification='not_established',semantic_grounding='not_established',physical_causation='not_established',
                intervention_distribution='not_validated',intervention=spec,hypothesis=hypothesis,
                instrumentation=dict(version='wspl.interventions.v0',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                     protocol=['baseline','noop','intervention','repeat'],state_policy='registered-state integrity guard; no state rollback'),
                runtime=dict(torch_version=torch.__version__,numpy_version=np.__version__,python=sys.version,platform=sys.platform,
                             controlled_cuda_devices=devices,cuda_version=torch.version.cuda,
                             declared_stateless=True,output_kind=output_kind,float32_matmul_precision=torch.get_float32_matmul_precision(),
                             deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),seed=seed,atol=atol,rtol=rtol),
                forward_calls=0,input=dict(shape=list(inputs.shape),dtype=str(inputs.dtype),device=str(inputs.device),sha256=tensor_digest(inputs)),runs={})
    hook=None
    try:
        model.eval();before=model_identity(model,runtime_id);result['model']=before
        module=model.get_submodule(intervention.module)
        target_value=intervention.value.detach().clone() if isinstance(intervention.value,torch.Tensor) else intervention.value
        def run(label):
            nonlocal hook
            # Seed CPU without torch.manual_seed's all-device side effect; only
            # explicitly snapshotted model/input CUDA streams are controlled.
            torch.random.default_generator.manual_seed(seed);random.seed(seed);np.random.seed(seed%(2**32))
            for device in devices:
                with torch.cuda.device(device):torch.cuda.manual_seed(seed)
            calls=0;captured={}
            def change(module,args,out):
                nonlocal calls
                current=calls;calls+=1
                if current!=intervention.call_index:return None
                captured['original']=_summary(out);copy=out.clone()
                if intervention.features is not None:
                    if out.ndim==0 or max(intervention.features)>=out.shape[-1]:raise ValueError('intervention feature outside site shape')
                    selector=(...,list(intervention.features))
                else:selector=Ellipsis
                if label=='intervention':
                    if intervention.kind=='zero':copy[selector]=0
                    elif intervention.kind=='scale':copy[selector]=copy[selector]*target_value
                    else:
                        if isinstance(target_value,torch.Tensor):
                            if target_value.shape!=copy[selector].shape:raise ValueError('replacement shape must match selected site exactly')
                            value=target_value.to(device=out.device,dtype=out.dtype)
                        else:value=target_value
                        copy[selector]=value
                captured['installed']=_summary(copy)
                return copy
            if label in ('noop','intervention'):hook=module.register_forward_hook(change)
            try:
                result['forward_calls']+=1
                with torch.inference_mode():output=model(inputs.detach().clone())
                summary=_summary(output)
            finally:
                if hook is not None:hook.remove();hook=None
            if label in ('noop','intervention') and 'original' not in captured:raise ValueError('target module invocation not reached')
            result['runs'][label]=dict(output=summary,site=captured)
            return output.detach().cpu().clone()
        a=run('baseline');noop=run('noop');changed=run('intervention');repeat=run('repeat')
        if any(x.shape!=a.shape or x.dtype!=a.dtype for x in (noop,changed,repeat)):raise ValueError('output schema changed; numeric effect adapter unsupported')
        after=model_identity(model,runtime_id);result['registered_state_unchanged']=before['sha256']==after['sha256']
        result['model_after_sha256']=after['sha256']
        result['controls']=dict(noop_matches=torch.allclose(a,noop,atol=atol,rtol=rtol),repeat_matches=torch.allclose(a,repeat,atol=atol,rtol=rtol))
        delta=changed.double()-a.double()
        result['effect']=dict(max_abs=float(delta.abs().max()),mean_abs=float(delta.abs().mean()))
        if output_kind=='logits':
            if a.ndim<1 or a.shape[-1]<2:raise ValueError('logit metrics require at least two last-axis classes')
            result['effect']['argmax_changed_fraction']=float((a.argmax(-1)!=changed.argmax(-1)).double().mean())
            result['effect']['baseline_min_top2_margin']=float((a.topk(2,dim=-1).values[...,0]-a.topk(2,dim=-1).values[...,1]).min())
        if not all(math.isfinite(x) for x in result['effect'].values()):
            result['effect']=None;raise ValueError('nonfinite difference metric; effect not established')
        if not result['registered_state_unchanged'] or not all(result['controls'].values()):result['status']='INVALID_CONTROL'
        else:result['status']='NO_OBSERVED_EFFECT' if torch.allclose(a,changed,atol=atol,rtol=rtol) else 'OBSERVED_EFFECT'
    except (ValueError,RuntimeError,TypeError,AttributeError) as exc:result.update(status='INVALID',reason=f'{type(exc).__name__}: {exc}')
    finally:
        if hook is not None:hook.remove()
        for module,mode in modes:module.training=mode
        torch.set_rng_state(cpu_rng);random.setstate(py_rng);np.random.set_state(np_rng)
        for device,state in cuda_rng.items():torch.cuda.set_rng_state(state,device)
        result['wall_seconds']=time.monotonic()-started
    return result


def replay_intervention(model,inputs,record,intervention=None):
    if intervention is None:
        s=record['intervention']
        if s.get('value_kind')=='tensor':raise ValueError('tensor replacement must be supplied for replay; hashes do not contain payloads')
        intervention=Intervention(s['module'],s['kind'],s['call_index'],None if s['features'] is None else tuple(s['features']),s.get('value'))
    r=record['runtime']
    replay=audit_intervention(model,inputs,intervention,runtime_id=record['model']['runtime_id'],declared_stateless=True,
                              seed=r['seed'],atol=r['atol'],rtol=r['rtol'],output_kind=r['output_kind'],hypothesis=record.get('hypothesis'))
    checks=dict(model=replay.get('model',{}).get('sha256')==record['model']['sha256'],inputs=replay['input']==record['input'],
                intervention=replay['intervention']==record['intervention'],runtime=replay['runtime']==record['runtime'],
                instrumentation=replay['instrumentation']==record['instrumentation'],status=replay['status']==record['status'],
                outputs={k:v['output']['sha256'] for k,v in replay['runs'].items()}=={k:v['output']['sha256'] for k,v in record['runs'].items()})
    return dict(exact_replay=all(checks.values()),checks=checks,evidence=replay)


def _validate_evidence(evidence):
    if (not isinstance(evidence,dict) or evidence.get('format')!='wspl.intervention_evidence.v0' or
        evidence.get('evidence_tier')!='empirical_computational_intervention' or
        evidence.get('status') not in ('INVALID','INVALID_CONTROL','OBSERVED_EFFECT','NO_OBSERVED_EFFECT') or
        any(evidence.get(k)!='not_established' for k in ('universal_verification','semantic_grounding','physical_causation'))):
        raise ValueError('log accepts empirical intervention evidence, not proof or world-causal claims')


def verify_log(path,expected_head=None):
    previous=None;records=[]
    for i,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
        d=json.loads(line)
        if set(d)!={'format','index','previous_hash','evidence','record_hash'} or d['format']!='wspl.intervention_log.v0' or d['index']!=i or d['previous_hash']!=previous:
            raise ValueError('invalid evidence chain')
        _validate_evidence(d['evidence'])
        body={k:v for k,v in d.items() if k!='record_hash'}
        if hashlib.sha256(_canonical(body)).hexdigest()!=d['record_hash']:raise ValueError('evidence record hash differs')
        previous=d['record_hash'];records.append(d)
    if expected_head is not None and previous!=expected_head:raise ValueError('evidence head differs from trusted anchor')
    return records


def append_evidence(path,evidence):
    """Single-writer integrity chain; no signature/authenticity guarantee."""
    _validate_evidence(evidence)
    path=Path(path);records=verify_log(path) if path.exists() else []
    body=dict(format='wspl.intervention_log.v0',index=len(records),previous_hash=None if not records else records[-1]['record_hash'],evidence=evidence)
    body['record_hash']=hashlib.sha256(_canonical(body)).hexdigest()
    with path.open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(body,sort_keys=True,allow_nan=False)+'\n')
    return body['record_hash']
