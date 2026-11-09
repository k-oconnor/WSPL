"""Bounded native weight-plus-attention authoring contracts."""
from contextlib import contextmanager
import numpy as np
from .arc_eval import encode_choices


def validate(bank,reference):
    if set(bank)!=set(reference):raise ValueError('tensor names differ')
    for name,old in reference.items():
        a=bank[name]
        if a.shape!=old.shape or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('shape/dtype/finite')
        if name not in ('K','V','W27') and a.tobytes()!=old.tobytes():raise ValueError('read-only '+name)
    for name in ('K','V'):
        a=bank[name].copy();a[2,0]=reference[name][2,0]
        if a.tobytes()!=reference[name].tobytes():raise ValueError('only layer26/group0 editable')
        if np.linalg.norm(bank[name][2,0].astype(np.float64))>10*float(reference[name+'_scale'][2,0])+1e-5:raise ValueError('slot norm')
    delta=bank['W27'].astype(np.float64)-reference['W27'].astype(np.float64)
    if np.linalg.norm(delta)>.05*np.linalg.norm(reference['W27'].astype(np.float64))+1e-7:raise ValueError('weight norm')


@contextmanager
def weight_edit(model,bank,enabled=True):
    import torch
    weight=model.model.layers[27].self_attn.o_proj.weight
    old=weight.detach().clone()
    try:
        if enabled:
            with torch.no_grad():weight.copy_(torch.from_numpy(bank['W27']).to(weight.device))
        yield
    finally:
        with torch.no_grad():weight.copy_(old)
        if not torch.equal(weight,old):raise RuntimeError('weight restoration failed')


@contextmanager
def capture_boundary(model,position):
    result={};handles=[]
    def residual(name):
        def hook(module,args,out):
            x=out[0] if isinstance(out,tuple) else out
            result.setdefault(name,x[0,position].detach().cpu().float().numpy())
        return hook
    def inputs(module,args):result.setdefault('o27_input',args[0][0,position].detach().cpu().float().numpy())
    handles.append(model.model.layers[26].register_forward_hook(residual('layer26_out')))
    handles.append(model.model.layers[27].self_attn.o_proj.register_forward_pre_hook(inputs))
    handles.append(model.model.layers[27].register_forward_hook(residual('layer27_out')))
    try:yield result
    finally:
        for h in handles:h.remove()


def make_reference(study,parent,rows):
    from .arc_attention import score_boundary
    bank={k:parent[k].copy() for k in ('K','V','alpha','K_scale','V_scale')}
    bank['W27']=study.model.model.layers[27].self_attn.o_proj.weight.detach().cpu().float().numpy().copy()
    bank['original_W27']=bank['W27'].copy()
    bank['final_norm']=study.model.model.norm.weight.detach().cpu().float().numpy().copy()
    diagnostics={};baseline=[];initial=[];token_ids=set()
    for row in rows:
        encoded=encode_choices(study.tokenizer,row);position=encoded[0][1]-1
        for ids,start in encoded:token_ids.update(ids[start:])
        for label,mode in [('base','off'),('asp','retrieval')]:
            with capture_boundary(study.model,position) as captured:record=score_boundary(study,row,bank,mode)
            (baseline if label=='base' else initial).append(record)
            for name,a in captured.items():diagnostics.setdefault(label+'_'+name,[]).append(a)
    for name,a in diagnostics.items():bank[name]=np.stack(a).astype(np.float32)
    ids=sorted(token_ids);bank['answer_token_ids']=np.asarray(ids,np.float32)
    bank['answer_embeddings']=study.model.get_output_embeddings().weight.detach().cpu().numpy()[ids].astype(np.float32)
    validate(bank,bank)
    return bank,baseline,initial


def metrics(records,baseline,rows):
    base={r['id']:r for r in baseline};groups={r['id']:r['group'] for r in rows}
    return dict(n=len(records),correct=sum(r['correct'] for r in records),
                corrected=[r['id'] for r in records if r['correct'] and not base[r['id']]['correct']],
                broken=[r['id'] for r in records if not r['correct'] and base[r['id']]['correct']],
                groups={g:dict(n=sum(groups[r['id']]==g for r in records),correct=sum(r['correct'] for r in records if groups[r['id']]==g)) for g in sorted(set(groups.values()))})


def prompt(guide,packet,history,step):
    import json
    s=guide+'\nACQUISITION PACKET:\n'+json.dumps(packet)+'\nVISIBLE TRANSCRIPT:\n'+json.dumps(history)
    s+=f'\nResponse {step+1}/6. Return one fenced python program or FINAL.'
    if len(s.encode())>800000:raise ValueError('prompt limit')
    return s
