"""Read-only full-model acquisition queries, independent of final/dev evaluators."""
import time
from pathlib import Path
import numpy as np
import torch
from .authoring_sft import adapter_state, restore_adapter
from .robot_codec import CODES
from .reliable_json import save
from .wspl_lora import identity


def validate_request(request, row_count, projections):
    if set(request) != {'tool','candidate','rows','projections','tokens','weights'} or request['tool'] != 'inspect':
        raise ValueError('Inspect requires tool,candidate,rows,projections,tokens,weights')
    if request['candidate'] is not None and not isinstance(request['candidate'],str):
        raise ValueError('Candidate is null for original parent or a submission path')
    rows=request['rows']
    if rows=='all': rows=list(range(row_count))
    if not isinstance(rows,list) or not rows or len(rows)>row_count or any(type(i)!=int or not 0<=i<row_count for i in rows) or len(set(rows))!=len(rows):
        raise ValueError('Unique acquisition row indices required')
    names=request['projections']
    if not isinstance(names,list) or len(names)>4 or any(not isinstance(n,str) or n not in projections for n in names) or len(set(names))!=len(names):
        raise ValueError('Choose zero to four known projection names')
    if request['tokens'] not in ('last','all') or type(request['weights'])!=bool:
        raise ValueError('tokens must be last/all and weights boolean')
    if request['tokens']=='all' and len(rows)>32:
        raise ValueError('Full-sequence capture permits at most32 acquisition rows')
    return rows,names


def inspect_acquisition(model, tokenizer, rows, bank, request, out, guard,
                        max_elements=64_000_000, max_seconds=300):
    """No arbitrary text, paths, labels or executable callback supplied by author."""
    projections={n.rsplit('.lora_',1)[0] for n in bank}
    indices,names=validate_request(request,len(rows),projections)
    out=Path(out);out.mkdir()
    records=[rows[i]['record']['ids'] for i in indices]
    if any(not r or len(r)>2048 for r in records):raise ValueError('Invalid acquisition context')
    modules={n:model.get_submodule(n) for n in names}
    capture_tokens=sum(len(r) for r in records) if request['tokens']=='all' else len(indices)
    # Account before creating output arrays; permit arbitrary sites, not arbitrary size.
    dims=sum(m.base.in_features+m.base.out_features for m in modules.values())
    hidden=model.config.hidden_size
    elements=capture_tokens*dims+len(indices)*(hidden+len(CODES)+12)+hidden*(1+len(CODES))
    if request['weights']:
        elements+=sum(m.base.weight.numel()+(0 if m.base.bias is None else m.base.bias.numel()) for m in modules.values())
    if elements>max_elements:raise ValueError('Query exceeds64M output elements; reduce rows/projections or omit weights')
    code_ids=[tokenizer(c,add_special_tokens=False)['input_ids'] for c in CODES]
    if any(len(x)!=1 for x in code_ids):raise RuntimeError('Action token contract changed')
    code_ids=[x[0] for x in code_ids]
    original=adapter_state(model);was_training=model.training
    checkpointing=getattr(model,'is_gradient_checkpointing',False)
    handles=[];captured={};started=time.monotonic()
    def check():
        guard()
        if time.monotonic()-started>max_seconds:raise RuntimeError('Acquisition query wall guard exceeded')
    def capture(key, tensor, full=False):
        value=tensor[0] if full else tensor[0,-1]
        captured[key]=value.detach().float().cpu().numpy().copy()
    arrays={};summary=[]
    try:
        restore_adapter(model,{k:torch.from_numpy(v) for k,v in bank.items()})
        model.eval();model.gradient_checkpointing_disable()
        for j,(name,module) in enumerate(modules.items()):
            def pre(m,args,j=j):capture(f'p{j}_input',args[0],request['tokens']=='all')
            def post(m,args,value,j=j):capture(f'p{j}_output',value,request['tokens']=='all')
            handles.extend([module.register_forward_pre_hook(pre),module.register_forward_hook(post)])
        handles.append(model.model.norm.register_forward_pre_hook(lambda m,args:capture('residual',args[0])))
        for position,index in enumerate(indices):
            check();captured.clear()
            ids=torch.tensor([records[position]],device=next(model.parameters()).device,dtype=torch.long)
            with torch.inference_mode():
                logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1).logits[0,-1].float()
                if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite full-vocabulary logits')
                target=code_ids[CODES.index(rows[index]['code'])]
                nll=float(torch.logsumexp(logits,0)-logits[target])
                values,top=torch.topk(logits,5)
            for key,value in captured.items():arrays[f'row{index}_{key}']=value
            arrays[f'row{index}_action_logits']=logits[code_ids].cpu().numpy()
            summary.append(dict(row=index,label=rows[index]['code'],target_nll=nll,
                predicted_token=int(top[0]),predicted_text=tokenizer.decode([int(top[0])],skip_special_tokens=False),
                correct=int(top[0])==target,top_tokens=top.cpu().tolist(),top_logits=values.cpu().tolist()))
        if request['weights']:
            for j,module in enumerate(modules.values()):
                arrays[f'p{j}_base_weight']=module.base.weight.detach().float().cpu().numpy().copy()
                if module.base.bias is not None:arrays[f'p{j}_base_bias']=module.base.bias.detach().float().cpu().numpy().copy()
        arrays['final_norm_scale']=model.model.norm.weight.detach().float().cpu().numpy().copy()
        arrays['action_head_rows']=model.lm_head.weight[code_ids].detach().float().cpu().numpy().copy()
        if not all(np.isfinite(x).all() for x in arrays.values()):raise RuntimeError('Nonfinite diagnostic output')
        check();np.savez(out/'arrays.npz',**arrays)
        info=dict(status='complete',adapter_id=identity(bank),rows=summary,projections={f'p{j}':n for j,n in enumerate(names)},
                  tokens=request['tokens'],weights=request['weights'],array_shapes={k:list(v.shape) for k,v in arrays.items()},
                  mean_target_nll=float(np.mean([x['target_nll'] for x in summary])),
                  action_correct=sum(x['correct'] for x in summary),examples=len(summary),seconds=time.monotonic()-started)
        save(out/'diagnostics.json',info)
        return info
    finally:
        for h in handles:h.remove()
        restore_adapter(model,original)
        model.train(was_training)
        if checkpointing:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
