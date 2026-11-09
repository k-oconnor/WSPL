"""Common bounded training service for every authoring interface."""
import time,os,random,json
from pathlib import Path
import torch
from .grid_copy import batch_tensors
from .capability_codec import encode,evaluate
from .capability_bridge import modules,attached,export


def loss(model,batch,cell_weights):
    ids,mask,labels=batch;logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
    active=labels[:,1:]!=-100;per=torch.nn.functional.cross_entropy(logits[:,:-1][active].float(),labels[:,1:][active],reduction='none')
    w=torch.tensor(cell_weights,device=per.device,dtype=per.dtype)
    return (per*w).sum()/w.sum()


def save_checkpoint(path,bank,opt,budget,source,rng,graphs):
    path=Path(path);tmp=path.with_suffix('.tmp')
    try:
        torch.save(dict(modules=bank.state_dict(),graphs=graphs,optimizer=opt.state_dict(),budget=budget.copy(),manifest_sha256=source,
            rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp);os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)


def train_chunk(study,plan,train,encoded,budget,out,source,check,update):
    bank=modules(plan);bank.train();opt=torch.optim.AdamW(bank.parameters(),lr=plan['recipe']['lr'],weight_decay=0,foreach=False)
    rng=random.Random(7301+budget['updates']);last_save=time.monotonic();graphs={k:m.graph for k,m in bank.items()}
    with attached(study.model,bank),(Path(out)/'training.jsonl').open('a') as log:
        for local in range(plan['recipe']['steps']):
            if budget['updates']>=8192 or budget['train_seconds']>=3600:break
            try:check()
            except BaseException:save_checkpoint(Path(out)/'last.pt',bank,opt,budget,source,rng,graphs);raise
            choices=[];items=[];cell_weights=[]
            for _ in range(4):
                if budget['updates']<512:key='copy4' if rng.random()<.5 else 'copy6'
                else:
                    z=rng.random();fraction=plan['recipe']['copy_fraction'];key='move' if z>=fraction else ('copy4' if z<fraction/2 else 'copy6')
                i=rng.randrange(len(train[key]));row=train[key][i];choices.append((key,i));items.append(encoded[key][i])
                cell_weights.extend([plan['recipe']['changed_cell_weight'] if key=='move' and a!=b else 1. for a,b in zip(row['input'],row['target'])])
            torch.cuda.synchronize();start=time.monotonic()
            batch=batch_tensors(items,study.tokenizer.eos_token_id,'cuda');opt.zero_grad(set_to_none=True);value=loss(study.model,batch,cell_weights)
            if not torch.isfinite(value):raise ValueError('nonfinite training')
            value.backward();norm=torch.nn.utils.clip_grad_norm_(bank.parameters(),1.,error_if_nonfinite=True)
            for group in opt.param_groups:group['lr']=plan['recipe']['lr']*min(1.,(local+1)/32)
            opt.step();torch.cuda.synchronize();budget['train_seconds']+=time.monotonic()-start;budget['updates']+=1;budget['targets']+=len(cell_weights)
            record=dict(budget,loss=float(value.detach()),grad_norm=float(norm),choices=choices)
            log.write(json.dumps(record)+'\n');log.flush();update(record)
            if time.monotonic()-last_save>=1800:save_checkpoint(Path(out)/'last.pt',bank,opt,budget,source,rng,graphs);last_save=time.monotonic()
    opt.zero_grad(set_to_none=True);save_checkpoint(Path(out)/'last.pt',bank,opt,budget,source,rng,graphs)
    result=export(plan,bank);del opt,bank;return result


def feedback(study,plan,dev,check):
    bank=modules(plan);bank.eval().requires_grad_(False);report={}
    with attached(study.model,bank):
        for name,rows in dev.items():report[name]=evaluate(study,rows,check)
    del bank;return report


def compact_feedback(report):
    # Selection is fixed by fixture order, never worst/best selected cases.
    return {k:dict(n=v['n'],exact=v['exact'],examples=v['records'][:8]) for k,v in report.items()}
