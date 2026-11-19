"""Matched warm-start continuation; never reinitialize the learned LoRA."""
import gc
import hashlib
import json
import math
import random
import time
import torch
from .authoring_sft import adapter_state,restore_adapter,frozen_hash,encode,loss_sum,checkpoint


def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def draw_schedule(ordinary_count,teacher_count,arm,updates=128):
    if arm not in ('rehearsal','taught') or ordinary_count<1 or teacher_count<1:raise ValueError('Invalid corpus/arm')
    old=random.Random(881);new=random.Random(991)
    batches=[]
    for _ in range(updates):
        indices=[old.randrange(ordinary_count) for _ in range(8)]
        extra=[new.randrange(teacher_count) for _ in range(4)]
        batches.append([('ordinary',i) for i in indices] if arm=='rehearsal' else
                       [('ordinary',i) for i in indices[:4]]+[('teacher',i) for i in extra])
    return batches


def fit(model,tokenizer,parent,ordinary,teacher,arm,out,manifest,guard,report,updates=128):
    out.mkdir();restore_adapter(model,parent['adapter'])
    expected=state_hash(parent['adapter'])
    if state_hash(adapter_state(model))!=expected:raise ValueError('Warm-start identity failure')
    torch.manual_seed(143);torch.cuda.manual_seed_all(143)
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    data={k:[encode(tokenizer,row['messages'],4096) for row in rows] for k,rows in [('ordinary',ordinary),('teacher',teacher)]}
    schedule=draw_schedule(len(ordinary),len(teacher),arm,updates)
    (out/'schedule.json').write_text(json.dumps(schedule))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.00003,weight_decay=.01,foreach=False)
    step=targets=0;last=time.monotonic();updating=False;seen={'ordinary':0,'teacher':0}
    def preserve():
        checkpoint(out/'rolling.pt',model,optimizer,dict(step=step,targets=targets,arm=arm,manifest_sha256=manifest,
                   base_hash=parent['base_hash'],parent_adapter_sha256=expected,seen=seen))
    try:
        with (out/'training.jsonl').open('x') as log:
            for batch in schedule:
                guard();optimizer.zero_grad(set_to_none=True)
                count=sum(data[k][i]['targets'] for k,i in batch);value=0.
                for k,i in batch:
                    loss=loss_sum(model,data[k][i]);value+=float(loss.detach())
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    (loss/count).backward();del loss;seen[k]+=1
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True)
                fraction=max(0.,(step+1-16)/max(1,updates-16))
                lr=.00003*min(1.,(step+1)/16)*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
                for group in optimizer.param_groups:group['lr']=lr
                updating=True;optimizer.step();updating=False;optimizer.zero_grad(set_to_none=True)
                step+=1;targets+=count
                record=dict(arm=arm,step=step,total=updates,nll=value/count,grad_norm=float(norm),lr=lr,targets=targets,seen=dict(seen))
                log.write(json.dumps(record)+'\n');log.flush();report(status='training',**record)
                if time.monotonic()-last>=1800:preserve();last=time.monotonic()
        if frozen_hash(model)!=parent['base_hash']:raise ValueError('Base changed')
        preserve();state=adapter_state(model)
        torch.save(dict(adapter=state,base_hash=parent['base_hash'],manifest_sha256=manifest,parent_adapter_sha256=expected,
                        adapter_sha256=state_hash(state),arm=arm,step=step,targets=targets,seen=seen),out/'adapter_final.pt')
    except BaseException:
        if not updating:optimizer.zero_grad(set_to_none=True);preserve()
        raise
    finally:
        del optimizer;gc.collect();torch.cuda.empty_cache()
