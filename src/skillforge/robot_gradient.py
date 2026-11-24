"""Fresh matched-data gradient control; called only by the approved study runner."""
import gc
import json
import math
import random
import time
import numpy as np
import torch
from .authoring_sft import adapter_state,restore_adapter,frozen_hash,checkpoint
from .fresh_robot import token_loss
from .robot_trials import CONDITIONS
from .teaching_training import state_hash
from .reliable_json import save


def schedule(rows, updates, seed):
    pools={c:[i for i,r in enumerate(rows) if r['condition']==c] for c in CONDITIONS}
    if any(not p for p in pools.values()):raise ValueError('All four supervised conditions required')
    rng=random.Random(seed)
    return [[rng.choice(pools[c]) for c in CONDITIONS for _ in range(2)] for _ in range(updates)]


def fit(model, parent, rows, config, out, identity, base_hash, guard, report):
    out.mkdir();restore_adapter(model,parent)
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    torch.manual_seed(config['gradient_seed']);torch.cuda.manual_seed_all(config['gradient_seed'])
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=config['learning_rate'],weight_decay=.01,foreach=False)
    batches=schedule(rows,config['gradient_updates'],config['gradient_seed'])
    save(out/'schedule.json',batches)
    step=targets=0;updating=False;last=time.monotonic()
    def preserve():
        checkpoint(out/'rolling.pt',model,optimizer,dict(step=step,targets=targets,manifest_sha256=identity,
            base_hash=base_hash,parent_sha256=state_hash(parent),purpose='fresh_adaptation_control'))
    try:
        with (out/'training.jsonl').open('x') as log:
            for batch in batches:
                guard();optimizer.zero_grad(set_to_none=True);value=0.
                for i in batch:
                    guard();loss=token_loss(model,rows[i]['record'])
                    if not torch.isfinite(loss):raise ValueError('Nonfinite gradient-control loss')
                    value+=float(loss.detach());(loss/len(batch)).backward();del loss
                norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
                progress=max(0.,(step+1-16)/max(1,config['gradient_updates']-16))
                lr=config['learning_rate']*min(1.,(step+1)/16)*(.1+.9*.5*(1+math.cos(math.pi*progress)))
                for group in optimizer.param_groups:group['lr']=lr
                updating=True;optimizer.step();updating=False
                step+=1;targets+=len(batch)
                item=dict(step=step,targets=targets,nll=value/len(batch),lr=lr,grad_norm=float(norm))
                log.write(json.dumps(item)+'\n');log.flush();report(status='gradient_control',**item)
                if time.monotonic()-last>=1800:preserve();last=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        if frozen_hash(model)!=base_hash:raise ValueError('Frozen base changed during control')
        preserve()
        state=adapter_state(model)
        np.savez(out/'candidate.npz',**{k:v.numpy() for k,v in state.items()})
        save(out/'frozen.json',dict(step=step,targets=targets,manifest_sha256=identity,
                                  base_hash=base_hash,adapter_sha256=state_hash(state)))
        return out/'candidate.npz'
    except BaseException:
        if not updating:optimizer.zero_grad(set_to_none=True);preserve()
        raise
    finally:
        del optimizer;gc.collect();torch.cuda.empty_cache()
