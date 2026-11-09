"""Adapter-only reachability; all evaluation is after the fixed training budget."""
import json,time,random
from pathlib import Path
from contextlib import nullcontext
import torch
from skillforge.grid_movement import encode_task
from skillforge.grid_copy import batch_tensors,copy_loss,checkpoint
from skillforge.grid_io import generate
from skillforge.relu_adapter import ResidualReLU,attached
from skillforge.arc_eval import model_hash
from skillforge.go_replay import sha
from skillforge.reliable_json import save,status


def run(constructor,output_dir,config):
    out=Path(output_dir);receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('receipt')
    def resources():
        for p,h in config['resources'].items():
            if sha(p)!=h:raise ValueError('resource changed '+p)
    resources();start=time.monotonic();state=dict(status='loading',step=0,manifest_sha256=receipt['manifest_sha256'])
    rng=random.Random(347);step=0;last_save=start
    def snapshot():state['active_seconds']=time.monotonic()-start;status(out/'status.json',state)
    def stop():
        p=out/'control.json'
        return time.monotonic()-start>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    try:
        study=constructor();before=model_hash(study.model);data=json.loads(Path(config['data']).read_text());demos=data['demonstrations']
        adapter=ResidualReLU(seed=173).to('cuda');assert sum(p.numel() for p in adapter.parameters())==37472
        opt=torch.optim.AdamW(adapter.parameters(),lr=config['lr'],weight_decay=0,foreach=False)
        movement=[encode_task(study.tokenizer,r,'movement',demos) for r in data['train']]
        copying=[encode_task(study.tokenizer,r,'copy',demos) for r in data['copy_train']]
        def save_checkpoint():
            checkpoint(out/'last.pt',adapter,opt,step,receipt['manifest_sha256'],[],rng)
        state.update(status='training',total_steps=config['steps']);snapshot()
        with attached(study.model,adapter), (out/'training.jsonl').open('w') as log:
            for step in range(1,config['steps']+1):
                if stop():step-=1;save_checkpoint();state['status']='paused';return
                indices=[rng.randrange(len(movement)) for _ in range(3)];copy_index=rng.randrange(len(copying))
                batch=batch_tensors([movement[i] for i in indices]+[copying[copy_index]],study.tokenizer.eos_token_id,'cuda')
                opt.zero_grad(set_to_none=True);loss=copy_loss(study.model,batch)
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True)
                lr=config['lr']*min(1.,step/32)
                for group in opt.param_groups:group['lr']=lr
                opt.step()
                record=dict(step=step,nll=float(loss.detach()),grad_norm=float(norm),lr=lr,movement_indices=indices,copy_index=copy_index,targets=step*64,active_seconds=time.monotonic()-start)
                log.write(json.dumps(record)+'\n');log.flush();state.update({k:v for k,v in record.items() if k not in ('movement_indices','copy_index')});snapshot()
                if time.monotonic()-last_save>=1800:save_checkpoint();last_save=time.monotonic()
        opt.zero_grad(set_to_none=True);save_checkpoint();adapter.eval().requires_grad_(False)
        if model_hash(study.model)!=before or any(p.grad is not None for p in study.model.parameters()):raise ValueError('base was modified')
        adapter_hash=model_hash(adapter);summary={};state['status']='evaluating'
        # Frozen baseline and final adapter use identical panels. No selection on scores.
        for condition in ('foundation','adapter'):
            summary[condition]={}
            with (attached(study.model,adapter) if condition=='adapter' else nullcontext()):
                for split in ('training_diagnostic','familiar','novel','copy_domain','copy_random'):
                    task='copy' if split.startswith('copy_') else 'movement';summary[condition][split]={}
                    for mode in ('palette_only','unconstrained'):
                        records=[];state.update(condition=condition,split=split,mode=mode,completed=0);snapshot()
                        with (out/f'{condition}_{split}_{mode}.jsonl').open('w') as f:
                            for row in data[split]:
                                if stop():state['status']='incomplete';return
                                answer=generate(study,row['input'],task,demos,mode);target=row['input'] if task=='copy' else row['output']
                                record=dict(answer,input=row['input'],target=target,exact=answer['output']==target)
                                records.append(record);f.write(json.dumps(record)+'\n');f.flush();state['completed']=len(records);snapshot()
                        summary[condition][split][mode]=dict(n=len(records),exact=sum(r['exact'] for r in records),valid=sum(r['format_valid'] for r in records))
                        save(out/'summary.json',summary)
        if model_hash(study.model)!=before or model_hash(adapter)!=adapter_hash:raise ValueError('evaluation mutation')
        resources();passed=all(summary['adapter'][s]['palette_only']['exact']>=48 and summary['adapter'][s]['palette_only']['exact']-summary['foundation'][s]['palette_only']['exact']>=16 for s in ('familiar','novel'))
        retained=all(summary['adapter'][s]['palette_only']['exact']>=61 and summary['adapter'][s]['palette_only']['exact']>=summary['foundation'][s]['palette_only']['exact']-3 for s in ('copy_domain','copy_random'))
        state.update(status='complete',summary=summary,foundation_sha256=before,adapter_sha256=adapter_hash,reachability_gate_passed=passed and retained,
            movement_gate_passed=passed,retention_gate_passed=retained,peak_allocated_bytes=torch.cuda.max_memory_allocated(),next_step='No automatic authoring or follow-on training.')
    except BaseException as e:state.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:snapshot();save(out/'results.json',state)
