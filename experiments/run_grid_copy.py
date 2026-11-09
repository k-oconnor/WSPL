"""Bounded full-model copy fine-tuning; movement training is not included."""
import json,time,random
from pathlib import Path
import torch
from skillforge.grid_copy import encode_copy,batch_tensors,copy_loss,checkpoint
from skillforge.grid_io import generate
from skillforge.arc_eval import model_hash
from skillforge.go_replay import sha
from skillforge.reliable_json import save,status


def run(constructor,output_dir,config):
    out=Path(output_dir);receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('receipt')
    def resources():
        for p,h in config['resources'].items():
            if sha(p)!=h:raise ValueError('resource changed: '+p)
    resources();start=time.monotonic();state=dict(status='loading',step=0,manifest_sha256=receipt['manifest_sha256'])
    study=None;optimizer=None;step=0;rng=random.Random(211);order=[];last_save=start
    def snapshot():state['active_seconds']=time.monotonic()-start;status(out/'status.json',state)
    def stopped():
        p=out/'control.json'
        return time.monotonic()-start>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    def save_checkpoint():
        checkpoint(out/'last.pt',study.model,optimizer,step,receipt['manifest_sha256'],order,rng)
    try:
        study=constructor()
        if model_hash(study.model)!=config['model_sha256']:raise ValueError('base identity')
        data=json.loads(Path(config['data']).read_text());demos=data['demonstrations']
        encoded=[encode_copy(study.tokenizer,r['input'],demos) for r in data['train']]
        order=list(range(len(encoded)));rng.shuffle(order)
        study.model.requires_grad_(True);study.model.train();torch.manual_seed(211)
        optimizer=torch.optim.AdamW(study.model.parameters(),lr=config['lr'],weight_decay=0,foreach=False)
        state.update(status='training',total_steps=config['steps']);snapshot()
        with (out/'training.jsonl').open('w') as log:
            for step in range(1,config['steps']+1):
                if stopped():step-=1;state['status']='paused';save_checkpoint();return
                offset=(step-1)*config['batch_size']
                if offset+config['batch_size']>len(order):raise ValueError('training budget exceeds unique shuffled pool')
                batch=batch_tensors([encoded[i] for i in order[offset:offset+config['batch_size']]],study.tokenizer.eos_token_id,'cuda')
                optimizer.zero_grad(set_to_none=True);loss=copy_loss(study.model,batch)
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(study.model.parameters(),1.,error_if_nonfinite=True)
                lr=config['lr']*min(1.,step/32)
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.step()
                record=dict(step=step,nll=float(loss.detach()),grad_norm=float(norm),lr=lr,targets=step*config['batch_size']*16,active_seconds=time.monotonic()-start)
                log.write(json.dumps(record)+'\n');log.flush();state.update(record);snapshot()
                if time.monotonic()-last_save>=1800:save_checkpoint();last_save=time.monotonic()
        optimizer.zero_grad(set_to_none=True);save_checkpoint()
        study.model.eval().requires_grad_(False);frozen=model_hash(study.model)
        # Holdouts are first evaluated only after the fixed training budget and checkpoint freeze.
        summary={};state['status']='evaluating'
        for split in ('heldout','historical_copy'):
            summary[split]={}
            for mode in ('palette_only','unconstrained'):
                records=[];state.update(split=split,mode=mode,completed=0);snapshot()
                with (out/(split+'_'+mode+'.jsonl')).open('w') as f:
                    for row in data[split]:
                        if stopped():state['status']='incomplete';return
                        answer=generate(study,row['input'],'copy',demos,mode)
                        record=dict(answer,input=row['input'],exact=answer['output']==row['input'])
                        f.write(json.dumps(record)+'\n');f.flush();records.append(record);state['completed']=len(records);snapshot()
                summary[split][mode]=dict(n=len(records),exact=sum(r['exact'] for r in records),valid=sum(r['format_valid'] for r in records))
                save(out/'summary.json',summary)
        if model_hash(study.model)!=frozen:raise ValueError('evaluation mutated checkpoint')
        resources();state.update(status='complete',summary=summary,trained_model_sha256=frozen,
            copy_gate_passed=summary['heldout']['palette_only']['exact']>=244 and summary['historical_copy']['palette_only']['exact']>=61,
            next_step='Separate approval required for movement adapter training; no automatic follow-on.',peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:state.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:snapshot();save(out/'results.json',state)
