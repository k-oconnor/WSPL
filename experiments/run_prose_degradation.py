"""Small frozen prose-loss and completion comparison; no fitting."""
import json,math,time
from pathlib import Path
from contextlib import nullcontext
from skillforge.arc_eval import model_hash
from skillforge.capability_bridge import modules,attached
from skillforge.go_replay import sha
from skillforge.reliable_json import save,status


def next_token_loss(model,ids):
    import torch
    with torch.inference_mode():
        logits=model(input_ids=ids,use_cache=False).logits
        loss=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='sum')
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    return float(loss),ids.shape[0]*(ids.shape[1]-1)


def run(constructor,output_dir,config):
    import torch
    out=Path(output_dir);start=time.monotonic();state=dict(status='loading')
    receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('source receipt')
    state['manifest_sha256']=receipt['manifest_sha256']
    def verify():
        for p,h in config['resources'].items():
            if sha(p)!=h:raise ValueError('resource changed: '+p)
    def update():
        if time.monotonic()-start>config['wall_guard_seconds']:raise RuntimeError('wall guard')
        state['active_seconds']=time.monotonic()-start;status(out/'status.json',state)
    try:
        update();verify();study=constructor();before=model_hash(study.model)
        bank=modules(json.loads(Path(config['adapter']).read_text())).eval().requires_grad_(False);bank_hash=model_hash(bank)
        rows=json.loads(Path(config['data']).read_text());state['results']={}
        for arm in ('foundation','wspl_0'):
            scores=[];completions=[]
            with nullcontext() if arm=='foundation' else attached(study.model,bank):
                for i,row in enumerate(rows):
                    ids=torch.tensor([study.tokenizer.encode(row['text'],add_special_tokens=False)[:513]],device='cuda')
                    if ids.shape[1]<2:raise ValueError('short text')
                    loss,n=next_token_loss(study.model,ids);scores.append(dict(id=row['id'],loss_sum=loss,targets=n))
                    state.update(status='scoring',arm=arm,completed=i+1,total=len(rows));update()
                for prompt in config['prompts']:
                    ids=study.tokenizer(prompt,return_tensors='pt').to('cuda')
                    with torch.inference_mode():generated=study.model.generate(**ids,max_new_tokens=64,do_sample=False,use_cache=True,pad_token_id=study.tokenizer.eos_token_id)
                    completions.append(dict(prompt=prompt,continuation=study.tokenizer.decode(generated[0,ids['input_ids'].shape[1]:],skip_special_tokens=False)))
                    update()
            n=sum(s['targets'] for s in scores);nll=sum(s['loss_sum'] for s in scores)/n
            state['results'][arm]=dict(nll=nll,perplexity=math.exp(nll),targets=n,documents=scores,completions=completions)
            if model_hash(study.model)!=before or model_hash(bank)!=bank_hash:raise ValueError('weights changed')
        a=state['results']['foundation'];b=state['results']['wspl_0']
        state.update(status='complete',delta_nll=b['nll']-a['nll'],perplexity_ratio=b['perplexity']/a['perplexity'])
        verify()
    except BaseException as e:
        state.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:
        state['active_seconds']=time.monotonic()-start;status(out/'status.json',state);save(out/'results.json',state)
