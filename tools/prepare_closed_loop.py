"""Prepare an unapproved closed-loop study; no learned predictions or fitting."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.executive_study import load_library
from skillforge.closed_loop_control import Episode,make_task,teacher,prompt
from skillforge.local_mechanism import calibration,author_prompt


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    from transformers import AutoTokenizer
    old_path=ROOT.parent/'WSPL-executive-v2/experiments/source_pins/neural_executive_v2.manifest.json'
    if digest(old_path)!='43dabbb5b31820586f368de2037e2673767c71eafa89c55833d2d1fa6b60f5fc':
        raise ValueError('Historical source identity changed')
    old=json.loads(old_path.read_text());config=old['runner_config']
    blocks,specs=load_library(config['library'])
    tokenizer=AutoTokenizer.from_pretrained(old['constructor_kwargs']['model_path'],local_files_only=True)
    lengths=[];outcomes=[]
    # Engineering scenes only, not the declared final panel seeds or learned policies.
    for i,n in enumerate((2,5,8,12)):
        for disturbance in ('none','slip','moved','blocked','combined'):
            task=make_task(100+i,n,disturbance,renamed=True)
            episode=Episode(task,blocks,specs)
            while not episode.finished:
                observation=episode.observation();action=teacher(observation)
                messages=[dict(role='user',content=prompt(observation)),dict(role='assistant',content=action)]
                text=tokenizer.apply_chat_template(messages,tokenize=False)
                length=len(tokenizer(text,add_special_tokens=False)['input_ids'])
                if length>2048:raise ValueError('Engineering record exceeds training context')
                lengths.append(length);episode.execute(action)
            outcome=episode.result(False)
            if not outcome['task_success']:raise ValueError('Engineering reference is infeasible')
            outcomes.append(dict(objects=n,disturbance=disturbance,ticks=outcome['ticks'],decisions=outcome['decisions']))
    replacement_spec=dict(specs['move'],gain=-specs['move']['gain'])
    text=tokenizer.apply_chat_template([dict(role='user',content=author_prompt(replacement_spec,calibration(replacement_spec)))],tokenize=False,add_generation_prompt=True)
    author_length=len(tokenizer(text,add_special_tokens=False)['input_ids'])
    if author_length+512>8192:raise ValueError('Author context exceeds declared limit')
    resources={}
    for path,h in config['resources'].items():
        if path==config['prior_adapter'] or path.endswith('EXECUTIVE_CURRICULUM_V2.md'):continue
        if digest(path)!=h:raise ValueError('Historical resource changed: '+path)
        resources[path]=h
    author=ROOT.parent/'WSPL-controller-author-v1/results/controller_author_v1/adapter_final.pt'
    previous=ROOT.parent/'WSPL-executive-v2/results/neural_executive_v2/adapter_final.pt'
    import torch
    checkpoint_identities={}
    for name,path,expected in [('author',author,'5de7ada4ce5d2c3c41614896e568cca4b91bc63991f5f4ce997a7e9404d1f3ae'),
                               ('previous',previous,'43dabbb5b31820586f368de2037e2673767c71eafa89c55833d2d1fa6b60f5fc')]:
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['manifest_sha256']!=expected or saved['base_hash']!='540b6c0285e9d26362156163cef352c1a27657dc7b0372d0c36404a6c5baddc1':
            raise ValueError('Historical checkpoint identity mismatch')
        checkpoint_identities[name]=dict(source=expected,base_hash=saved['base_hash'],sha256=digest(path))
        del saved
    proof=Path('C:/Users/kevin/skillforge-runtime/venv-wspl-match')
    extra=[old_path,author,previous,Path(sys.executable),ROOT/'docs/wspl/CLOSED_LOOP_V1.md',
           proof/'Scripts/python.exe',proof/'pyvenv.cfg']
    extra.extend(p for p in (proof/'Lib/site-packages/z3').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for path in extra:resources[str(path.resolve())]=digest(path)
    report=dict(status='passed',engineering_scenes=outcomes,max_engineering_record_tokens=max(lengths),
                author_prompt_tokens=author_length,author_token_cap=512,resources=len(resources),
                checkpoint_identities=checkpoint_identities,
                library={k:b.sha256 for k,b in blocks.items()},learned_predictions=0,training_launched=False)
    (ROOT/'docs/wspl/closed_loop_preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    spec=dict(model_id='compact-closed-loop-v1',source_roots=['src','experiments','tools','tests'],
              constructor=old['constructor'],constructor_kwargs=old['constructor_kwargs'],
              runner=dict(module='run_closed_loop',symbol='run',source='experiments/run_closed_loop.py'),
              runner_config=dict(resources=resources,library=config['library'],seeds=[113,227,341],
              train_rows=16384,dev_rows=256,epochs=2,replacement_updates=4096,wall_guard_seconds=36*3600,
              author_adapter=str(author.resolve()),previous_adapter=str(previous.resolve()),
              proof_python=str((proof/'Scripts/python.exe').resolve())))
    target=ROOT/'experiments/source_pins/compact_closed_loop_v1.spec.json'
    target.write_text(json.dumps(spec,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status='prepared',resources=len(resources),engineering_scenes=len(outcomes),
                         max_record=max(lengths),author_prompt=author_length,approved=False)))


if __name__=='__main__':main()
