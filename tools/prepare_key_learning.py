"""Prepare bounded recipe selection and learning curves; no scientific calls/fits."""
import json
from pathlib import Path
import shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from prepare_arc_baseline import sha
from source_consistency import inventory,verify,write_json
from skillforge.heat_curve_data import build_splits
from skillforge.arc_eval import encode_choices
from skillforge.codex_one_shot import preflight
from run_key_learning import make_prompt
from transformers import AutoTokenizer


def main():
    check=json.loads((ROOT/'docs/wspl/key_learning_preflight.json').read_text());assert check['status']=='passed'
    base=ROOT.parent/'Skillforge-attention-probe-resume-v1/results/attention_probe_resume_v1'
    previous=json.loads((base/'results.json').read_text());assert previous['status']=='complete'
    old=json.loads((ROOT/'experiments/source_pins/arc_baseline_v1.manifest.json').read_text());resources=dict(old['runner_config']['resources'])
    folder=ROOT/'data/key_learning_v1';folder.mkdir(exist_ok=False);splits=build_splits()
    tokenizer=AutoTokenizer.from_pretrained(old['constructor_kwargs']['model_path'],local_files_only=True,trust_remote_code=False)
    lengths={}
    for split,rows in splits.items():
        write_json(folder/(split+'.json'),rows);lengths[split]=0
        for row in rows:
            encoded=encode_choices(tokenizer,row);assert all(len(ids)-start==1 for ids,start in encoded)
            lengths[split]=max(lengths[split],max(len(ids) for ids,start in encoded))
    guide=ROOT/'docs/wspl/KEY_LEARNING_INTERFACE_V1.md';protocol=ROOT/'docs/wspl/KEY_LEARNING_V1.md'
    local=ROOT/'artifacts/local_paths.json';cli=Path(json.loads(local.read_text())['codex_cli'])
    catalog=ROOT/'data/astra_native_v1/catalog_frozen.json';instructions=ROOT/'data/astra_native_v1/instructions.md'
    text=make_prompt(guide.read_text(),dict(training_sample=splits['train'][:16],note='Runtime adds measured development metrics; final cases never supplied.'),[],0)
    p=folder/'author.prompt.txt';p.write_text(text,encoding='utf-8');pf=folder/'preflight';pf.mkdir();(pf/'empty').mkdir();preflight(cli,catalog,instructions,pf,text)
    bound=[guide,protocol,local,cli,catalog,instructions,p,ROOT/'docs/wspl/key_learning_preflight.json',base/'retrieval/final.npz',base/'results.json',base/'source_receipt.json']
    bound += [folder/(s+'.json') for s in splits]
    for p in bound:resources[str(p)]=sha(p)
    for p,h in resources.items():
        if sha(p)!=h:raise ValueError('resource changed: '+p)
    spec=dict(schema=1,model_id='key-learning-v1',source_roots=['src','experiments','tests','tools'],constructor=old['constructor'],constructor_kwargs=old['constructor_kwargs'],
        runner=dict(module='run_key_learning',source='experiments/run_key_learning.py',symbol='run'),
        runner_config=dict(resources=resources,parent_bank=str(base/'retrieval/final.npz'),model_sha256=previous['model_sha256'],
            train=str(folder/'train.json'),development=str(folder/'development.json'),final=str(folder/'final.json'),guide=str(guide),
            local_paths=str(local),catalog=str(catalog),instructions=str(instructions),wall_guard_seconds=14400,optimizer_steps=1000))
    rel='experiments/source_pins/key_learning_v1'
    for suffix in ('.spec.json','.manifest.json'):
        if (ROOT/(rel+suffix)).exists():raise ValueError('refuse overwrite')
    write_json(ROOT/(rel+'.spec.json'),spec);manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    isolated=ROOT.parent/'Skillforge-key-learning-v1';isolated.mkdir(exist_ok=False)
    for name in ('src','experiments','tests','tools'):shutil.copytree(ROOT/name,isolated/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',isolated/'AGENTS.md');h=sha(ROOT/(rel+'.manifest.json'));verify(isolated,isolated/(rel+'.manifest.json'),h)
    report=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),isolated=str(isolated),
        examples={k:len(v) for k,v in splits.items()},max_encoded_lengths=lengths,max_author_calls=4,max_key_fits=28,new_scientific_fits=0,new_benchmark_predictions=0,new_author_calls=0)
    write_json(ROOT/'docs/wspl/key_learning_v1_preparation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
