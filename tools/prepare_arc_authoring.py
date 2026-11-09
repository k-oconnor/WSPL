"""Freeze resources and prompts for an unapproved authoring screen; no model calls."""
import json
import hashlib
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import torch
from safetensors.torch import load_file
from skillforge.smollm_adapter import TARGETS,initial_bank
from skillforge.arc_authoring import prompt,statistics
from skillforge.arc_eval import validate_row,encode_choices
from skillforge.codex_one_shot import preflight
from transformers import AutoTokenizer
from source_consistency import inventory,write_json,verify
from prepare_arc_baseline import sha


def main():
    baseline_root=ROOT.parent/'Skillforge-arc-baseline-v1/results/arc_baseline_v1'
    baseline= json.loads((baseline_root/'results.json').read_text())
    assert baseline['status']=='complete' and baseline['suitable_for_authoring']
    check=json.loads((ROOT/'docs/wspl/arc_authoring_preflight.json').read_text())
    assert check['status']=='passed' and check['three_arm_effective_weights_bit_identical']
    old=json.loads((baseline_root/'approved_manifest.json').read_text());resources=dict(old['runner_config']['resources'])
    target=ROOT/'data/arc_authoring_v1';target.mkdir(exist_ok=False)
    data=Path('C:/Users/kevin/skillforge-runtime/datasets/arc_authoring_v1')
    train=[json.loads(x) for x in (data/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    examples=sorted(train,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:32]
    examples_path=target/'examples.json';write_json(examples_path,examples)
    tests={name:str(data/(name+'.jsonl')) for name in ('challenge_test','easy_test')}
    model=Path(old['constructor_kwargs']['model_path']);tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    # Integrity/token-length checks only, not test-model scoring; do not print examples.
    sets={};max_lengths={}
    paths=dict(train=str(data/'train.jsonl'),validation=old['runner_config']['datasets']['challenge'],**tests)
    for name,path in paths.items():
        rows=[json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]
        for row in rows:validate_row(row)
        sets[name]={r['id'] for r in rows};assert len(sets[name])==len(rows)
        max_lengths[name]=max(len(ids) for row in rows for ids,_ in encode_choices(tokenizer,row))
    assert not (sets['train'] & sets['challenge_test']) and not (sets['validation'] & sets['challenge_test'])
    tensors=load_file(model/'model.safetensors');base={k:tensors[name].float().numpy().copy() for k,name in TARGETS.items()};del tensors
    rows=[json.loads(x) for x in Path(paths['validation']).read_text(encoding='utf-8').splitlines()]
    predictions=[json.loads(x) for x in (baseline_root/'challenge.jsonl').read_text().splitlines()]
    assert len(rows)==len(predictions)==299
    initial_validation=dict(n=299,accuracy=baseline['summaries']['challenge']['accuracy'],raw_accuracy=baseline['summaries']['challenge']['raw_accuracy'],
        fixed_examples=[dict(question=r,score=s) for r,s in zip(rows[:16],predictions[:16])])
    architecture=json.loads((model/'config.json').read_text());guide=ROOT/'docs/wspl/ARC_AUTHORING_INTERFACE_V1.md'
    packet=dict(model='SmolLM2-135M-Instruct',architecture=architecture,training_examples=examples,baseline_validation=initial_validation,targets=TARGETS,rank=8,scale=1,max_relative_frobenius=.1)
    local=ROOT/'artifacts/local_paths.json';cli=Path(json.loads(local.read_text())['codex_cli']);catalog=ROOT/'data/astra_native_v1/catalog_frozen.json';instructions=ROOT/'data/astra_native_v1/instructions.md'
    bound=[local,cli,catalog,instructions,guide,ROOT/'docs/wspl/ARC_AUTHORING_V1.md',ROOT/'docs/wspl/arc_authoring_preflight.json',examples_path,
        baseline_root/'results.json',baseline_root/'challenge.jsonl',baseline_root/'source_receipt.json']
    for arm in ('native','wspl','lora'):
        mode='lora' if arm=='lora' else 'direct';p=dict(packet,current_tensors=statistics(initial_bank(base,mode)))
        text=prompt(guide.read_text(encoding='utf-8'),p,arm,[],0)
        path=target/(arm+'.prompt.txt');path.write_text(text,encoding='utf-8');bound.append(path)
        folder=target/(arm+'_preflight');folder.mkdir();(folder/'empty').mkdir();preflight(cli,catalog,instructions,folder,text)
    linux=json.loads((ROOT/'data/go_authoring_v1/linux_runtime.json').read_text())
    identity=target/'linux_runtime.json';write_json(identity,linux);bound.append(identity)
    for p in bound+[Path(p) for p in paths.values()]:resources[str(p)]=sha(p)
    for path,h in resources.items():
        if sha(path)!=h:raise ValueError('resource changed: '+path)
    spec=dict(schema=1,model_id='arc-authoring-v1',source_roots=['src','experiments','tests','tools'],constructor=old['constructor'],constructor_kwargs=old['constructor_kwargs'],
        runner=dict(module='run_arc_authoring',source='experiments/run_arc_authoring.py',symbol='run'),
        runner_config=dict(resources=resources,architecture=architecture,baseline_results=str(baseline_root/'results.json'),baseline_predictions=str(baseline_root/'challenge.jsonl'),
            validation=paths['validation'],examples=str(examples_path),test_datasets=tests,guide=str(guide),local_paths=str(local),catalog=str(catalog),instructions=str(instructions),
            python_worker='tools/arc_python_worker.py',wspl_worker='tools/arc_wspl_worker.py',python_runtime='/home/kevin/venvs/skillforge-author',
            runtime_verify='tools/construction_runtime_identity.py',linux_identity=str(identity),wall_guard_seconds=43200))
    rel='experiments/source_pins/arc_authoring_v1'
    for suffix in ('.spec.json','.manifest.json'):
        if (ROOT/(rel+suffix)).exists():raise ValueError('refuse overwrite')
    write_json(ROOT/(rel+'.spec.json'),spec);manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    isolated=ROOT.parent/'Skillforge-arc-authoring-v1';isolated.mkdir(exist_ok=False)
    for name in ('src','experiments','tests','tools'):shutil.copytree(ROOT/name,isolated/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',isolated/'AGENTS.md');h=sha(ROOT/(rel+'.manifest.json'));verify(isolated,isolated/(rel+'.manifest.json'),h)
    report=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),linux_resources=len(linux),max_encoded_lengths=max_lengths,author_calls=0,test_predictions=0)
    write_json(ROOT/'docs/wspl/arc_authoring_v1_preparation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
