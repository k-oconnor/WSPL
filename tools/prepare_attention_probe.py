"""Freeze one chosen ARC case and a designer-authored evaluation panel. No scoring."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from prepare_arc_baseline import sha
from source_consistency import inventory,write_json,verify
from skillforge.attention_probe import author_prompt
from skillforge.codex_one_shot import preflight
from skillforge.arc_eval import encode_choices
from transformers import AutoTokenizer


def panel(target):
    pairs=[
        ('paraphrase','A metal spoon sits in hot tea and becomes warm through direct contact. Which process transfers the heat into the spoon?','conduction'),
        ('paraphrase','Hot soup warms the submerged end of a metal spoon. Heat then travels along the spoon to its handle. What is this process called?','conduction'),
        ('paraphrase','A cool metal rod touches a hot metal block. Thermal energy passes across their contact surface. Which process explains this?','conduction'),
        ('paraphrase','A metal pan handle becomes hot as thermal energy travels through the solid metal from the pan. Which mechanism is responsible?','conduction'),
        ('different_mechanism','Sunlight carries thermal energy across the vacuum of space to Earth. Which process transfers this energy?','radiation'),
        ('different_mechanism','A person feels warmth from infrared waves emitted by a fire without touching it. Which process carries the energy in those waves?','radiation'),
        ('different_mechanism','Warm water rises and cool water sinks, circulating heat around a pot. What is this transfer by bulk fluid motion called?','convection'),
        ('different_mechanism','Warm air rises from a radiator and circulates around a room. Which process transfers heat by this movement of air?','convection'),
        ('different_mechanism','A puddle slowly disappears as liquid water molecules escape into the air as vapor. What is this process called?','evaporation'),
        ('different_mechanism','Sweat changes from liquid to vapor at the skin surface and carries energy away. What is that phase-change process called?','evaporation'),
        ('lexical_control','Which of the listed words begins with the letter r?','radiation'),
        ('lexical_control','Which of the listed words begins with the letters evap?','evaporation'),
    ]
    rows=[]
    for i,(category,question,answer) in enumerate(pairs):
        choices={k:list(v) for k,v in target['choices'].items()}
        rows.append(dict(id=f'attention_panel_{i:02d}',question=question,choices=choices,
                         answerKey=choices['label'][choices['text'].index(answer)],category=category))
    originals=[dict(target,category='target')]+rows.copy()
    for row in originals:
        choices=dict(label=list(row['choices']['label']),text=list(reversed(row['choices']['text'])))
        correct=row['choices']['text'][row['choices']['label'].index(row['answerKey'])]
        rows.append(dict(row,id=row['id']+'_reversed',choices=choices,answerKey=choices['label'][choices['text'].index(correct)]))
    return rows


def main():
    check=json.loads((ROOT/'docs/wspl/attention_probe_preflight.json').read_text())
    assert check['status']=='passed' and check['restored_logits_bit_identical']
    oldroot=ROOT.parent/'Skillforge-arc-baseline-v1/results/arc_baseline_v1'
    old=json.loads((oldroot/'approved_manifest.json').read_text());resources=dict(old['runner_config']['resources'])
    rows=[json.loads(x) for x in Path(old['runner_config']['datasets']['challenge']).read_text().splitlines()]
    scores=[json.loads(x) for x in (oldroot/'challenge.jsonl').read_text().splitlines()]
    selected='Mercury_SC_408700';target=next(r for r in rows if r['id']==selected);score=next(r for r in scores if r['id']==selected)
    assert not score['correct'] and target['answerKey']=='C'
    folder=ROOT/'data/attention_probe_v1';folder.mkdir(exist_ok=False)
    write_json(folder/'target.json',target);write_json(folder/'panel.json',panel(target))
    tokenizer=AutoTokenizer.from_pretrained(old['constructor_kwargs']['model_path'],local_files_only=True,trust_remote_code=False)
    for row in [target]+panel(target):
        encoded=encode_choices(tokenizer,row)
        assert all(len(ids)-start==1 for ids,start in encoded)
    guide=ROOT/'docs/wspl/ATTENTION_PROBE_INTERFACE_V1.md';protocol=ROOT/'docs/wspl/ATTENTION_PROBE_V1.md'
    local=ROOT/'artifacts/local_paths.json';cli=Path(json.loads(local.read_text())['codex_cli'])
    catalog=ROOT/'data/astra_native_v1/catalog_frozen.json';instructions=ROOT/'data/astra_native_v1/instructions.md'
    identity=folder/'linux_runtime.json';write_json(identity,json.loads((ROOT/'data/go_authoring_v1/linux_runtime.json').read_text()))
    bound=[guide,protocol,local,cli,catalog,instructions,identity,folder/'target.json',folder/'panel.json',oldroot/'challenge.jsonl',
           ROOT/'docs/wspl/attention_probe_preflight.json',ROOT/'docs/wspl/attention_probe_sandbox_preflight.json']
    for stage in ('forced','retrieval'):
        text=author_prompt(guide.read_text(),dict(target,baseline_score=score),stage,[],0)
        p=folder/(stage+'.prompt.txt');p.write_text(text,encoding='utf-8');bound.append(p)
        pf=folder/(stage+'_preflight');pf.mkdir();(pf/'empty').mkdir();preflight(cli,catalog,instructions,pf,text)
    for path in bound:resources[str(path)]=sha(path)
    for path,h in resources.items():
        if sha(path)!=h:raise ValueError('resource differs: '+path)
    spec=dict(schema=1,model_id='attention-probe-v1',source_roots=['src','experiments','tests','tools'],
        constructor=old['constructor'],constructor_kwargs=old['constructor_kwargs'],
        runner=dict(module='run_attention_probe',source='experiments/run_attention_probe.py',symbol='run'),
        runner_config=dict(resources=resources,target=str(folder/'target.json'),panel=str(folder/'panel.json'),
            model_sha256=check['model_sha256'],saved_target_score=score,guide=str(guide),local_paths=str(local),catalog=str(catalog),instructions=str(instructions),
            python_worker='tools/arc_python_worker.py',python_runtime='/home/kevin/venvs/skillforge-author',
            runtime_verify='tools/construction_runtime_identity.py',linux_identity=str(identity),wall_guard_seconds=14400))
    rel='experiments/source_pins/attention_probe_v1'
    for suffix in ('.spec.json','.manifest.json'):
        if (ROOT/(rel+suffix)).exists():raise ValueError('refuse overwrite')
    write_json(ROOT/(rel+'.spec.json'),spec);manifest={**spec, 'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    isolated=ROOT.parent/'Skillforge-attention-probe-v1';isolated.mkdir(exist_ok=False)
    for name in ('src','experiments','tests','tools'):shutil.copytree(ROOT/name,isolated/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',isolated/'AGENTS.md')
    h=sha(ROOT/(rel+'.manifest.json'));verify(isolated,isolated/(rel+'.manifest.json'),h)
    report=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),
                linux_resources=len(json.loads(identity.read_text())),isolated=str(isolated),author_calls=0,new_benchmark_predictions=0,
                target_id=selected,evaluation_rows=1+len(panel(target)),max_author_calls=12)
    write_json(ROOT/'docs/wspl/attention_probe_v1_preparation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
