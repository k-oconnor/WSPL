"""Prepare frozen acquisition/final panels and an unapproved source identity."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from prepare_arc_baseline import sha
from source_consistency import inventory,verify,write_json
from skillforge.attention_probe import author_prompt
from skillforge.codex_one_shot import preflight
from skillforge.arc_eval import encode_choices
from transformers import AutoTokenizer


def final_panel(choices):
    # Every setting includes a matched quartet; no author can see these cases.
    settings=[
      ('pool',[
        'Warm pool water touches a cooler metal thermometer. Energy passes directly into the metal at their contact surface. Which process is this?',
        'In a swimming pool, warmer water rises and cooler water sinks, carrying thermal energy around the pool. Which process transfers heat through this circulation?',
        'Sunlight crosses the air and is absorbed by the swimming pool, warming the water through electromagnetic waves. Which heat-transfer process is this?',
        'On a dry day, liquid pool water escapes from its surface into the air as water vapor. Which process causes this loss of liquid water?']),
      ('greenhouse',[
        'A cool metal probe is pushed into warm greenhouse soil. The probe warms where it directly touches the soil. Which heat-transfer process occurs across that contact?',
        'Air warmed by the greenhouse floor rises while cooler air moves down, carrying heat through moving air. Which process is this?',
        'Infrared waves from a greenhouse heater warm the soil without contact or moving warm air. Which heat-transfer process carries that energy?',
        'Water on greenhouse leaves changes from liquid into vapor and enters the air. Which process describes this phase change?']),
      ('ironing',[
        'The hot metal base of an iron touches a damp shirt. Heat passes directly from metal into the cloth. Which heat-transfer process is this?',
        'Air next to a hot iron warms, rises, and is replaced by cooler air, carrying heat in a circulating current. Which process is this?',
        'A hot iron emits infrared waves that transfer energy to a nearby surface without contact. Which heat-transfer process do these waves represent?',
        'Moisture in a damp shirt changes from liquid water into vapor while the shirt dries. Which process removes the liquid moisture?']),
      ('engine',[
        'A hot engine block is in direct contact with a cooler metal bracket. Thermal energy passes into the bracket through their contact. Which process is responsible?',
        'Warm coolant rises and cooler coolant sinks in a tank next to an engine, transporting heat through bulk liquid motion. Which process is this?',
        'A hot engine surface emits infrared electromagnetic waves toward nearby objects. Which heat-transfer process carries this energy?',
        'A thin film of water on an engine cover changes from liquid to vapor and disappears into the air. Which process is occurring?']),
      ('roof',[
        'A sun-warmed metal roof directly touches a cooler metal support. Heat flows through the contact into the support. Which process transfers the heat?',
        'Warm air above a hot roof rises while cooler air takes its place, transporting thermal energy by air movement. Which process is this?',
        'Energy from the Sun travels through space as electromagnetic waves and warms a roof. Which process transfers this energy?',
        'Rainwater on a roof slowly changes from liquid into water vapor as the roof dries. Which process is this?']),
      ('oven',[
        'A hot baking tray touches the bottom of a cooler piece of bread. Heat moves directly across their contact. Which process is this?',
        'Air warmed at the bottom of an oven rises and circulates, carrying thermal energy to other parts of the oven. Which process is this?',
        'An oven element sends infrared electromagnetic waves to food without requiring contact or air circulation. Which heat-transfer process is this?',
        'Liquid moisture at the surface of bread becomes water vapor during drying. Which process is responsible for this change from liquid to gas?']),
      ('spacesuit',[
        'Inside a spacesuit, a warm metal instrument directly touches a cooler metal mount. Energy passes across their contact. Which heat-transfer process is this?',
        'Inside a sealed habitat, warmer air rises and cooler air sinks, carrying heat through the air. Which process is responsible for this circulation-based transfer?',
        'An astronaut receives energy from sunlight through the vacuum outside a spacecraft. Which process can carry this thermal energy through a vacuum?',
        'Sweat inside a spacesuit changes from liquid to water vapor at the skin surface. Which process is this?']),
      ('tent',[
        'A warm metal camping mug directly touches a cold metal stand. Heat passes through the contact into the stand. Which process is this?',
        'Warm air inside a tent rises and cooler air sinks, carrying heat through circulating air. Which heat-transfer process is this?',
        'A camper feels warmth from infrared electromagnetic waves emitted by hot coals without touching them. Which heat-transfer process is this?',
        'Water droplets on a tent fabric change from liquid into vapor as the fabric dries. Which process causes this change?'])
    ]
    rows=[]
    mechanisms=['conduction','convection','radiation','evaporation']
    for setting,questions in settings:
        for mechanism,question in zip(mechanisms,questions):
            rows.append(dict(id='selective_'+setting+'_'+mechanism,question=question,choices=choices,
                answerKey=choices['label'][choices['text'].index(mechanism)],category=mechanism,setting=setting))
    return rows


def main():
    check=json.loads((ROOT/'docs/wspl/selective_key_preflight.json').read_text());assert check['status']=='passed'
    oldroot=ROOT.parent/'Skillforge-attention-probe-resume-v1/results/attention_probe_resume_v1'
    prior=json.loads((oldroot/'results.json').read_text());assert prior['status']=='complete'
    original=ROOT/'data/attention_probe_v1'
    target=json.loads((original/'target.json').read_text())
    training=[dict(target,category='target')]+[r for r in json.loads((original/'panel.json').read_text()) if not r['id'].endswith('_reversed')]
    panel=final_panel(target['choices']);assert len(training)==13 and len(panel)==32
    assert not {r['question'] for r in training}&{r['question'] for r in panel}
    old=json.loads((ROOT/'experiments/source_pins/arc_baseline_v1.manifest.json').read_text());resources=dict(old['runner_config']['resources'])
    tokenizer=AutoTokenizer.from_pretrained(old['constructor_kwargs']['model_path'],local_files_only=True,trust_remote_code=False)
    for r in training+panel:
        assert all(len(ids)-start==1 for ids,start in encode_choices(tokenizer,r))
    folder=ROOT/'data/selective_key_v1';folder.mkdir(exist_ok=False)
    write_json(folder/'training.json',training);write_json(folder/'panel.json',panel)
    guide=ROOT/'docs/wspl/SELECTIVE_KEY_INTERFACE_V1.md';protocol=ROOT/'docs/wspl/SELECTIVE_KEY_V1.md'
    local=ROOT/'artifacts/local_paths.json';cli=Path(json.loads(local.read_text())['codex_cli'])
    catalog=ROOT/'data/astra_native_v1/catalog_frozen.json';instructions=ROOT/'data/astra_native_v1/instructions.md'
    identity=original/'linux_runtime.json'
    text=author_prompt(guide.read_text(),dict(examples=training,note='Runtime adds measured acquisition scores and diagnostics.'),'retrieval',[],0)
    prompt=folder/'author.prompt.txt';prompt.write_text(text,encoding='utf-8');pf=folder/'preflight';pf.mkdir();(pf/'empty').mkdir()
    preflight(cli,catalog,instructions,pf,text)
    bound=[guide,protocol,local,cli,catalog,instructions,identity,folder/'training.json',folder/'panel.json',prompt,
           ROOT/'docs/wspl/selective_key_preflight.json',oldroot/'retrieval/final.npz',oldroot/'results.json',oldroot/'source_receipt.json']
    for p in bound:resources[str(p)]=sha(p)
    for p,h in resources.items():
        if sha(p)!=h:raise ValueError('resource changed: '+p)
    spec=dict(schema=1,model_id='selective-key-v1',source_roots=['src','experiments','tests','tools'],constructor=old['constructor'],constructor_kwargs=old['constructor_kwargs'],
        runner=dict(module='run_selective_key',source='experiments/run_selective_key.py',symbol='run'),
        runner_config=dict(resources=resources,parent_bank=str(oldroot/'retrieval/final.npz'),parent_results=str(oldroot/'results.json'),model_sha256=prior['model_sha256'],
            training=str(folder/'training.json'),panel=str(folder/'panel.json'),guide=str(guide),local_paths=str(local),catalog=str(catalog),instructions=str(instructions),
            python_worker='tools/arc_python_worker.py',python_runtime='/home/kevin/venvs/skillforge-author',runtime_verify='tools/construction_runtime_identity.py',linux_identity=str(identity),wall_guard_seconds=14400))
    rel='experiments/source_pins/selective_key_v1'
    for suffix in ('.spec.json','.manifest.json'):
        if (ROOT/(rel+suffix)).exists():raise ValueError('refuse overwrite')
    write_json(ROOT/(rel+'.spec.json'),spec);manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    isolated=ROOT.parent/'Skillforge-selective-key-v1';isolated.mkdir(exist_ok=False)
    for name in ('src','experiments','tests','tools'):shutil.copytree(ROOT/name,isolated/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',isolated/'AGENTS.md');h=sha(ROOT/(rel+'.manifest.json'));verify(isolated,isolated/(rel+'.manifest.json'),h)
    report=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),isolated=str(isolated),new_author_calls=0,new_benchmark_predictions=0,training_examples=13,final_examples=32,max_calls=6)
    write_json(ROOT/'docs/wspl/selective_key_v1_preparation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
