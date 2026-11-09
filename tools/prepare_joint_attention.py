"""Prepare one native joint author session; no scientific scoring or calls."""
import json,sys,shutil,re,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import inventory,verify,write_json
from skillforge.go_replay import sha
from skillforge.joint_family import family
from skillforge.joint_attention import prompt
from skillforge.codex_one_shot import preflight
from skillforge.diffusion_edit import linux_path
from skillforge.arc_eval import encode_choices
from transformers import AutoTokenizer


def main():
    check=json.loads((ROOT/'docs/wspl/joint_attention_preflight.json').read_text());assert check['status']=='passed'
    old=json.loads((ROOT/'experiments/source_pins/arc_baseline_v1.manifest.json').read_text());resources=dict(old['runner_config']['resources'])
    transfer=ROOT.parent/'Skillforge-arc-attention-transfer-v2/results/arc_attention_transfer_v1'
    candidates=[];bound=[]
    for split,path in old['runner_config']['datasets'].items():
        rows={r['id']:r for r in map(json.loads,Path(path).read_text(encoding='utf-8').splitlines())}
        record_path=transfer/(split+'_untouched.jsonl');bound.append(record_path)
        for record in map(json.loads,record_path.read_text().splitlines()):
            row=rows[record['id']]
            if record['correct'] and not re.search(r'\b(conduction|convection|radiation|evaporation|heat transfer|thermal energy)\b',row['question']+' '+' '.join(row['choices']['text']),re.I):
                candidates.append(dict(row,group='retention',source_split=split))
    candidates.sort(key=lambda r:hashlib.sha256(('joint-retention-v1:'+r['id']).encode()).hexdigest())
    assert len({r['id'] for r in candidates})==len(candidates)
    target=json.loads((ROOT/'data/attention_probe_v1/target.json').read_text());bound.append(ROOT/'data/attention_probe_v1/target.json')
    acquisition=family(0)+[dict(target,group='known_target')]+candidates[:12]
    reserved=family(1)+family(2)+candidates[12:24]
    assert len(acquisition)==21 and len(reserved)==28
    assert not {r['id'] for r in acquisition}&{r['id'] for r in reserved}
    tokenizer=AutoTokenizer.from_pretrained(old['constructor_kwargs']['model_path'],local_files_only=True,trust_remote_code=False)
    longest=max(len(ids) for row in acquisition+reserved for ids,start in encode_choices(tokenizer,row))
    folder=ROOT/'data/joint_attention_v1';folder.mkdir(exist_ok=False)
    write_json(folder/'acquisition.json',acquisition);write_json(folder/'reserved.json',reserved)
    guide=ROOT/'docs/wspl/JOINT_ATTENTION_INTERFACE_V1.md';protocol=ROOT/'docs/wspl/JOINT_ATTENTION_V1.md'
    local=ROOT/'artifacts/local_paths.json';cli=Path(json.loads(local.read_text())['codex_cli'])
    catalog=ROOT/'data/astra_native_v1/catalog_frozen.json';instructions=ROOT/'data/astra_native_v1/instructions.md'
    identity=ROOT/'data/go_authoring_v1/linux_runtime.json'
    subprocess.run(['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(ROOT/'tools/construction_runtime_identity.py'),'--verify',linux_path(identity)],check=True,capture_output=True,timeout=90)
    text=prompt(guide.read_text(),dict(examples=acquisition,note='Runtime adds acquisition-only scores and tensor shapes.'),[],0)
    (folder/'author.prompt.txt').write_text(text,encoding='utf-8');pf=folder/'preflight';pf.mkdir();(pf/'empty').mkdir();preflight(cli,catalog,instructions,pf,text)
    parent=ROOT.parent/'Skillforge-attention-probe-resume-v1/results/attention_probe_resume_v1/retrieval/final.npz'
    bound += [guide,protocol,local,cli,catalog,instructions,identity,parent,folder/'acquisition.json',folder/'reserved.json',folder/'author.prompt.txt',ROOT/'docs/wspl/joint_attention_preflight.json']
    for path in bound:resources[str(path)]=sha(path)
    for path,h in resources.items():
        if sha(path)!=h:raise ValueError('changed resource '+path)
    spec=dict(schema=1,model_id='joint-attention-v1',source_roots=['src','experiments','tests','tools'],constructor=old['constructor'],constructor_kwargs=old['constructor_kwargs'],
              runner=dict(module='run_joint_attention',source='experiments/run_joint_attention.py',symbol='run'),
              runner_config=dict(resources=resources,model_sha256=check['model_sha256'],parent_bank=str(parent),acquisition=str(folder/'acquisition.json'),reserved=str(folder/'reserved.json'),guide=str(guide),local_paths=str(local),catalog=str(catalog),instructions=str(instructions),
                                 python_worker='tools/arc_python_worker.py',python_runtime='/home/kevin/venvs/skillforge-author',runtime_verify='tools/construction_runtime_identity.py',linux_identity=str(identity),wall_guard_seconds=14400))
    rel='experiments/source_pins/joint_attention_v1'
    for suffix in ('.spec.json','.manifest.json'):assert not (ROOT/(rel+suffix)).exists()
    write_json(ROOT/(rel+'.spec.json'),spec);manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    iso=ROOT.parent/'Skillforge-joint-attention-v1';iso.mkdir(exist_ok=False)
    for name in spec['source_roots']:shutil.copytree(ROOT/name,iso/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',iso/'AGENTS.md');h=sha(ROOT/(rel+'.manifest.json'));verify(iso,iso/(rel+'.manifest.json'),h)
    result=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),linux_resources=len(json.loads(identity.read_text())),isolated=str(iso),acquisition=len(acquisition),reserved=len(reserved),max_encoded_length=longest,author_calls=0,benchmark_predictions=0)
    write_json(ROOT/'docs/wspl/joint_attention_v1_preparation.json',result);print(json.dumps(result))


if __name__=='__main__':main()
