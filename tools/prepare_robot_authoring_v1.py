"""Bind completed fixtures and all runtimes; never register or launch a study."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import inventory,verify,write_json
from skillforge.fresh_robot import sha


def main():
    destination=ROOT/'experiments/source_pins/robot_authoring_v1.manifest.json'
    if destination.exists():raise ValueError('Candidate already exists; preserve source identities')
    frozen=ROOT.parent/'WSPL-fresh-robot-parent-v2'
    old_path=frozen/'experiments/source_pins/fresh_robot_parent_v2.manifest.json'
    old=verify(frozen,old_path,'3ecff9baac23eab515a586d3bb629f9123bc97f08d243dcaf75c27021c10716e')
    resources=dict(old['runner_config']['resources'])
    parent=frozen/'results/fresh_robot_parent_v2/adapter_final.pt'
    if sha(parent)!='2cd5bd6662db5bdb39e6d341250abfca083672e2b9794c939abc145b49470276':raise ValueError('Parent changed')
    runtime_root=Path(r'C:\Users\kevin\wspl-author-python-v1')
    env_path=runtime_root/'environment_manifest.json';environment=json.loads(env_path.read_text())
    if sha(env_path)!='40edaf0b63eab1881648b5ca4d4ddc077ddb4c48c2e6ace0167091368681e84b':raise ValueError('Author environment identity changed')
    for row in environment['files']:resources[str(Path(environment['root'])/row['path'])]=row['sha256']
    native_env_path=Path(r'C:\Users\kevin\wspl-author-native-v1\environment_manifest.json')
    if sha(native_env_path)!='9d7d23116e9c1418da46f3c2bc955d2a18c44bd7e27dad1ecaaa5df0cc27ecd4':raise ValueError('Native runtime identity changed')
    native_environment=json.loads(native_env_path.read_text())
    common_hashes={row['path']:row['sha256'] for row in environment['files']}
    native_hashes={row['path']:row['sha256'] for row in native_environment['files']}
    if any(common_hashes[p]!=h for p,h in native_hashes.items() if p in common_hashes):raise ValueError('Unmatched numerical runtime bytes')
    if set(common_hashes)-set(native_hashes)!={p for p in common_hashes if p.startswith('Lib/site-packages/skillforge/')}:
        raise ValueError('Unexpected native runtime exclusions')
    if set(native_hashes)-set(common_hashes)!={'native_build_receipt.json'}:raise ValueError('Unexpected native runtime additions')
    for row in native_environment['files']:resources[str(Path(native_environment['root'])/row['path'])]=row['sha256']
    checks=[]
    for name in ('model','prompt','tools'):
        path=ROOT/f'docs/wspl/robot_authoring_{name}_preflight.json'
        if json.loads(path.read_text())['status']!='passed':raise ValueError('Missing passing '+name+' preflight')
        checks.append(path)
    unit=ROOT/'docs/wspl/robot_authoring_unit_checks.json'
    if json.loads(unit.read_text())['status']!='passed':raise ValueError('Focused tests required')
    cli=Path(r'C:\Users\kevin\.vscode\extensions\openai.chatgpt-26.1117.41037-win32-x64\bin\windows-x86_64\codex.exe')
    data=ROOT/'data/robot_authoring_v1'
    for path in [parent,old_path,env_path,native_env_path,cli,data/'catalog.json',data/'instructions.md',unit,
                 data/'teacher_preflight.json',ROOT/'docs/wspl/ROBOT_AUTHORING_V1.md',
                 ROOT/'docs/wspl/AUTHOR_PYTHON_ENVIRONMENT_V1.md',
                 ROOT/'docs/wspl/author_python_isolation_v1.json',
                 ROOT/'docs/wspl/author_python_workspace_v1.json',*checks]:resources[str(path)]=sha(path)
    def check(item):
        p,h=item
        if sha(p)!=h:raise ValueError('Bound resource mismatch: '+p)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(check,resources.items()):pass
    cfg=dict(resources=resources,learner_python=old['runner_config']['learner_python'],
        sim_python=old['runner_config']['sim_python'],sim_worker=str(ROOT/'tools/teaching_sim_server.py'),
        motor_artifact=old['runner_config']['motor_artifact'],parent_checkpoint=str(parent),
        parent_factor_hash='431806f5f4fc6788969a0f954d8fe594caa7b8597b3b293c2d23f57e61459c75',
        base_hash='540b6c0285e9d26362156163cef352c1a27657dc7b0372d0c36404a6c5baddc1',
        codec_source=str(ROOT/'src/skillforge/robot_codec.py'),author_runtime=environment['root'],native_runtime=native_environment['root'],
        submission_worker=str(ROOT/'tools/robot_submission_worker.py'),cli=str(cli),
        catalog=str(data/'catalog.json'),instructions=str(data/'instructions.md'),
        acquisition_seed=73300000,development_seed=73400000,final_seed=73500000,
        gradient_seed=114991,gradient_updates=384,learning_rate=3e-5,
        author_responses=16,proposal_attempts=4,author_call_seconds=1200,python_call_seconds=300,
        construction_seconds=180,development_wall_seconds=86400,final_wall_seconds=21600,compute_seconds=28800)
    spec=dict(schema=1,model_id='robot-authoring-v1',source_roots=['src','experiments','tools','tests'],
        constructor=old['constructor'],constructor_kwargs=old['constructor_kwargs'],
        runner=dict(module='run_robot_authoring_v1',symbol='run',source='experiments/run_robot_authoring_v1.py'),
        runner_config=cfg,files=inventory(ROOT,['src','experiments','tools','tests']))
    write_json(destination,spec);pin=sha(destination);verify(ROOT,destination,pin)
    report=dict(status='prepared_unapproved',sha256=pin,manifest=str(destination),sources=len(spec['files']),
        resources=len(resources),registered=False,launched=False,author_calls=0,
        source_checkout=str(ROOT),runtime_manifest_sha256=sha(env_path),native_runtime_manifest_sha256=sha(native_env_path),
        parent_checkpoint_sha256=sha(parent))
    write_json(ROOT/'docs/wspl/robot_authoring_v1_preparation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
