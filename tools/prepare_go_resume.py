"""Prepare an unapproved recovery pin; never register or launch."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import inventory,write_json,verify
from skillforge.go_replay import sha
from run_go_authoring import import_frozen
import tempfile


def main():
    old=ROOT.parent/'Skillforge-go-authoring-v1'
    parent=old/'results/go_authoring_v1'
    digest='dcf87c7a310ea245af2411152065b5c6e9593d1cbc576862562c522cdfe27f1d'
    verify(old,parent/'approved_manifest.json',digest)
    if list(parent.rglob('evaluation')):raise ValueError('unexpected prior evaluation')
    spec=json.loads((parent/'approved_manifest.json').read_text());spec.pop('files')
    spec['model_id']='go-authoring-resume-v1';config=spec['runner_config']
    config['wall_guard_seconds']=14400
    config['resume']=dict(parent=str(parent),files={p.relative_to(parent).as_posix():sha(p) for p in parent.rglob('*') if p.is_file()})
    protocol=ROOT/'docs/wspl/GO_AUTHORING_RESUME_V1.md'
    config['windows_resources'][str(protocol)]=sha(protocol)
    for path,h in config['windows_resources'].items():
        if sha(path)!=h:raise ValueError('bound resource changed: '+path)
    with tempfile.TemporaryDirectory() as d:import_frozen(Path(d),config)
    isolated=ROOT.parent/'Skillforge-go-authoring-resume-v1'
    isolated.mkdir(exist_ok=False)
    for name in ('src','experiments','tests','tools'):
        shutil.copytree(ROOT/name,isolated/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',isolated/'AGENTS.md')
    relative='experiments/source_pins/go_authoring_resume_v1'
    write_json(ROOT/(relative+'.spec.json'),spec)
    manifest={**spec,'files':inventory(ROOT,spec['source_roots'])}
    write_json(ROOT/(relative+'.manifest.json'),manifest)
    for suffix in ('.spec.json','.manifest.json'):shutil.copy2(ROOT/(relative+suffix),isolated/(relative+suffix))
    pin=sha(ROOT/(relative+'.manifest.json'))
    verify(isolated,isolated/(relative+'.manifest.json'),pin)
    print(json.dumps(dict(status='UNAPPROVED',sha256=pin,sources=len(manifest['files']),parent_files=len(config['resume']['files']),isolated=str(isolated))))


if __name__=='__main__':main()
