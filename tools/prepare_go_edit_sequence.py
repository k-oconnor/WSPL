"""Linux preparation only: load/hash archived weights, never play games."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import inventory,write_json,verify
from skillforge.go_replay import sha
from skillforge.go_edit_sequence import GoEditSequence


def main():
    parent=ROOT.parent/'Skillforge-go-authoring-resume-v1/results/go_authoring_resume_v1'
    receipt=json.loads((parent/'source_receipt.json').read_text())
    if receipt['status']!='complete' or receipt['manifest_sha256']!='cf96be2cc6651cd382e27f77f7cfdc347b98a5a124e9e924c587e569fe3eb108':raise ValueError('parent identity differs')
    old=json.loads((ROOT/'experiments/source_pins/go_replay_v1.spec.json').read_text())
    resources={str(Path(p) if Path(p).is_absolute() else ROOT/p):h for p,h in old['runner_config']['resources'].items()}
    checkpoints={}
    for i in range(7):
        path=parent/f'2_wspl/turn_{i:02d}/constructed/candidate.npz'
        checkpoints[f'edit_{i}']=dict(path=str(path),sha256=sha(path));resources[str(path)]=sha(path)
    historical=parent/'2_wspl/evaluation';batches=sorted(historical.glob('s*_batch_*.npz'))
    if len(batches)!=189:raise ValueError('historical batches missing')
    for p in batches:resources[str(p)]=sha(p)
    for p in (parent/'source_receipt.json',parent/'approved_manifest.json',parent/'frozen_candidates.json',ROOT/'docs/wspl/GO_EDIT_SEQUENCE_V1.md'):
        resources[str(p)]=sha(p)
    final=json.loads((parent/'frozen_candidates.json').read_text())['2_wspl']
    if sha(parent/final['path'])!=checkpoints['edit_6']['sha256']:raise ValueError('last checkpoint differs from frozen winner')
    for path,h in resources.items():
        if sha(path)!=h:raise ValueError('resource changed: '+path)
    kwargs=dict(checkpoints=checkpoints,opponent_root=old['constructor_kwargs']['reference_root'])
    study=GoEditSequence(**kwargs)
    assert len(study.actors)==7 and len(study.opponents)==3
    spec=dict(schema=1,model_id='go-edit-sequence-v1',source_roots=['src','experiments','tests','tools'],
        constructor=dict(module='skillforge.go_edit_sequence',source='src/skillforge/go_edit_sequence.py',symbol='GoEditSequence'),constructor_kwargs=kwargs,
        runner=dict(module='run_go_edit_sequence',source='experiments/run_go_edit_sequence.py',symbol='run'),
        runner_config=dict(resources=resources,historical_final=str(historical),wall_guard_seconds=7200))
    rel='experiments/source_pins/go_edit_sequence_v1'
    for suffix in ('.spec.json','.manifest.json'):
        if (ROOT/(rel+suffix)).exists():raise ValueError('refuse overwrite')
    write_json(ROOT/(rel+'.spec.json'),spec)
    manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    isolated=ROOT.parent/'Skillforge-go-edit-sequence-v1';isolated.mkdir(exist_ok=False)
    for name in ('src','experiments','tests','tools'):
        shutil.copytree(ROOT/name,isolated/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',isolated/'AGENTS.md')
    h=sha(ROOT/(rel+'.manifest.json'));verify(isolated,isolated/(rel+'.manifest.json'),h)
    print(json.dumps(dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),checkpoints=7,historical_batches=189,model_calls=0,games=0)))


if __name__=='__main__':main()
