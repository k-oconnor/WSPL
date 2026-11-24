"""Read-only validation of the interrupted prefix; no forward passes or calls."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import verify
from skillforge.fresh_robot import sha
from skillforge.robot_submissions import load_arrays
from skillforge.teaching_training import state_hash
from skillforge.wspl_lora import identity
from skillforge.reliable_json import save


def main():
    old=ROOT.parent/'WSPL-robot-authoring-v1';out=old/'results/robot_authoring_v1'
    pin='b5f628915f5ea93b34decb7e225592366cb01f0ad3605146b82db472b92fa410'
    manifest=verify(old,out/'approved_manifest.json',pin)
    def check(item):
        path,expected=item
        if sha(path)!=expected:raise ValueError('Changed prefix resource: '+path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(check,manifest['runner_config']['resources'].items()):pass
    receipt=json.loads((out/'source_receipt.json').read_text());status=json.loads((out/'status.json').read_text())
    assert receipt['status']=='failed' and receipt['constructor_calls']==1
    events=[json.loads(line) for line in (out/'native0/turn_00/collection/events.jsonl').read_text().splitlines()]
    assert any('code-mode host is disabled' in e.get('item',{}).get('message','') for e in events)
    assert not any(e['type']=='turn.completed' for e in events)
    assert all(not p.read_text().strip() for p in out.glob('*/turn_*/collection/answer.md'))
    assert not list(out.glob('final_*')) and not (out/'all_frozen.json').exists()
    hashes=json.loads((out/'common_input_hashes.json').read_text())
    assert {p.name for p in (out/'common_inputs').iterdir()}==set(hashes)
    for name,expected in hashes.items():assert sha(out/'common_inputs'/name)==expected
    acquisition=json.loads((out/'acquisition/teacher_episodes.json').read_text())
    rows=json.loads((out/'common_inputs/examples.json').read_text())
    assert len(acquisition)==16 and all(r['success'] for r in acquisition) and len(rows)==255
    assert rows==json.loads((out/'acquisition/rows.json').read_text())
    saved_rows=[]
    for folder in sorted((out/'acquisition').glob('mission_*')):
        episode=json.loads((folder/'result.json').read_text())
        decisions=json.loads((folder/'decisions.json').read_text())
        assert episode['decisions']==len(decisions) and (folder/'trace.json.gz').exists()
        saved_rows.extend(decisions)
    assert len(saved_rows)==len(rows)
    features=load_arrays(out/'common_inputs/features.npz')
    assert features['X'].shape==(255,8960) and features['Y'].shape==(255,9)
    parent=load_arrays(out/'common_inputs/parent.npz')
    assert state_hash({k:torch.from_numpy(v) for k,v in parent.items()})==manifest['runner_config']['parent_factor_hash']
    gradient=load_arrays(out/'gradient/candidate.npz')
    frozen=json.loads((out/'gradient/frozen.json').read_text())
    rolling=torch.load(out/'gradient/rolling.pt',map_location='cpu',weights_only=True)
    assert state_hash(rolling['adapter'])==frozen['adapter_sha256']
    assert all(np.array_equal(rolling['adapter'][k].numpy().view(np.uint32),v.view(np.uint32)) for k,v in gradient.items())
    assert frozen['step']==384 and frozen['targets']==3072 and frozen['manifest_sha256']==pin
    assert frozen['base_hash']==manifest['runner_config']['base_hash']
    training=[json.loads(line) for line in (out/'gradient/training.jsonl').read_text().splitlines()]
    assert [r['step'] for r in training]==list(range(1,385)) and all(np.isfinite(r['nll']) for r in training)
    development=json.loads((out/'parent_development/episodes.json').read_text())
    assert len(development)==16
    for row in development:
        folder=out/'parent_development'/f"mission_{row['mission']:03d}"
        decisions=json.loads((folder/'decisions.json').read_text())
        assert len(decisions)==row['model_calls'] and (folder/'trace.json.gz').exists()
    # Every old output is fixed as provenance, including the failed collector attempt.
    files={str(p):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    recovery=dict(prior_root=str(out),prior_manifest_sha256=pin,files=files,common_input_hashes=hashes,
                  gradient_adapter_id=identity(gradient),parent_adapter_id=identity(parent),
                  prior_compute_seconds=status['development_compute_seconds'],prior_elapsed_seconds=status['elapsed_seconds'])
    save(ROOT/'docs/wspl/robot_authoring_v2_recovery_inputs.json',recovery)
    result=dict(status='audited_interrupted',reason='collector disabled required Code Mode host',
        prior_manifest_sha256=pin,sources=len(manifest['files']),resources=len(manifest['runner_config']['resources']),
        artifact_files=len(files),teacher_missions=16,teacher_successes=16,acquisition_decisions=255,
        parent_development_missions=16,gradient_updates=384,gradient_targets=3072,
        gradient_adapter_id=identity(gradient),gradient_checkpoint_matches=True,
        attempted_author_invocations=1,completed_author_responses=0,prior_author_token_usage='unknown',
        final_evaluations=0,elapsed_seconds=status['elapsed_seconds'],new_model_forwards=0,
        next_run_launched=False,source_preserved=True)
    save(ROOT/'docs/wspl/robot_authoring_v1_failure_audit.json',result);print(json.dumps(result))


if __name__=='__main__':main()
