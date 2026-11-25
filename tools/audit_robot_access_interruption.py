"""Read-only audit of completed author work and the usage-limit boundary."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import verify,write_json
from skillforge.fresh_robot import sha
from skillforge.robot_submissions import load_arrays
from skillforge.wspl_lora import identity


def main():
    prior=ROOT.parent/'WSPL-robot-authoring-access-v1'
    pin='fddea45cf93addfd070f01877b1eb8f096df2bfcf18fc5597a331298f7332d71'
    verify(prior,prior/'experiments/source_pins/robot_authoring_access_v2.manifest.json',pin)
    out=prior/'results/robot_authoring_access_v1'
    status=json.loads((out/'status.json').read_text())
    assert status['status']=='failed'
    assert json.loads(out.with_suffix('.exit.json').read_text())['returncode']==1
    assert not (out/'results.json').exists() and not (out/'all_frozen.json').exists() and not list(out.glob('final_*'))
    completed=['native0','wspl0','wspl1'];candidate_hashes={};counts={}
    for name in completed+['native1']:
        folder=out/name;local=json.loads((folder/'status.json').read_text())
        used=list(folder.glob('turn_*/usage.json'));assert len(used)==local['turns']
        assert len(json.loads((folder/'transcript.json').read_text()))==2*local['turns']
        if name in completed:
            frozen=json.loads((folder/'frozen.json').read_text());assert frozen['status']=='frozen'
            assert frozen['candidate']==local['candidate']
        else:assert not (folder/'frozen.json').exists() and local['turns']==21
        path=Path(local['candidate']);bank=load_arrays(path)
        candidate_hashes[name]=dict(file=str(path),sha256=sha(path),adapter_id=identity(bank))
        counts[name]=dict(completed_responses=len(used),inspection_attempts=local['inspection_attempts'],development_attempts=local['proposal_attempts'])
    failed=out/'native1/turn_21/collection'
    events=[json.loads(s) for s in (failed/'events.jsonl').read_text().splitlines() if s.strip()]
    assert any(e.get('type')=='error' and 'usage limit' in e.get('message','') for e in events)
    assert not any(e.get('type')=='turn.completed' or e.get('item',{}).get('type')=='agent_message' for e in events)
    assert not (out/'native1/turn_21/usage.json').exists()
    assert not (failed/'answer.md').exists() or not (failed/'answer.md').read_bytes().strip()
    assert not (out/'native2').exists() and not (out/'wspl2').exists()
    paths=[]
    for path in sorted(out.rglob('*')):
        assert not path.is_symlink() and path.resolve().is_relative_to(out.resolve())
        if path.is_file():assert path.stat().st_nlink==1;paths.append(path)
    paths += [out.with_suffix('.exit.json'),out.with_suffix('.launch.json')]
    with ThreadPoolExecutor(max_workers=8) as pool:digests=list(pool.map(sha,paths))
    info=dict(status='audited',prior_output=str(out),prior_manifest_sha256=pin,
        files={str(p):h for p,h in zip(paths,digests)},completed_authors=completed,
        resume_author='native1',resume=dict(author_folder=str(out/'native1'),consumed_responses=22),
        counts=counts,candidates=candidate_hashes,failed_invocations=1,failed_usage='unknown',
        prior_completed_responses=sum(c['completed_responses'] for c in counts.values()),
        fresh_author_invocations_max=50,prior_compute_seconds=status['development_compute_seconds'],
        prior_elapsed_seconds=status['elapsed_seconds'],final_opened=False)
    write_json(ROOT/'docs/wspl/robot_access_resume_v1_inventory.json',info)
    print(json.dumps({k:v for k,v in info.items() if k not in ('files','candidates')},indent=2))
    print('Output files verified:',len(paths))


if __name__=='__main__':main()
