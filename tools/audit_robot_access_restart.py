"""Verify that the final native1 factors already existed before the interruption."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old=ROOT.parent/'WSPL-robot-authoring-access-v1/results/robot_authoring_access_v1/native1'
    new=ROOT.parent/'WSPL-robot-access-resume-v1/results/robot_access_resume_v1/native1'
    prior=old/'sandbox/work/candidate5.npz'
    final=Path(json.loads((new/'frozen.json').read_text())['candidate'])
    before=json.loads((old/'transcript.json').read_text())
    after=json.loads((new/'transcript.json').read_text())
    assert after[:len(before)]==before
    requests=[json.loads(x['content']) for x in after[len(before):] if x['role']=='assistant']
    assert [x['tool'] for x in requests]==['evaluate','finish']
    assert requests[0]['path']=='submission5.json'
    with np.load(prior,allow_pickle=False) as a,np.load(final,allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        assert all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and np.array_equal(a[k].view(np.uint32),b[k].view(np.uint32)) for k in a.files)
        arrays=len(a.files);coefficients=sum(a[k].size for k in a.files)
    assert sha(prior)==sha(final.parent/'arrays.npz')
    report=dict(status='verified',prior_candidate=str(prior),final_candidate=str(final),
        prior_archive_sha256=sha(prior),reconstructed_archive_sha256=sha(final),staged_submission_file_identical=True,
        arrays=arrays,coefficients=coefficients,all_factors_bit_identical=True,visible_history_prefix_identical=True,
        prior_history_entries=len(before),resumed_tools=[x['tool'] for x in requests],new_construction_calls=0,
        interpretation='The numerical repair predates the restart. The resumed session evaluated and froze it; uninterrupted subsequent choices remain unknown.',
        scope='Saved files and visible transcripts only; no inference, fitting or author calls.')
    (ROOT/'docs/wspl/robot_access_restart_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
