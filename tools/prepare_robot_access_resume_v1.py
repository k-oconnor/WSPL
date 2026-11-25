"""Bind audited prefix and repaired orchestration; no registration or launch."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import verify,inventory,write_json
from skillforge.fresh_robot import sha


def main():
    target=ROOT/'experiments/source_pins/robot_access_resume_v1.manifest.json'
    if target.exists():raise ValueError('Preserve candidate identity')
    prior=ROOT.parent/'WSPL-robot-authoring-access-v1'
    prior_path=prior/'experiments/source_pins/robot_authoring_access_v2.manifest.json'
    old=verify(prior,prior_path,'fddea45cf93addfd070f01877b1eb8f096df2bfcf18fc5597a331298f7332d71')
    for name in ('robot_access','robot_trials','robot_submissions','wspl_lora','authoring_sft','robot_codec','robot_recovery'):
        path='src/skillforge/'+name+'.py'
        if sha(ROOT/path)!=old['files'][path]:raise ValueError('Scientific interface changed: '+path)
    invpath=ROOT/'docs/wspl/robot_access_resume_v1_inventory.json';inv=json.loads(invpath.read_text())
    resources=dict(old['runner_config']['resources']);resources.update(inv['files'])
    for name in ('robot_access_resume_v1_preflight.json','robot_access_resume_v1_unit_checks.json'):
        path=ROOT/'docs/wspl'/name
        if json.loads(path.read_text())['status']!='passed':raise ValueError('Missing passing '+name)
        resources[str(path)]=sha(path)
    for path in (invpath,prior_path,ROOT/'docs/wspl/ROBOT_ACCESS_RESUME_V1.md',ROOT/'data/robot_access_resume_v1/catalog.json',ROOT/'data/robot_access_resume_v1/instructions.md'):
        resources[str(path)]=sha(path)
    def check(item):
        path,expected=item
        if sha(path)!=expected:raise ValueError('Bound resource changed: '+path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(check,resources.items()):pass
    cfg=dict(old['runner_config']);cfg.update(resources=resources,author_recovery_inventory=str(invpath),
        prior_compute_seconds=inv['prior_compute_seconds'],prior_elapsed_seconds=inv['prior_elapsed_seconds'],
        development_wall_seconds=86400-math.ceil(inv['prior_elapsed_seconds']),
        sim_worker=str(ROOT/'tools/teaching_sim_server.py'),codec_source=str(ROOT/'src/skillforge/robot_codec.py'),
        submission_worker=str(ROOT/'tools/robot_submission_worker.py'),
        catalog=str(ROOT/'data/robot_access_resume_v1/catalog.json'),instructions=str(ROOT/'data/robot_access_resume_v1/instructions.md'))
    spec=dict(schema=1,model_id='robot-access-resume-v1',source_roots=old['source_roots'],
        constructor=old['constructor'],constructor_kwargs=old['constructor_kwargs'],
        runner=dict(module='run_robot_access_resume_v1',symbol='run',source='experiments/run_robot_access_resume_v1.py'),
        runner_config=cfg,files=inventory(ROOT,old['source_roots']))
    write_json(target,spec);pin=sha(target);verify(ROOT,target,pin)
    report=dict(status='prepared_unapproved',manifest=str(target),sha256=pin,source_checkout=str(ROOT),
        sources=len(spec['files']),resources=len(resources),registered=False,launched=False,
        completed_authors_reused=3,interrupted_author_remaining_responses=2,fresh_author_invocations_max=50,
        fresh_inspection_attempts_max=19,fresh_development_attempts_max=10,final_missions_max=512,
        prior_failed_invocations=1,prior_failed_usage='unknown',prior_completed_responses=92,
        development_wall_seconds=cfg['development_wall_seconds'],prior_compute_seconds=cfg['prior_compute_seconds'])
    write_json(ROOT/'docs/wspl/robot_access_resume_v1_preparation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
