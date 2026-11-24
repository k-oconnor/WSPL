"""Prepare a separately approved continuation; never retry the failed study."""
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
    target=ROOT/'experiments/source_pins/robot_authoring_v2.manifest.json'
    if target.exists():raise ValueError('Preserve existing candidate identity')
    oldroot=ROOT.parent/'WSPL-robot-authoring-v1'
    oldmanifest=oldroot/'experiments/source_pins/robot_authoring_v1.manifest.json'
    old=verify(oldroot,oldmanifest,'b5f628915f5ea93b34decb7e225592366cb01f0ad3605146b82db472b92fa410')
    for name in ('robot_authoring','robot_trials','robot_gradient','wspl_lora','robot_codec','authoring_sft'):
        path='src/skillforge/'+name+'.py'
        if sha(ROOT/path)!=old['files'][path]:raise ValueError('Task/interface changed unexpectedly: '+name)
    recovery_path=ROOT/'docs/wspl/robot_authoring_v2_recovery_inputs.json'
    recovery=json.loads(recovery_path.read_text())
    resources=dict(old['runner_config']['resources']);resources.update(recovery['files'])
    reports=['robot_authoring_v2_unit_checks.json','robot_authoring_v2_recovery_preflight.json',
             'robot_collector_transport_v2_healthcheck.json','robot_authoring_prompt_preflight.json']
    for name in reports:
        path=ROOT/'docs/wspl'/name
        if json.loads(path.read_text())['status']!='passed':raise ValueError('Missing passing '+name)
    binary_folder=Path(old['runner_config']['cli']).parent
    paths=[p for p in binary_folder.rglob('*') if p.is_file()]
    paths += [recovery_path,oldmanifest,ROOT/'docs/wspl/ROBOT_AUTHORING_V2.md',
              ROOT/'docs/wspl/robot_authoring_v1_failure_audit.json',
              ROOT/'data/robot_authoring_v1/catalog.json',ROOT/'data/robot_authoring_v1/instructions.md']
    paths += [ROOT/'docs/wspl'/name for name in reports]
    for path in paths:resources[str(path)]=sha(path)
    def check(item):
        path,expected=item
        if sha(path)!=expected:raise ValueError('Changed bound resource: '+path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(check,resources.items()):pass
    cfg=dict(old['runner_config'])
    cfg.update(resources=resources,recovery_manifest=str(recovery_path),
        prior_compute_seconds=recovery['prior_compute_seconds'],prior_elapsed_seconds=recovery['prior_elapsed_seconds'],
        development_wall_seconds=86400-math.ceil(recovery['prior_elapsed_seconds']),
        sim_worker=str(ROOT/'tools/teaching_sim_server.py'),codec_source=str(ROOT/'src/skillforge/robot_codec.py'),
        submission_worker=str(ROOT/'tools/robot_submission_worker.py'),
        catalog=str(ROOT/'data/robot_authoring_v1/catalog.json'),instructions=str(ROOT/'data/robot_authoring_v1/instructions.md'))
    spec=dict(schema=1,model_id='robot-authoring-v2',source_roots=old['source_roots'],
        constructor=old['constructor'],constructor_kwargs=old['constructor_kwargs'],
        runner=dict(module='run_robot_authoring_v2',symbol='run',source='experiments/run_robot_authoring_v2.py'),
        runner_config=cfg,files=inventory(ROOT,old['source_roots']))
    write_json(target,spec);pin=sha(target);verify(ROOT,target,pin)
    report=dict(status='prepared_unapproved',manifest=str(target),sha256=pin,sources=len(spec['files']),
        resources=len(resources),registered=False,launched=False,source_checkout=str(ROOT),
        reused_control_updates=384,new_control_updates=0,prior_failed_author_invocations=1,
        fresh_author_invocations_max=96,transport_healthchecks=1,
        prior_compute_seconds=cfg['prior_compute_seconds'],development_wall_seconds=cfg['development_wall_seconds'])
    write_json(ROOT/'docs/wspl/robot_authoring_v2_preparation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
