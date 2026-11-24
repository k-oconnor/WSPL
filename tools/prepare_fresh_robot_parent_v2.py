"""Verify unchanged runtime, bind process-lifecycle repair, never register/launch."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments')]
from skillforge.fresh_robot import motor_artifact, sha
from source_consistency import inventory, verify, write_json


def main():
    prior = ROOT.parent/'WSPL-supervision-wspl-lora-v1'
    previous = prior/'experiments/source_pins/fresh_robot_parent_v1.manifest.json'
    expected = 'a6e5bfa2b42d0e9fa1b2aca423f2bb398383ba3e7ea8b90e93d33c5aa65dc757'
    old = verify(prior, previous, expected)
    destination = ROOT/'experiments/source_pins/fresh_robot_parent_v2.manifest.json'
    if destination.exists():
        raise ValueError('Candidate exists; do not replace identities')
    resources = {p: h for p, h in old['runner_config']['resources'].items()
                 if not Path(p).is_relative_to(prior)}
    def check(item):
        p, h = item
        if sha(p) != h:
            raise ValueError('Historical bound resource changed: '+p)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(check, old['runner_config']['resources'].items()):
            pass
    preflight = ROOT/'docs/wspl/fresh_robot_rpc_v2_preflight.json'
    report = json.loads(preflight.read_text())
    if report['status'] != 'passed' or report['workers'] != 16 or not report['clean_child_exit']:
        raise ValueError('Repeated process-lifecycle preflight required')
    paths = [preflight, previous, ROOT/'docs/wspl/FRESH_ROBOT_PARENT_V2.md']
    # These unchanged-model and physics checks are historical supporting fixtures,
    # not newly executed full-size training or held-out policy evaluation.
    paths += [prior/f'docs/wspl/fresh_robot_{kind}_preflight.json' for kind in ('student', 'physics')]
    data = ROOT/'data/fresh_robot_parent_v2'
    data.mkdir(parents=True, exist_ok=True)
    motors = data/'motors.json'
    write_json(motors, motor_artifact())
    if sha(motors) != sha(old['runner_config']['motor_artifact']):
        raise ValueError('Motor artifact changed')
    paths.append(motors)
    for path in paths:
        resources[str(path)] = sha(path)
    spec = json.loads(previous.read_text())
    spec['model_id'] = 'fresh-robot-parent-v2'
    spec['runner'] = dict(module='run_fresh_robot_parent_v2', symbol='run',
                          source='experiments/run_fresh_robot_parent_v2.py')
    spec['runner_config'].update(resources=resources, motor_artifact=str(motors),
        sim_worker=str(ROOT/'tools/teaching_sim_server.py'))
    spec['files'] = inventory(ROOT, spec['source_roots'])
    write_json(destination, spec)
    digest = sha(destination)
    verify(ROOT, destination, digest)
    preparation = dict(status='prepared_unapproved', manifest=str(destination), sha256=digest,
        sources=len(spec['files']), resources=len(resources), prior_manifest_sha256=expected,
        change='One simulator process per mission; unchanged physics/model/training protocol',
        repeated_worker_preflight=report, replacement_launched=False, registry_changed=False)
    write_json(ROOT/'docs/wspl/fresh_robot_parent_v2_preparation.json', preparation)
    print(json.dumps({k: v for k, v in preparation.items() if k != 'repeated_worker_preflight'}))


if __name__ == '__main__':
    main()
