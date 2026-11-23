"""Build an unapproved concrete manifest; never register or launch a study."""
import json
import platform
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments')]
from skillforge.fresh_robot import motor_artifact, sha
from source_consistency import inventory, write_json, verify


def main():
    runtime = Path(sys.argv[1]).resolve()
    destination = ROOT/'experiments/source_pins/fresh_robot_parent_v1.manifest.json'
    if destination.exists():
        raise ValueError('Candidate already exists; do not overwrite a source identity')
    checks = {}
    resources = {}
    for kind in ('physics', 'student', 'rpc'):
        path = ROOT/f'docs/wspl/fresh_robot_{kind}_preflight.json'
        checks[kind] = json.loads(path.read_text())
        if checks[kind]['status'] != 'passed' or not checks[kind]['engineering_only']:
            raise ValueError('Engineering preflight is not passed: '+kind)
        resources[str(path)] = sha(path)
    if checks['student']['synthetic_context_tokens'] != 2048:
        raise ValueError('Full-context backward preflight required')
    for name in ('sim', 'learner'):
        executable = runtime/name/'Scripts/python.exe'
        report_path = runtime/(name+'_final_inventory.json')
        subprocess.run([str(executable), str(ROOT/'tools/fingerprint_fresh_runtime.py'),
                        str(report_path)], check=True, timeout=600)
        report = json.loads(report_path.read_text())
        if report['shared_packages']:
            raise ValueError('Shared system packages are not permitted')
        resources.update(report['resources'])
        resources[str(report_path)] = sha(report_path)
    model = runtime/'models/Qwen2.5-Coder-1.5B-Instruct'
    provenance = json.loads((model/'download_provenance.json').read_text())
    if provenance['revision'] != '2e1fd397ee46e1388853d2af2c993145b0f1098a':
        raise ValueError('Model revision mismatch')
    for path in model.iterdir():
        if path.is_file():
            resources[str(path)] = sha(path)
    data = ROOT/'data/fresh_robot_parent_v1'
    data.mkdir(parents=True, exist_ok=True)
    motors = data/'motors.json'
    write_json(motors, motor_artifact())
    machine = data/'machine.json'
    write_json(machine, dict(os=platform.platform(), gpu=subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
        text=True).strip()))
    for path in (motors, machine, ROOT/'docs/wspl/FRESH_ROBOT_PARENT_V1.md',
                 ROOT/'artifacts/astra_supervisor_adaptation/representation.py'):
        resources[str(path)] = sha(path)
    if sha(ROOT/'src/skillforge/robot_codec.py') != sha(
            ROOT/'artifacts/astra_supervisor_adaptation/representation.py'):
        raise ValueError('Historical raw-observation codec changed')
    config = dict(resources=resources, model_path=str(model), motor_artifact=str(motors),
        sim_python=str(runtime/'sim/Scripts/python.exe'),
        sim_worker=str(ROOT/'tools/teaching_sim_server.py'),
        learner_python=str(runtime/'learner/Scripts/python.exe'), wall_guard_seconds=28800,
        train_seed=73100000, train_missions=64, reserved_seed=73200000, reserved_missions=16,
        updates=384, batch_size=8, schedule_seed=113991, learning_rate=.00004, minimum_success=14)
    spec = dict(schema=1, model_id='fresh-robot-parent-v1',
        source_roots=['src', 'experiments', 'tools', 'tests'],
        constructor=dict(module='skillforge.authoring_sft', symbol='AuthoringSFTStudy',
                         source='src/skillforge/authoring_sft.py'),
        constructor_kwargs=dict(model_path=str(model), seed=113, rank=16, alpha=32),
        runner=dict(module='run_fresh_robot_parent', symbol='run',
                    source='experiments/run_fresh_robot_parent.py'), runner_config=config)
    spec['files'] = inventory(ROOT, spec['source_roots'])
    write_json(destination, spec)
    digest = sha(destination)
    verify(ROOT, destination, digest)
    report = dict(status='prepared_unapproved', manifest=str(destination), sha256=digest,
                  sources=len(spec['files']), resources=len(resources), checks=checks,
                  scientific_runs=0, author_calls=0, registry_changed=False)
    write_json(ROOT/'docs/wspl/fresh_robot_parent_preparation.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}))


if __name__ == '__main__':
    main()
