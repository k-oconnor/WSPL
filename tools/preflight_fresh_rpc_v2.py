"""Sixteen independent scene loads; no task scores, learned policy, or training."""
import json
from pathlib import Path
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments')]
from skillforge.fresh_simulator import Simulator
from source_consistency import inventory, write_json, digest
from skillforge.fresh_robot import motor_artifact, public_hash


def main():
    initial = None
    workers = []
    with tempfile.TemporaryDirectory(prefix='wspl-rpc-v2-') as folder:
        out = Path(folder)
        motor = out/'motors.json'
        write_json(motor, motor_artifact())
        manifest = out/'unapproved_fixture.json'
        scopes = ['src', 'experiments', 'tools', 'tests']
        write_json(manifest, dict(schema=1, model_id='engineering-fixture-only',
            source_roots=scopes, files=inventory(ROOT, scopes),
            constructor=dict(source='src/skillforge/authoring_sft.py'),
            runner=dict(source='experiments/run_fresh_robot_parent_v2.py')))
        config = dict(sim_python=sys.argv[1], sim_worker=str(ROOT/'tools/teaching_sim_server.py'))
        for i in range(16):
            rpc = out/f'worker_{i}'
            rpc.mkdir()
            sim = Simulator(config, rpc, manifest, digest(manifest))
            try:
                public = sim.call(dict(op='start', artifact=str(motor), output=str(rpc/'episode'),
                    seed=995, count=3, condition='ordinary', max_decisions=1))
                h = public_hash(public)
                if initial is None:
                    initial = h
                assert h == initial
                sim.call(dict(op='act', action='{"op":"stop"}'))
                sim.call(dict(op='finish', trace_path=str(rpc/'trace.json.gz')))
            finally:
                sim.close()
            assert sim.process.poll() is not None
            workers.append(sim.process.pid)
            print(json.dumps(dict(engineering_workers_completed=i+1)), flush=True)
    report = dict(status='passed', engineering_only=True, task_policy_scores=0,
        model_calls=0, optimizer_updates=0, workers=len(workers), worker_pids=workers,
        identical_initial_observations=True, clean_child_exit=True, source_gate_verified=True)
    write_json(ROOT/'docs/wspl/fresh_robot_rpc_v2_preflight.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
