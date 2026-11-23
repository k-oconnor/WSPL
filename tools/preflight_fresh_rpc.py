"""Cross-interpreter RPC boundary fixture; no scientific task/model scoring."""
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
    with tempfile.TemporaryDirectory(prefix='wspl-rpc-fixture-') as folder:
        out = Path(folder)
        motor = out/'motors.json'
        write_json(motor, motor_artifact())
        manifest = out/'unapproved_fixture.json'
        scopes = ['src', 'experiments', 'tools', 'tests']
        write_json(manifest, dict(schema=1, model_id='engineering-fixture-only',
            source_roots=scopes, files=inventory(ROOT, scopes),
            constructor=dict(source='src/skillforge/authoring_sft.py'),
            runner=dict(source='experiments/run_fresh_robot_parent.py')))
        config = dict(sim_python=sys.argv[1], sim_worker=str(ROOT/'tools/teaching_sim_server.py'))
        sim = Simulator(config, out, manifest, digest(manifest))
        try:
            public = sim.call(dict(op='start', artifact=str(motor), output=str(out/'episode'),
                seed=995, count=3, condition='ordinary', max_decisions=32))
            assert public['observation']['telemetry']['sequence'] == 0
            selected = sim.call(dict(op='act', action=json.dumps(dict(op='select',
                              module=public['observation']['available_modules'][0]))))
            assert selected['observation']['telemetry']['sequence'] == 0
            probed = sim.call(dict(op='act', action=json.dumps(dict(
                op='probe', command=[.12, 0, 0], ticks=6))))
            assert probed['observation']['telemetry']['sequence'] == 6
            stopped = sim.call(dict(op='act', action='{"op":"stop"}'))
            assert stopped['finished']
            result = sim.call(dict(op='finish', trace_path=str(out/'trace.json.gz')))
            assert result['success'] is False and result['physical_ticks'] == 6
            assert (out/'trace.json.gz').is_file()
            repeated = sim.call(dict(op='start', artifact=str(motor), output=str(out/'repeat'),
                seed=995, count=3, condition='ordinary', max_decisions=32))
            assert public_hash(repeated) == public_hash(public)
            sim.call(dict(op='act', action='{"op":"stop"}'))
            sim.call(dict(op='finish', trace_path=str(out/'repeat.trace.json.gz')))
        finally:
            sim.close()
        assert sim.process.poll() is not None
    report = dict(status='passed', engineering_only=True, model_calls=0,
                  updates=0, simulated_probe_ticks=6, child_exited=True,
                  source_gate_verified_in_worker=True, paired_initial_state_equal=True,
                  approval_registry_changed=False)
    write_json(ROOT/'docs/wspl/fresh_robot_rpc_preflight.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
