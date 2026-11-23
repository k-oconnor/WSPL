"""Deterministic engineering fixtures only; no Qwen predictions/training."""
import json
from pathlib import Path
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from skillforge.executive_teaching import TeachingMission, CONDITIONS
from skillforge.fresh_robot import motor_artifact
from skillforge.robot_codec import encode_action, decode_action


def main():
    folder = Path(tempfile.mkdtemp(prefix='wspl-fresh-physics-'))
    artifact = folder/'motors.json'
    artifact.write_text(json.dumps(motor_artifact()))
    checks = []
    for i, condition in enumerate(CONDITIONS):
        mission = TeachingMission(artifact, folder/condition, 991+i, 3, condition, 32)
        try:
            while not mission.episode.finished:
                needed = mission.compensated if ((condition == 'changed') or (
                    condition == 'switch' and mission.changed) or (
                    condition == 'return' and not mission.changed)) else mission.nominal
                public = mission.public()
                o = public['observation']
                if o['selected_module'] != needed:
                    action = dict(op='select', module=needed)
                else:
                    pending = [g for g in o['requested_order'] if g not in o['observed_visits']]
                    action = dict(op='reach', goal=pending[0]) if pending else dict(op='stop')
                assert decode_action(public, encode_action(public, action)) == action
                mission.execute(json.dumps(action))
            result = mission.result()
            assert result['success'], (condition, result)
            assert mission.changed == (condition in ('switch', 'return'))
            if i == 0:
                from PIL import Image
                frame = mission.backend.render('frontview')
                assert frame.shape == (256, 256, 3)
                Image.fromarray(frame).save(folder/'camera.png')
            checks.append(dict(condition=condition, success=True,
                               ticks=result['physical_ticks'], decisions=result['decisions']))
        finally:
            mission.close()
    report = dict(status='passed', engineering_only=True, model_calls=0, updates=0,
                  privileged_fixture_selector=True, conditions=checks, folder=str(folder))
    (ROOT/'docs/wspl/fresh_robot_physics_preflight.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
