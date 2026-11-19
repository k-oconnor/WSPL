"""Synthetic interface checks only; no task scoring or author calls."""
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from skillforge.factory_observation import PandaBackend, ObservationSession, CAMERAS
from skillforge.factory_pilot import FactoryStudy, MOTOR_SOURCE, SELECT_SOURCE, rollout


def main():
    output=ROOT/'data/factory_preflight'
    output.mkdir(parents=True,exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='synthetic_',dir=output))
    backend=PandaBackend(123)
    try:
        session=ObservationSession(backend,folder/'observations',123)
        before=backend.env.sim.get_state().flatten().copy()
        observation=session.observe()
        assert np.array_equal(before,backend.env.sim.get_state().flatten())
        # Zero-noise nominal FK must agree with the simulator's arm site geometry.
        q=backend.sensors()['joint_position_rad']
        fk=backend.forward_kinematics(q)['position_m']
        error=float(np.max(np.abs(fk-backend.env.sim.data.site_xpos[backend._eef_site])))
        assert error<1e-9,error
        for camera in CAMERAS:
            frame=session.capture(camera)
            from PIL import Image
            image=np.asarray(Image.open(session.output/frame['path']))
            assert image.std()>2
        reply=session.act([.01,0,0,-1],2,CAMERAS,('command_start','command_end'),1)
        assert len(reply['telemetry'])==2 and len(reply['frames'])==8
        assert all(abs(b['timestamp_s']-a['timestamp_s']-.05)<1e-9 for a,b in zip([observation]+reply['telemetry'],reply['telemetry']))
        report=dict(status='passed',fk_max_error_m=error,frames=len(session.frames),
                    physics_steps=session.steps,folder=str(folder),scientific_scoring=False,model_calls=0)
        (folder/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report))
    finally: backend.close()
    # Separate engineering fixtures, never the calibration or reserved seed sets.
    study=FactoryStudy()
    repair=MOTOR_SOURCE.replace('10*x','(25/3)*y').replace('10*y','-12.5*x').replace('10*z','(100/9)*z')
    selector=SELECT_SOURCE.replace('original = 1','original = dx').replace('repaired = 0','repaired = dy')
    study.install(dict(probe=[.12,0,0],probe_ticks=5,motor=repair,selector=selector))
    fixture=[]
    for name,mat,arm in [('original',np.eye(3).tolist(),'selected'),
                         ('changed',[[0,-.8,0],[1.2,0,0],[0,0,.9]],'selected'),
                         ('reference',[[0,-.8,0],[1.2,0,0],[0,0,.9]],'reference')]:
        row=rollout(study,folder/name,124,mat,[.025,-.02,.015],arm)
        fixture.append(dict(name=name,success=row['success'],steps=row['steps'],error=row['final_error_m']))
        assert row['success'],fixture
    report['synthetic_controller_fixtures']=fixture
    (folder/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
