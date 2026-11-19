import tempfile
import unittest
from pathlib import Path
import numpy as np
from skillforge.factory_observation import ObservationSession, CAMERAS


class Backend:
    def __init__(self): self.t = 0.; self.actions = []
    def sensors(self):
        return dict(timestamp_s=self.t, joint_position_rad=np.zeros(7),
            joint_velocity_rad_s=np.zeros(7), gripper_joint_position_m=np.zeros(2),
            hidden_fault='DO_NOT_EXPOSE', object_position=[9,9,9], success=True)
    def forward_kinematics(self, q):
        return dict(position_m=q[:3], quaternion_xyzw=[0,0,0,1])
    def step(self,c): self.t += .05; self.actions.append(list(c))
    def render(self,c): return np.full((256,256,3),int(self.t*100),dtype=np.uint8)


class TestObservation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.backend = Backend()
        self.session = ObservationSession(self.backend,Path(self.temp.name)/'session',3,max_steps=30,max_frames=20)
    def tearDown(self): self.temp.cleanup()
    def test_sensor_allowlist_and_stable_read(self):
        a=self.session.observe();self.assertEqual(a,self.session.observe())
        self.assertNotIn('hidden_fault',a);self.assertNotIn('object_position',a);self.assertNotIn('success',a)
        self.assertEqual(a['joint_position_rad'][:3],a['nominal_tool_position_m'])
        self.session.capture(CAMERAS[0]);b=self.session.observe()
        self.assertEqual(a['joint_position_rad'],b['joint_position_rad'])
    def test_prospective_transition_and_clip(self):
        r=self.session.act([.1,0,0,1],4,CAMERAS,('command_start','command_end','gripper_command_change'),2)
        self.assertEqual(len(r['frames']),10)
        self.assertEqual(len(r['telemetry']),4)
        self.assertEqual(r['frames'][0]['timestamp_s'],0.)
        self.assertEqual(r['frames'][-1]['timestamp_s'],.2)
        self.assertEqual(self.session.steps,4)
        self.assertEqual(len(self.backend.actions),4)
    def test_atomic_rejection(self):
        for kwargs in [dict(command=[1,0,0,0],ticks=2),dict(command=[0,0,0,0],ticks=True),
                       dict(command=[0,0,0,0],ticks=2,events=['grasp_success']),
                       dict(command=[0,0,0,0],ticks=20,cameras=CAMERAS,clip_stride=1)]:
            with self.assertRaises(ValueError): self.session.act(**kwargs)
        self.assertEqual(self.session.steps,0);self.assertEqual(len(self.session.frames),0)
    def test_no_implicit_camera(self):
        self.session.act([0,0,0,-1],2)
        self.assertEqual(self.session.frames,[])
    def test_bad_camera_does_not_execute(self):
        with self.assertRaises(ValueError):self.session.act([0,0,0,-1],2,('secret',))
        self.assertEqual(self.backend.actions,[])


if __name__=='__main__': unittest.main()
