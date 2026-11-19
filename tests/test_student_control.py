import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from skillforge.factory_observation import ObservationSession
from skillforge.factory_pilot import FactoryStudy
from skillforge.student_control import StudentExecutiveEpisode,prompt


class SensorBackend:
    def __init__(self):self.p=np.zeros(3);self.t=0.;self.calls=0
    def sensors(self):return dict(timestamp_s=self.t,joint_position_rad=np.r_[self.p,np.zeros(4)],joint_velocity_rad_s=np.zeros(7),gripper_joint_position_m=np.zeros(2))
    def forward_kinematics(self,q):return dict(position_m=q[:3],quaternion_xyzw=[0,0,0,1])
    def step(self,c):self.p+=np.asarray(c[:3])*.05;self.t+=.05;self.calls+=1


class StudentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.backend=SensorBackend()
        self.session=ObservationSession(self.backend,Path(self.temp.name)/'s',7,max_steps=100)
        motor=FactoryStudy().original
        self.e=StudentExecutiveEpisode(self.session,{'tool_17':motor,'tool_92':motor},{'goal_8':[.02,0,0],'goal_3':[0,.02,0]})
    def tearDown(self):self.temp.cleanup()
    def do(self,**a):return self.e.execute(json.dumps(a))
    def test_no_startup_probe_or_selection(self):
        self.assertEqual(self.backend.calls,0);self.assertIsNone(self.e.selected)
        result=self.do(op='reach',goal='goal_8')
        self.assertEqual(result['status'],'invalid_action');self.assertEqual(self.backend.calls,0)
    def test_probe_does_not_choose_controller(self):
        self.do(op='probe',command=[.1,0,0],ticks=3)
        self.assertEqual(self.backend.calls,3);self.assertIsNone(self.e.selected)
        self.assertEqual(self.e.observed_visits,[])
    def test_select_is_explicit_and_nonphysical(self):
        self.do(op='select',module='tool_92');self.assertEqual(self.e.selected,'tool_92')
        self.assertEqual(self.backend.calls,0)
    def test_reach_does_not_advance_or_select(self):
        self.do(op='select',module='tool_17');self.do(op='reach',goal='goal_8')
        self.assertLessEqual(self.backend.calls,10);self.assertEqual(self.e.selected,'tool_17')
        self.assertNotIn('goal_3',self.e.observed_visits);self.assertFalse(self.e.finished)
    def test_stop_does_not_complete_goals(self):
        self.do(op='stop');self.assertTrue(self.e.finished);self.assertEqual(self.e.observed_visits,[])
    def test_invalid_action_does_not_repair(self):
        result=self.do(op='select',module='unknown');self.assertEqual(result['status'],'invalid_action')
        self.assertIsNone(self.e.selected);self.assertEqual(self.backend.calls,0)
    def test_budget_prevents_unbounded_actions(self):
        self.e.max_decisions=1
        self.do(op='select',module='tool_17');self.do(op='probe',command=[.1,0,0],ticks=3)
        self.assertTrue(self.e.finished);self.assertEqual(self.backend.calls,0)
    def test_prompt_has_no_runtime_teacher(self):
        text=prompt(self.e.observe(),{})
        self.assertIn('No external teacher',text);self.assertNotIn('mapping',text)


if __name__=='__main__':unittest.main()
