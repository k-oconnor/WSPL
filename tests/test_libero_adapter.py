"""Simulator-independent bridge faults; optional NumPy, no MuJoCo imported."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
try:
    import numpy as np
except ImportError:
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skillforge.libero_adapter import LiberoMotorAdapter
from skillforge.motor import SensorUnavailable


def fake_env():
    # Native geometry IDs: object 0, left finger/pad 1/2, right finger/pad 3/4,
    # table 5. Pads have 4 mm half-thickness along jaw direction.
    names = {"obj": 0, "left": 1, "lp": 2, "right": 3, "rp": 4, "table": 5}
    model = NS(geom_name2id=names.__getitem__, geom_type=[6]*6,
               geom_size=np.array([[.008, .004, .008]]*6))
    data = NS(ncon=0, contact=[], time=0., geom_xpos=np.array([[0., 0., 0.]]*6),
              geom_xmat=np.array([np.eye(3).flatten()]*6),
              get_site_xvelp=lambda name: np.array([.1, .2, .3]))
    data.geom_xpos[2] = [0, -.03, 0]; data.geom_xpos[4] = [0, .03, 0]
    gripper = type("PandaGripper", (), {})()
    gripper.important_geoms = {"left_finger": ["left", "lp"], "right_finger": ["right", "rp"],
                               "left_fingerpad": ["lp"], "right_fingerpad": ["rp"]}
    gripper.important_sites = {"grip_site": "grip_site"}
    controller = type("OperationalSpaceController", (), {})()
    controller.use_delta = controller.use_ori = True
    controller.impedance_mode = "fixed"
    controller.input_min = -1; controller.input_max = 1
    controller.output_min = [-.05]*3 + [-.5]*3
    controller.output_max = [.05]*3 + [.5]*3
    base = NS(robots=[NS(gripper=gripper, controller=controller)], control_freq=20, action_dim=7,
              sim=NS(data=data, model=model), get_object=lambda name: NS(contact_geoms=["obj"]))
    obs = {"robot0_eef_pos": [0, 0, 0], "test_pos": [1, 2, 3]}
    env = NS(env=base, commands=[], done=False)
    def step(action):
        env.commands.append(action.copy()); data.time += .05
        return obs, 1., env.done, {}
    env.step = step
    return env, obs


@unittest.skipIf(np is None, "optional NumPy adapter dependency unavailable")
class LiberoAdapterTests(unittest.TestCase):
    def test_independent_contacts_are_object_specific_and_symmetric(self):
        env, obs = fake_env(); d = env.env.sim.data
        d.contact = [NS(geom1=5, geom2=2, dist=-.001), NS(geom1=0, geom2=1, dist=-.001),
                     NS(geom1=4, geom2=0, dist=-.001)]
        d.ncon = 3
        a = LiberoMotorAdapter(env, "test", (2, 3, 4)); f = a.observe(obs)
        self.assertTrue(f.left_contact and f.right_contact)
        self.assertAlmostEqual(f.aperture, .052)
        d.contact[1].dist = .001  # proximity is not physical contact
        self.assertFalse(a.observe(obs).left_contact)

    def test_command_is_exactly_one_native_step_and_terminal_is_guarded(self):
        env, obs = fake_env(); a = LiberoMotorAdapter(env, "test", (0, 0, 0))
        with self.assertRaises(RuntimeError): a.step((0, 0, 0, 0))
        a.observe(obs); env.done = True
        f, log = a.step((1, 0, 0, -1))
        self.assertEqual(len(env.commands), 1)
        self.assertAlmostEqual(f.time, .05)
        np.testing.assert_allclose(env.commands[0], [.4, 0, 0, 0, 0, 0, -1])
        self.assertTrue(log["native_done"])
        with self.assertRaises(RuntimeError): a.step((0, 0, 0, 0))

    def test_failed_sensor_read_disarms_motion(self):
        env, obs = fake_env(); a = LiberoMotorAdapter(env, "test", (0, 0, 0))
        a.observe(obs)
        with self.assertRaises(SensorUnavailable): a.observe({})
        with self.assertRaises(RuntimeError): a.step((0, 0, 0, 0))
        self.assertFalse(env.commands)

    def test_nonfinite_physical_geometry_not_silently_zero(self):
        env, obs = fake_env(); a = LiberoMotorAdapter(env, "test", (0, 0, 0))
        env.env.sim.data.geom_xpos[2, 0] = float("nan")
        with self.assertRaises(SensorUnavailable): a.observe(obs)

    def test_mismatched_controller_rejected_before_motion(self):
        env, obs = fake_env(); env.env.robots[0].controller.use_delta = False
        with self.assertRaises(ValueError): LiberoMotorAdapter(env, "test", (0, 0, 0))
        self.assertFalse(env.commands)

    def test_motor_budget_does_not_trust_native_horizon(self):
        env, obs = fake_env(); a = LiberoMotorAdapter(env, "test", (0, 0, 0), max_steps=1)
        a.observe(obs); _, log = a.step((0, 0, 0, 0))
        self.assertFalse(log["native_done"])
        self.assertTrue(log["budget_exhausted"])
        with self.assertRaises(RuntimeError): a.step((0, 0, 0, 0))
        self.assertEqual(len(env.commands), 1)


if __name__ == "__main__": unittest.main()
