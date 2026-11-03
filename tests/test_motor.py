"""Deterministic sensor/observer checks; no robot simulation or training."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skillforge.motor import MotorFrame, OSCCommand, ObserverConfig, SkillObserver, SensorUnavailable


def frame(t=0., contacts=False, **kw):
    return MotorFrame(t*.05, kw.get("eef", (0., 0., 0.)), kw.get("obj", (0., 0., 0.)),
                      kw.get("target", (0., 0., 0.)), kw.get("velocity", (0., 0., 0.)),
                      .04, contacts, contacts)


class MotorTests(unittest.TestCase):
    def test_channel_units_and_order(self):
        f = MotorFrame(0., (1, 2, 3), (2, 4, 6), (0, 0, 0), (.1, .2, .3), .04, True, False)
        self.assertEqual(f.channels, (1., 2., 3., -1., -2., -3., .1, .2, .3, .04, 1., 0.))

    def test_unknown_nonfinite_and_malformed_sensors_rejected(self):
        for values in ({"velocity": (float("nan"), 0, 0)}, {"aperture": -1},
                       {"left_contact": None}, {"eef": (1, 2)}, {"time": float("inf")}):
            with self.subTest(values=values), self.assertRaises(SensorUnavailable):
                replace(frame(), **values)

    def test_native_scaling_orientation_and_gripper(self):
        for actual, expected in zip(OSCCommand().encode((1, -1, .5, -1)), (.4, -.4, .2, 0, 0, 0, -1)):
            self.assertAlmostEqual(actual, expected)
        for cmd in ((2, 0, 0, 0), (0, 0, 0), (0, 0, 0, float("nan"))):
            with self.assertRaises(ValueError):
                OSCCommand().encode(cmd)
        with self.assertRaises(ValueError):
            OSCCommand(.06, .05)

    def test_approach_requires_stability_and_low_speed(self):
        o = SkillObserver(); o.begin("a", "approach", frame())
        self.assertFalse(o.update(frame(1.))["approach_done"])
        self.assertFalse(o.update(frame(2., velocity=(1, 0, 0)))["approach_done"])
        self.assertFalse(o.update(frame(3.))["approach_done"])
        self.assertFalse(o.update(frame(4.))["approach_done"])
        self.assertTrue(o.update(frame(5.))["approach_done"])
        self.assertFalse(o.update(frame(6.))["approach_done"])

    def test_close_never_claims_pickup(self):
        o = SkillObserver(); o.begin("c", "close", frame())
        for t in (1., 2., 3.):
            out = o.update(frame(t, True))
        self.assertTrue(out["close_done"])
        self.assertFalse(out["lift_done"])
        self.assertFalse(out["elevated"])

    def test_lift_requires_elevation_and_uninterrupted_contact(self):
        o = SkillObserver(); o.begin("l", "lift", frame(contacts=True))
        self.assertFalse(o.update(frame(1., True))["lift_done"])
        for t in (2., 3., 4.):
            out = o.update(frame(t, True, obj=(0, 0, .06)))
        self.assertTrue(out["lift_done"])
        self.assertFalse(o.update(frame(5., True, obj=(0, 0, .06)))["lift_done"])

    def test_recontact_does_not_hide_lift_failure(self):
        o = SkillObserver(); o.begin("l", "lift", frame(contacts=True))
        self.assertTrue(o.update(frame(1.))["contact_lost"])
        for t in range(2, 8):
            self.assertFalse(o.update(frame(float(t), True, obj=(0, 0, .1)))["lift_done"])

    def test_no_contact_at_start_cannot_certify_lift(self):
        o = SkillObserver(); o.begin("l", "lift", frame())
        for t in range(1, 6):
            self.assertFalse(o.update(frame(float(t), True, obj=(0, 0, .1)))["lift_done"])

    def test_fresh_frame_and_fixed_binding_required(self):
        o = SkillObserver(); o.begin("a", "approach", frame())
        with self.assertRaises(SensorUnavailable): o.update(frame())
        o.begin("b", "approach", frame())
        with self.assertRaises(SensorUnavailable): o.update(frame(1., target=(1, 0, 0)))
        with self.assertRaises(ValueError): o.begin("a", "approach", frame())
        with self.assertRaises(ValueError): SkillObserver().update(frame())

    def test_missing_tick_aborts_instead_of_preserving_stability(self):
        o = SkillObserver(); o.begin("a", "lift", frame(contacts=True))
        with self.assertRaises(SensorUnavailable):
            o.update(frame(2., True, obj=(0, 0, .1)))
        with self.assertRaises(ValueError):
            o.update(frame(3., True, obj=(0, 0, .1)))

    def test_new_invocation_resets_baseline_and_pulse(self):
        o = SkillObserver(ObserverConfig(stable_ticks=1))
        o.begin("l1", "lift", frame(contacts=True))
        self.assertTrue(o.update(frame(1., True, obj=(0, 0, .06)))["lift_done"])
        o.begin("l2", "lift", frame(2., True, obj=(0, 0, .06)))
        self.assertFalse(o.update(frame(3., True, obj=(0, 0, .06)))["lift_done"])

    def test_invalid_thresholds(self):
        for kw in ({"stable_ticks": 0}, {"stable_ticks": True}, {"lift_height_m": float("nan")}):
            with self.assertRaises(ValueError): ObserverConfig(**kw)


if __name__ == "__main__":
    unittest.main()
