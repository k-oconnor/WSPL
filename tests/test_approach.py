"""Synthetic bridge fixtures test study bookkeeping, not physical reliability."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
try:
    import torch
    import numpy as np
except ImportError:
    torch = None
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
if torch is not None:
    from skillforge.approach import ApproachStudy, rollout, check_feasibility, audit_initialization, native_horizon
    from test_libero_adapter import fake_env
    from run_approach import fidelity


def fixture(reset_error=False, horizon=None):
    env, old = fake_env()
    obs = {'robot0_eef_pos': np.array([0., 0., .3]), 'cream_cheese_1_pos': np.zeros(3)}
    data = env.env.sim.data
    velocity = np.zeros(3)
    env.sim = env.env.sim
    env.seed = lambda seed: None
    env.native_calls = 0
    def reset():
        if reset_error: raise ValueError('bad layout')
        data.time = 0.; env.native_calls = 0; obs['robot0_eef_pos'][:] = (0, 0, .3)
        return obs
    env.env.reset = reset
    data.get_site_xvelp = lambda name: velocity
    def step(action):
        if horizon is not None and env.native_calls >= horizon:
            raise ValueError('executing action in terminated episode')
        env.native_calls += 1
        movement = np.asarray(action[:3])*.05
        obs['robot0_eef_pos'] += movement
        velocity[:] = movement/.05
        data.time += .05
        return obs, 0., False, {}
    env.step = step
    return env


@unittest.skipIf(torch is None, 'optional PyTorch/NumPy dependencies unavailable')
class ApproachTests(unittest.TestCase):
    def test_reference_vs_zero_and_costed_paired_reset(self):
        study = ApproachStudy()
        ref = rollout(fixture(), study, 'reference', 1, settling_steps=2)
        zero = rollout(fixture(), study, 'zero', 1, settling_steps=2)
        self.assertTrue(check_feasibility(ref, zero))
        self.assertEqual(ref['preparation_calls'], 2)
        self.assertEqual(ref['reset_calls'], 1)
        self.assertEqual(ref['motor_calls'], len(ref['trace']))
        self.assertEqual(zero['motor_calls'], 120)
        self.assertEqual(zero['status'], 'timeout')

    def test_failed_reset_is_recorded_not_retried(self):
        result = rollout(fixture(reset_error=True), ApproachStudy(), 'reference', 1)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['reset_calls'], 1)
        self.assertEqual(result['motor_calls'], 0)

    def test_native_horizon_includes_preparation_and_full_skill_budget(self):
        study = ApproachStudy()
        old = rollout(fixture(horizon=120), study, 'zero', 1)
        self.assertEqual(old['status'], 'error')
        self.assertEqual(old['motor_calls'], 101)  # includes rejected call
        env = fixture(horizon=native_horizon(120, 20))
        fixed = rollout(env, study, 'zero', 1)
        self.assertEqual(fixed['status'], 'timeout')
        self.assertEqual(fixed['motor_calls'], 120)
        self.assertEqual(fixed['preparation_calls'], 20)
        self.assertEqual(env.native_calls, 140)

    def test_fidelity_accepts_real_adapter_numpy_scalar_input_ports(self):
        env,obs=fake_env()
        from skillforge.libero_adapter import LiberoMotorAdapter
        f=LiberoMotorAdapter(env,'test',(0,0,0)).observe(obs)
        self.assertIsInstance(f.channels[9],np.floating)
        result=fidelity(ApproachStudy().models['rmv_authored'],[f.channels])
        self.assertLess(result,2e-6)

    def test_feasibility_rejects_mismatched_or_trivial_controls(self):
        a = dict(start_sha256='a', success=True)
        self.assertFalse(check_feasibility(a, a))
        with self.assertRaises(ValueError): check_feasibility(a, dict(start_sha256='b', success=False))

    def test_arms_have_separate_parameters_and_audit_leaves_weights_unchanged(self):
        study = ApproachStudy()
        ptrs = [p.data_ptr() for m in study.models.values() for p in m.parameters()]
        self.assertEqual(len(ptrs), len(set(ptrs)))
        for name in ('mlp_adamw', 'rmv_adamw'):
            m = study.models[name]; old = {n:p.detach().clone() for n,p in m.named_parameters()}
            g = torch.Generator().manual_seed(19)
            x = m.lower+(m.upper-m.lower)*torch.rand(32, 12, generator=g)
            result = audit_initialization(m, x)
            self.assertGreater(result['input_gradient_l1'], 0)
            self.assertTrue(all(torch.equal(old[n], p) for n,p in m.named_parameters()))

    def test_adamw_update_projection_and_state_roundtrip(self):
        # One optimizer transaction checks checkpoint compatibility; no fitting.
        import io
        study = ApproachStudy()
        for name in ('mlp_adamw', 'rmv_adamw'):
            m = study.models[name]; opt = torch.optim.AdamW(m.parameters(), lr=.003, weight_decay=0)
            x = ((m.lower+m.upper)/2).repeat(2, 1)
            m(x).square().mean().backward(); opt.step(); m.project_()
            b = io.BytesIO(); torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict()), b); b.seek(0)
            state = torch.load(b, weights_only=True)
            clone = type(m)(); clone.load_state_dict(state['model'])
            other = torch.optim.AdamW(clone.parameters()); other.load_state_dict(state['optimizer'])
            self.assertTrue(torch.equal(m(x), clone(x)))
            self.assertEqual(len(other.state), len(opt.state))

    def test_direct_scientific_runner_is_disabled(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy(); env['PYTHONPATH'] = str(root/'src')
        p = subprocess.run([sys.executable, str(root/'experiments/run_approach.py')],
                           cwd=root, env=env, capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('Direct training is disabled', p.stderr)


if __name__ == '__main__': unittest.main()
