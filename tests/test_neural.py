"""Numerical and gradient unit checks only; no fitting or physical evaluation."""
from pathlib import Path
import itertools
import sys
import unittest
try:
    import torch
except ImportError:
    torch = None
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if torch is not None:
    from skillforge.neural import MotorMLP, MotorRMV, authored_approach, reference_command
    from skillforge.motor import CHANNELS
    from skillforge.rmv import RMVReference


@unittest.skipIf(torch is None, "optional PyTorch dependency unavailable")
class NeuralTests(unittest.TestCase):
    def test_capacity_and_local_initialization_rng(self):
        state = torch.get_rng_state().clone()
        self.assertEqual(sum(p.numel() for p in MotorMLP().parameters()), 1604)
        self.assertEqual(sum(p.numel() for p in MotorRMV().parameters()), 1588)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_random_initialization_is_input_sensitive_with_task_gradients(self):
        for cls in (MotorMLP, MotorRMV):
            m = cls(); g = torch.Generator().manual_seed(4)
            x = (m.lower + (m.upper-m.lower)*(.25+.5*torch.rand(24, 12, generator=g))).requires_grad_()
            loss = ((m(x)-reference_command(x))**2).mean()
            loss.backward()
            self.assertGreater(float(x.grad.abs().sum()), 1e-7)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            self.assertTrue(all(float(p.grad.abs().sum()) > 0 for p in m.parameters()))
        m = MotorRMV(); d = m.internals((m.lower+m.upper)/2)
        self.assertTrue(((d['clauses'] > 0) & (d['clauses'] < 1)).all())
        self.assertEqual(len(torch.unique(m.and_weights, dim=0)), 56)

    def test_analytic_models_match_teacher(self):
        for cls in (MotorMLP, MotorRMV):
            m = authored_approach(cls())
            g = torch.Generator().manual_seed(19)
            x = m.lower + (m.upper-m.lower)*torch.rand(256, 12, generator=g)
            x[:125, 3:6] = torch.tensor(list(itertools.product((-.05, -.01, 0., .01, .05), repeat=3)))
            self.assertLess(float((m(x)-reference_command(x)).abs().max()), 2e-6)

    def test_scalar_export_corners_interior_and_boundaries(self):
        for authored in (False, True):
            m = MotorRMV()
            if authored: authored_approach(m)
            g = torch.Generator().manual_seed(17)
            p = torch.cat((torch.tensor(list(itertools.product((0., 1.), repeat=12))),
                           torch.rand(128, 12, generator=g), torch.full((1, 12), .5)), 0)
            x = m.lower+(m.upper-m.lower)*p
            ref = RMVReference(m.export())
            actual = m(x).detach().tolist()
            error = max(abs(a-b) for row, y in zip(x.tolist(), actual)
                        for a, b in zip(y, ref.evaluate(dict(zip(CHANNELS, row)))['commands']))
            self.assertLess(error, 2e-6)

    def test_projection_and_invalid_graph_rejection(self):
        m = MotorRMV()
        with torch.no_grad(): m.and_weights[0, 0] = -1
        with self.assertRaises(ValueError): m.export()
        m.project_(); self.assertEqual(float(m.and_weights[0, 0]), 0)
        with torch.no_grad(): m.readout_bias[0] = float('nan')
        with self.assertRaises(ValueError): m.project_()

    def test_shapes_nonfinite_inputs_and_bounded_commands(self):
        for cls in (MotorMLP, MotorRMV):
            m = cls()
            with self.assertRaises(ValueError): m(torch.zeros(3, 11))
            with self.assertRaises(ValueError): m(torch.full((12,), float('nan')))
            y = m(torch.full((3, 12), 100.))
            self.assertTrue((y.abs() <= 1).all())


if __name__ == '__main__': unittest.main()
