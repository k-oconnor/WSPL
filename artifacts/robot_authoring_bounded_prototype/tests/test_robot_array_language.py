import unittest
import numpy as np
from skillforge.robot_array_language import Calculator


class ArrayLanguageTests(unittest.TestCase):
    def test_bounded_native_assignment_and_loop(self):
        original = np.zeros((4, 2), np.float32)
        out, receipt = Calculator({'B': original}).run(
            'for i in range(4):\n B[i,0]=i/4\n B[i,1]=-i/8', ['B'])
        np.testing.assert_array_equal(out['B'], [[0, 0], [.25, -.125], [.5, -.25], [.75, -.375]])
        self.assertEqual(receipt['loop_iterations'], 4)
        self.assertFalse(original.any())

    def test_ridge_matches_direct_solution(self):
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=(4, 8)), rng.normal(size=(4, 2))
        out, _ = Calculator({'X': x, 'Y': y}).run('C=ridge(X,Y,0.1)', ['C'])
        expected = np.linalg.solve(x.T@x+.1*np.eye(8), x.T@y)
        np.testing.assert_allclose(out['C'], expected, rtol=1e-6, atol=1e-7)

    def test_rejects_host_access_and_unbounded_programs(self):
        for source in ['import os', 'A=open("secret")', 'A=np.load("secret")',
                       'A=X.__class__', 'A=__import__("os")', 'while True: pass',
                       'for i in range(1000000):\n A=1', 'A=zeros([1000000,1000000])',
                       'A=1/0', 'A=[x for x in X]', 'A=2**1000000']:
            with self.subTest(source=source), self.assertRaises((ValueError, FloatingPointError)):
                Calculator({'X': np.zeros((2, 2))}).run(source, ['A'])

    def test_rejects_broadcast_before_allocating(self):
        with self.assertRaisesRegex(ValueError, 'Broadcast'):
            Calculator({'X': np.ones((100, 1)), 'Y': np.ones((1, 100))}, max_elements=100).run('A=X+Y', ['A'])

    def test_failure_does_not_change_input(self):
        x = np.zeros((2, 2))
        with self.assertRaises(ValueError):
            Calculator({'X': x}).run('X[:,:]=1\nA=open("x")', ['X'])
        self.assertFalse(x.any())


if __name__ == '__main__':
    unittest.main()
