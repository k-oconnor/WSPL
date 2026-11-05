"""The vectorized write path must equal the original element-wise exact path.

Each case runs the same write twice: once normally and once with the fast path
disabled, which is exactly the original implementation. Words, rounding count,
first-64 rounding examples, write records, touched indices, charged operations
and error messages must all agree.
"""
from fractions import Fraction
import random
import unittest

import numpy as np

from skillforge.wsp_artifact import TensorArtifact, Bits
from skillforge.wsp_editor import Builder, plan
from skillforge.wsp_values import Limits

SPECIAL_BITS = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x00800000, 0x3f800000, 0xbf800000,
                0x7f7fffff, 0xff7fffff, 0x3eaaaaab, 0x4b800000, 0x33800000]


def rational(rng):
    kind = rng.randrange(10)
    if kind == 0:
        return Fraction(0)
    if kind == 1:
        return Fraction(rng.randint(-5, 5), rng.choice([1, 3, 7, 40, 1000]))  # Non-dyadic.
    if kind == 2:
        return Fraction(rng.choice([-1, 1])*rng.randint(1, 2**30), 2**rng.randint(0, 60))  # Wide dyadic.
    if kind == 3:
        return Fraction(rng.choice([-1, 1]), 2**rng.randint(140, 160))  # Subnormal/underflow, both signs.
    if kind == 4:
        return Fraction(rng.choice([-1, 1])*(2**128 - rng.choice([0, 2**103, 2**104])))  # Overflow boundary.
    if kind == 5:
        return Fraction(rng.choice([-1, 1])*(2*rng.randint(0, 2**23)+1), 2**25)  # Ties at FP32 rounding.
    if kind == 6:
        return Fraction(rng.choice([-1, 1])*rng.randint(1, 2**53), 2**rng.randint(0, 80))  # 53-bit significands.
    return Fraction(rng.randint(-8, 8), rng.choice([1, 2, 4, 8, 16]))


def item(rng):
    return Bits(rng.choice(SPECIAL_BITS + [rng.randrange(0, 0x7f800000) | rng.choice([0, 0x80000000])])) if rng.random() < .35 else rational(rng)


def parent_artifact(rng, shape):
    words = np.array([rng.choice(SPECIAL_BITS + [rng.randrange(0, 0x7f800000) | rng.choice([0, 0x80000000])])
                      for _ in range(int(np.prod(shape)))], dtype=np.uint32).reshape(shape)
    return TensorArtifact.from_arrays({'A': words.view(np.float32), 'B': np.zeros((3,), np.float32)})


def run(parent, steps, limits, fast):
    builder = Builder(parent, {}, limits)
    if not fast:
        builder._fast_words = lambda *a: None
    outcome = None
    try:
        for name, selection, value, op in steps:
            builder.write(name, selection, value, op, 'span')
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        outcome = f'{type(e).__name__}: {e}'
    touched = {k: np.unique(np.concatenate(v)).tolist() for k, v in builder.touched.items()}
    return dict(outcome=outcome, arrays={k: a.tolist() for k, a in builder.arrays.items()}, rounded=builder.rounded_count,
                rounding=builder.rounding, writes=builder.writes, touched=touched, operations=builder.arithmetic.operations)


class WritePathTests(unittest.TestCase):
    def test_random_writes_match_reference(self):
        rng = random.Random(20261105)
        shape = (6, 5)
        selections = [(slice(None), slice(None)), (2, slice(1, 4)), (slice(None, None, -2), 3), (1, 1), (slice(0, 0), slice(None)),
                      (slice(4, 1, -1), slice(0, 5, 2))]
        for case in range(400):
            parent = parent_artifact(rng, shape)
            limits = Limits(max_integer_bits=rng.choice([4096, 4096, 4096, 64, 1099]))
            steps = []
            for _ in range(rng.randint(1, 4)):
                selection = rng.choice(selections)
                target = np.zeros(shape)[selection]
                form = rng.randrange(3)
                if form == 0 or target.ndim == 0:
                    value = np.array(item(rng), dtype=object)
                elif form == 1:
                    value = np.array([item(rng) for _ in range(target.shape[-1])], dtype=object)
                else:
                    value = np.array([item(rng) for _ in range(target.size)], dtype=object).reshape(target.shape)
                steps.append(('A', selection, value, rng.choice(['=', '+=', '-=', '*='])))
            with self.subTest(case=case):
                self.assertEqual(run(parent, steps, limits, True), run(parent, steps, limits, False))

    def test_rounding_examples_cap_and_order(self):
        parent = TensorArtifact.from_arrays({'A': np.zeros((10, 10), np.float32)})
        third = np.array(Fraction(1, 3), dtype=object)
        seventh = np.array(Fraction(1, 7), dtype=object)
        # Both writes round on the fast path; the 64-example cap spans writes.
        steps = [('A', (slice(0, 5), slice(None)), third, '='), ('A', (slice(5, 10), slice(None)), seventh, '='),
                 ('A', (slice(None), slice(None)), third, '+=')]
        fast, reference = run(parent, steps, Limits(), True), run(parent, steps, Limits(), False)
        self.assertEqual(fast, reference)
        self.assertEqual(len(fast['rounding']), 64)
        self.assertEqual(fast['rounded'], 200)

    def test_signed_zero_rules(self):
        neg = np.array([-0.0, 1.0, -1.0, 2.0**-149], np.float32)
        parent = TensorArtifact.from_arrays({'A': neg})
        cases = [np.array(Bits(0x80000000), dtype=object), np.array(Fraction(0), dtype=object),
                 np.array(Fraction(-1, 2**200), dtype=object), np.array(Fraction(1, 2**151), dtype=object)]
        for value in cases:
            for op in ('=', '+=', '-=', '*='):
                with self.subTest(value=value, op=op):
                    steps = [('A', (slice(None),), value, op)]
                    self.assertEqual(run(parent, steps, Limits(), True), run(parent, steps, Limits(), False))

    def test_mixed_types_with_equal_hashes(self):
        # Fraction(1), 1 and True hash equal; only the rational is a valid value.
        parent = TensorArtifact.from_arrays({'A': np.ones(3, np.float32)})
        for op in ('=', '+=', '*='):
            for value in (np.array([Fraction(1), 1, Fraction(1)], dtype=object), np.array([Fraction(1), True, 2], dtype=object),
                          np.array([Bits(0x3f800000), Fraction(1), 1], dtype=object)):
                with self.subTest(op=op, value=value):
                    steps = [('A', (slice(None),), value, op)]
                    self.assertEqual(run(parent, steps, Limits(), True), run(parent, steps, Limits(), False))

    def test_plan_level_programs_match_reference(self):
        rng = np.random.default_rng(5)
        a = rng.standard_normal((8, 8)).astype(np.float32); a[0, 0] = -0.0
        parent = TensorArtifact.from_arrays({'A': a, 'c': rng.standard_normal(8).astype(np.float32)})
        bodies = ['A[:,:]=0;', 'A[1:5,:]+=parent.A[0:4,:]; A[::2,1]*=3; c[:]-=1/3;',
                  'for i in range(8) { A[i,i]=parent.A[i,7-i]*(1/7); A[i,:]+=c[:]; }',
                  'A[0,:]=[1,2,3,4,5,6,7,1/40]; A[:,0]-=parent.A[:,1]; A[2,2]*=0;',
                  'A[0,0]=340282366920938463463374607431768211456;']
        for body in bodies:
            source = 'wspl 0.2;target b=artifact("p",sha256="'+parent.sha256+'");edit b as c mode=fixed numeric=exact_to_storage {'+body+'}'
            results = []
            for fast in (True, False):
                original = Builder._fast_words
                if not fast:
                    Builder._fast_words = lambda *args: None
                try:
                    p = plan(source, parent=parent); body_d = p.to_dict(); body_d.pop('implementation_id')
                    results.append(body_d)
                except ValueError as e:
                    results.append(str(e))
                finally:
                    Builder._fast_words = original
            with self.subTest(body=body):
                self.assertEqual(results[0], results[1])


if __name__ == '__main__':
    unittest.main()
