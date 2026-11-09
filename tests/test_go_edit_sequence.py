import unittest
from unittest.mock import patch
import numpy as np
from skillforge.go_edit_sequence import paired_difference,validate_batch,GoEditSequence


class EditSequenceTests(unittest.TestCase):
    def test_paired_gain_loss_and_zero(self):
        a=np.array([[0,1,0,1]]*3);b=np.array([[1,0,1,1]]*3)
        r=paired_difference(a,b)
        self.assertEqual(r['difference'],.25);self.assertEqual(r['improved'],6);self.assertEqual(r['worsened'],3)
        self.assertEqual(paired_difference(a,a)['descriptive_normal_ci95'],[0.,0.])
        self.assertAlmostEqual(paired_difference(b,a)['difference'],-.25)

    def test_unmatched_or_invalid_outcomes_rejected(self):
        with self.assertRaises(ValueError):paired_difference(np.zeros((3,4)),np.zeros((3,5)))
        with self.assertRaises(ValueError):paired_difference(np.zeros((3,4)),np.full((3,4),2))
        for terminal,invalid,reward in [(False,False,1),(True,True,1),(True,False,0)]:
            with self.assertRaises(ValueError):validate_batch(dict(terminal=np.array([terminal]),invalid=np.array([invalid]),reward=np.array([reward])),[0])

    def test_checkpoint_hash_and_missing_sequence_refused(self):
        with self.assertRaises(ValueError):GoEditSequence({},'unused')
        rows={f'edit_{i}':dict(path='unused',sha256='expected') for i in range(7)}
        with patch('skillforge.go_edit_sequence.sha',return_value='wrong'):
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):GoEditSequence(rows,'unused')
