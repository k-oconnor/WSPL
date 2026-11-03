"""Synthetic native-edit parser checks; no model calls or game evaluation."""
import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from skillforge.native_weight_edits import NativeEditedPolicy,apply_statement


@unittest.skipUnless(importlib.util.find_spec('numpy'),'optional NumPy unavailable')
class NativeEditTests(unittest.TestCase):
    def test_multiline_slices_broadcast_and_order(self):
        p=NativeEditedPolicy()
        result=p.apply_answer('''```edits
W1[0:2, 0:2] = [[1, 2],
 [3, 4]]
b1[0:2] = -1
W1[:, 1] *= 2
W1[0, 0] += 2
W1[1, 0] -= 1
```''')
        self.assertEqual(result['applied'],5)
        self.assertEqual(p.weights['actor/dense1/kernel'][:2,:2].tolist(),[[3,4],[2,8]])
        self.assertEqual(p.weights['actor/dense1/bias'][:2].tolist(),[-1,-1])
        with self.assertRaises(ValueError):p.apply_answer('```edits\nb1[0]=1\n```')

    def test_last_block_and_invalid_statements_are_retained_without_execution(self):
        p=NativeEditedPolicy();r=p.apply_answer('''```edits
b1[0]=99
```
```edits
b1[0]=3
__import__('os').system('NO')
b1[0:2]=[1,2,3]
b1[0]+=1
```''')
        self.assertEqual((r['applied'],r['skipped']),(2,2))
        self.assertEqual(p.weights['actor/dense1/bias'][0],4)
        self.assertEqual(p.weights['actor/dense1/bias'][1],0)

    def test_unsupported_values_indices_and_overflows_are_rejected(self):
        for statement in ['b1[0]=True','b1[0]=float("nan")','b1[0]=1e3','b1[0]=0x10',
                          'b1[0]=[x for x in []]','b1[::0]=1','W1[0]=1','b1[999]=1',
                          'b1[0.1]=1','b1[0]=1000001','b1[0]=1; b1[1]=2']:
            p=NativeEditedPolicy();before=p.tensor_identity()
            with self.assertRaises((ValueError,IndexError,SyntaxError)):
                apply_statement(p.weights,statement)
            self.assertEqual(before,p.tensor_identity())

    def test_missing_or_oversized_blocks_are_invalid(self):
        with self.assertRaises(ValueError):NativeEditedPolicy().apply_answer('no weights')
        with self.assertRaises(ValueError):NativeEditedPolicy().apply_answer('```edits\n'+('b1[0]=1\n'*4097)+'```')


if __name__=='__main__':unittest.main()
