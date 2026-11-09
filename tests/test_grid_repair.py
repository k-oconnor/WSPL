import unittest,json
from pathlib import Path
import numpy as np
from skillforge.grid_repair import prepare_data,larger_domain,validate,weights,load_weights
from skillforge.grid_domain import enumerate_domain,prepare_sets
from skillforge.grid_copy import make_data
from skillforge.grid_movement import prepare_data as movement_data
from skillforge.relu_adapter import ResidualReLU


class RepairTests(unittest.TestCase):
    def test_new_reserved_and_shared_acquisition(self):
        copy=make_data([r['input'] for r in enumerate_domain()]);copy.update(demonstrations=prepare_sets()['demonstrations'],historical_copy=prepare_sets()['copy'])
        previous=movement_data(prepare_sets(),copy);acquired,reserved=prepare_data(previous,copy)
        self.assertEqual(len(acquired),96);self.assertEqual({k:len(v) for k,v in reserved.items()},{'fresh_L':128,'larger':64,'copy':64})
        old={tuple(r['input']) for rows in previous.values() for r in rows};fresh=set()
        for rows in reserved.values():
            for r in rows:self.assertNotIn(tuple(r['input']),old);self.assertNotIn(tuple(r['input']),fresh);fresh.add(tuple(r['input']))
        for key in ('fresh_L','larger'):
            for d in ('up','down','left','right'):
                for c in (1,2):self.assertEqual(sum(r['direction']==d and r['color']==c for r in reserved[key]),len(reserved[key])//8)

    def test_larger_rule(self):
        rows=larger_domain();self.assertEqual(len(rows),len({tuple(r['input']) for r in rows}))
        for row in rows:
            a,b=row['input'],row['output'];self.assertEqual(a.index(3),b.index(3));self.assertEqual(a.count(row['color']),b.count(row['color']))
            old=sorted(divmod(i,4) for i,c in enumerate(a) if c==row['color']);new=sorted(divmod(i,4) for i,c in enumerate(b) if c==row['color'])
            diff={(r1-r,c1-c) for (r,c),(r1,c1) in zip(old,new)};self.assertEqual(len(diff),1);dr,dc=diff.pop();self.assertEqual(abs(dr)+abs(dc),1)

    def test_edit_contract(self):
        a=ResidualReLU();bank=weights(a);bank['H']=np.zeros((2,16,576),np.float32);validate(bank,bank)
        candidate={k:v.copy() for k,v in bank.items()};candidate['H'][0,0,0]=1
        with self.assertRaises(ValueError):validate(candidate,bank)
        candidate={k:v.copy() for k,v in bank.items()};candidate['write.bias'][:]=100
        with self.assertRaises(ValueError):validate(candidate,bank)
        candidate={k:v.copy() for k,v in bank.items()};candidate['write.bias'][:]=.01;validate(candidate,bank);load_weights(a,candidate)
        self.assertTrue(np.array_equal(weights(a)['write.bias'],candidate['write.bias']))

if __name__=='__main__':unittest.main()
