import unittest
from fractions import Fraction
import torch
from skillforge.controller_author import construct,teacher,make_spec
from skillforge.local_mechanism import calibration,author_prompt,fit_replacement


class MechanismTests(unittest.TestCase):
    def test_calibration_and_author_share_signed_rule_and_examples(self):
        spec=dict(kind='move',gain=-.4,speed=.04)
        rows=calibration(spec,count=8)
        self.assertEqual(rows,calibration(spec,count=8))
        self.assertTrue(all(r['inputs']['x']*r['outputs']['vx']<=0 for r in rows))
        text=author_prompt(spec,rows)
        self.assertIn('negative sign',text)
        self.assertIn('27 scalar',text)

    def test_synthetic_fit_export_reproducible_and_input_unchanged(self):
        spec=make_spec('move',20);base=construct(teacher(spec),spec);identity=base.sha256
        rows=[dict(inputs=dict(x=0.,y=0.),outputs=dict(vx=0.,vy=0.,grip=0.))]*4
        rng=torch.get_rng_state().clone()
        a,metrics=fit_replacement(base,rows,seed=9,updates=4)
        b,_=fit_replacement(base,rows,seed=9,updates=4)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(a.sha256,b.sha256)
        self.assertEqual(base.sha256,identity)
        self.assertEqual(a.to_dict()['allocation']['widths'],base.to_dict()['allocation']['widths'])
        self.assertEqual(set(a.evaluate(dict(x=Fraction(0),y=Fraction(0)))),{'vx','vy','grip'})
        self.assertEqual(metrics['updates'],4)


if __name__=='__main__':unittest.main()
