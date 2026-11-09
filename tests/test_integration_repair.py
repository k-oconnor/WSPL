import unittest,copy
import torch
from skillforge.capability_bridge import lowrank,modules
from skillforge.integration_repair import scaled,choose_strength,assess


def report(loss,move=10,copy6=10):
    return dict(prose_nll=loss,grids={k:dict(exact=v,n=10) for k,v in [('move',move),('novel',move),('larger',move),('copy6',copy6)]})


class IntegrationTests(unittest.TestCase):
    def test_scaling_is_whole_branch_and_parent_unchanged(self):
        a=lowrank('0_q',rank=1);arrays=a.arrays();arrays['b'][:]=2
        from skillforge.wsp_artifact import TensorArtifact
        p=dict(recipe=dict(lr=.001,copy_fraction=.25,changed_cell_weight=1.,steps=128),branches={'0_q':TensorArtifact.from_arrays(arrays,graph=a.to_dict()['graph']).to_dict()})
        old=copy.deepcopy(p);x=torch.randn(2,3,576)
        b=modules(p,'cpu');half=modules(scaled(p,.5),'cpu');zero=modules(scaled(p,0),'cpu')
        self.assertTrue(torch.equal(half['0_q'](x),b['0_q'](x)*.5))
        self.assertEqual(float(zero['0_q'](x).abs().max()),0)
        self.assertEqual(p,old)

    def test_control_selection_retains_skill(self):
        rows={'1.0':report(5),'0.75':report(4,9),'0.5':report(3,8),'0.25':report(2,0)}
        self.assertEqual(choose_strength(rows),'0.75')
        rows['0.75']=report(4,9,8)
        self.assertEqual(choose_strength(rows),'1.0')

    def test_tradeoff_not_just_language_recovery(self):
        r=dict(foundation=report(2,0),original=report(6),scalar=report(5,9),repair=report(3,9))
        self.assertTrue(assess(r)['integration_pass']);self.assertTrue(assess(r)['beyond_scalar_control'])
        r['repair']=report(2,0)
        self.assertFalse(assess(r)['integration_pass'])
        r['repair']=r['scalar']
        self.assertFalse(assess(r)['beyond_scalar_control'])

    def test_absent_gap_is_not_success(self):
        r={k:report(2) for k in ('foundation','original','scalar','repair')}
        self.assertIsNone(assess(r)['prose_gap_recovered']);self.assertFalse(assess(r)['integration_pass'])
