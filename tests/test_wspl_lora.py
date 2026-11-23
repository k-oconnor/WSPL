"""Synthetic bridge checks only: no pretrained model, scored tasks or author calls."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from skillforge.wspl_lora import (AdapterLimits, snapshot, identity, describe,
    source_for_pair, construct_wspl, construct_native, model_state, install_state)


def parent():
    return {'model.layers.0.q_proj.lora_A':np.array([[1,2,3],[0,-1,1]],dtype=np.float32),
            'model.layers.0.q_proj.lora_B':np.zeros((4,2),dtype=np.float32),
            'model.layers.1.v_proj.lora_A':np.array([[1,-1]],dtype=np.float32),
            'model.layers.1.v_proj.lora_B':np.array([[.5],[.25]],dtype=np.float32)}


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.parent=parent();self.p='model.layers.0.q_proj'
        self.id=identity(self.parent)
        self.catalog=describe(self.parent)

    def program(self,body,projection=None):
        projection=projection or self.p
        return source_for_pair(self.catalog['projections'][projection]['pair_id'],body)

    def test_native_and_wspl_make_identical_factors(self):
        wspl,receipt=construct_wspl(self.parent,{self.p:self.program('for i in range(4) { B[i,0]=i/4; B[i,1]=-i/8; }')},expected_parent_id=self.id)
        replacement={'A':self.parent[self.p+'.lora_A'].copy(),
                     'B':np.array([[i/4,-i/8] for i in range(4)],dtype=np.float32)}
        native,_=construct_native(self.parent,{self.p:replacement},expected_parent_id=self.id)
        self.assertEqual(identity(wspl),identity(native));self.assertEqual(receipt['changed_elements'],6)
        self.assertEqual(receipt['behavioral_claim'],'none')
        self.assertEqual(identity(self.parent),self.id)
        for key in self.parent:
            if not key.startswith(self.p):np.testing.assert_array_equal(wspl[key],self.parent[key])

    def test_noop_preserves_signed_zero_and_exact_identity(self):
        self.parent[self.p+'.lora_B'][0,0]=np.float32(-0.0)
        before=identity(self.parent)
        result,receipt=construct_wspl(self.parent,{},expected_parent_id=before)
        self.assertEqual(identity(result),before);self.assertEqual(receipt['changed_elements'],0)
        self.assertTrue(np.signbit(result[self.p+'.lora_B'][0,0]))

    def test_stale_adapter_rejected_before_edits(self):
        self.parent[self.p+'.lora_A'][0,0]=9
        with self.assertRaisesRegex(ValueError,'Stale'):
            construct_wspl(self.parent,{},expected_parent_id=self.id)

    def test_stale_pair_rejected(self):
        bad=source_for_pair('0'*64,'B[:,:]=1;')
        with self.assertRaises(ValueError):construct_wspl(self.parent,{self.p:bad},expected_parent_id=self.id)
        self.assertEqual(identity(self.parent),self.id)

    def test_late_failure_does_not_modify_parent(self):
        other='model.layers.1.v_proj'
        programs={self.p:self.program('B[:,:]=1;'),other:self.program('B[:,:]=1/0;',other)}
        with self.assertRaises(ValueError):construct_wspl(self.parent,programs,expected_parent_id=self.id)
        self.assertEqual(identity(self.parent),self.id)

    def test_cross_pair_alias_cannot_be_silently_used(self):
        with self.assertRaises(ValueError):
            construct_wspl(self.parent,{self.p:self.program('B[:,:]=other.B;')},expected_parent_id=self.id)

    def test_base_parameter_not_accepted(self):
        self.parent['model.layers.0.q_proj.weight']=np.ones((4,3),dtype=np.float32)
        with self.assertRaises(ValueError):snapshot(self.parent)

    def test_dtype_nonfinite_and_incomplete_pairs_rejected(self):
        for bad in (np.ones((4,2),dtype=np.float64),np.full((4,2),np.nan,dtype=np.float32)):
            altered=parent();altered[self.p+'.lora_B']=bad
            with self.assertRaises(ValueError):snapshot(altered)
        self.parent.pop(self.p+'.lora_B')
        with self.assertRaises(ValueError):snapshot(self.parent)

    def test_rank_shape_and_shared_storage_rejected(self):
        bad=parent();bad[self.p+'.lora_B']=np.ones((4,3),dtype=np.float32)
        with self.assertRaises(ValueError):snapshot(bad)
        shared=np.ones((2,2),dtype=np.float32)
        with self.assertRaisesRegex(ValueError,'Shared'):
            snapshot({'x.lora_A':shared,'x.lora_B':shared})

    def test_native_rejects_shape_or_dtype_changes(self):
        for replacement in ({'A':np.ones((2,4),dtype=np.float32),'B':np.zeros((4,2),dtype=np.float32)},
                            {'A':self.parent[self.p+'.lora_A'],'B':np.zeros((4,2),dtype=np.float64)}):
            with self.assertRaises(ValueError):construct_native(self.parent,{self.p:replacement},expected_parent_id=self.id)
        self.assertEqual(identity(self.parent),self.id)

    def test_aggregate_budget_rejects_atomically(self):
        limits=replace(AdapterLimits(),max_total_statements=1)
        with self.assertRaisesRegex(ValueError,'Aggregate'):
            construct_wspl(self.parent,{self.p:self.program('B[0,0]=1; B[1,0]=2;')},expected_parent_id=self.id,adapter_limits=limits)
        self.assertEqual(identity(self.parent),self.id)

    def test_native_profile_not_mislabelled_wspl_exact(self):
        source=self.program('B[0,0]=1;').replace('exact_to_storage','native_numpy_v1')
        with self.assertRaises(ValueError):construct_wspl(self.parent,{self.p:source},expected_parent_id=self.id)

    def test_snapshot_and_candidate_do_not_alias_inputs(self):
        copied=snapshot(self.parent);copied[self.p+'.lora_A'][:]=99
        self.assertEqual(identity(self.parent),self.id)
        candidate,_=construct_wspl(self.parent,{},expected_parent_id=self.id)
        candidate[self.p+'.lora_B'][:]=2
        self.assertEqual(identity(self.parent),self.id)


class TorchInstallationTests(unittest.TestCase):
    def test_frozen_base_and_forward_parity_with_native(self):
        import torch
        from skillforge.authoring_sft import LoRALinear, frozen_hash
        torch.manual_seed(7)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.projection=LoRALinear(torch.nn.Linear(3,4),rank=2,alpha=4)
            def forward(self,x):return self.projection(x)
        model=Tiny();before=model_state(model);base_id=frozen_hash(model)
        x=torch.tensor([[1.,-.5,.2],[-1.,2.,0.]])
        with torch.no_grad():expected=model(x).clone()
        unchanged,_=construct_wspl(before,{},expected_parent_id=identity(before))
        install_state(model,unchanged,expected_current_id=identity(before))
        with torch.no_grad():self.assertTrue(torch.equal(expected,model(x)))
        pair=describe(before)['projections']['projection']['pair_id']
        candidate,_=construct_wspl(before,{'projection':source_for_pair(pair,'B[0,0]=1/2; B[1,1]=-1/4;')},expected_parent_id=identity(before))
        replacement={'A':candidate['projection.lora_A'],'B':candidate['projection.lora_B']}
        native,_=construct_native(before,{'projection':replacement},expected_parent_id=identity(before))
        install_state(model,candidate,expected_current_id=identity(before))
        with torch.no_grad():actual=model(x).clone()
        install_state(model,native,expected_current_id=identity(candidate))
        with torch.no_grad():self.assertTrue(torch.equal(actual,model(x)))
        self.assertEqual(frozen_hash(model),base_id);self.assertEqual(model.projection.scale,2)
        with self.assertRaisesRegex(ValueError,'identity'):
            install_state(model,before,expected_current_id=identity(before))
        install_state(model,before,expected_current_id=identity(native))
        with torch.no_grad():self.assertTrue(torch.equal(expected,model(x)))


if __name__=='__main__':unittest.main()
