"""Known-answer opaque-network interventions, not scientific causal results."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
try:
    import torch
    import numpy as np
except ImportError:torch=None
from skillforge.interventions import Intervention,audit_intervention,replay_intervention,append_evidence,verify_log


def model():
    m=torch.nn.Sequential(torch.nn.Linear(2,2,bias=False),torch.nn.Tanh(),torch.nn.Linear(2,1,bias=False))
    with torch.no_grad():m[0].weight.copy_(torch.eye(2));m[2].weight.copy_(torch.tensor([[1.,0.]]))
    return m


@unittest.skipIf(torch is None,'optional Torch/NumPy unavailable')
class InterventionTests(unittest.TestCase):
    def audit(self,m,spec,**kwargs):return audit_intervention(m,torch.tensor([[1.,2.]]),spec,runtime_id='unit-fixture-v1',declared_stateless=True,**kwargs)

    def test_non_pwa_network_effect_and_no_effect_are_empirical(self):
        m=model();effect=self.audit(m,Intervention('1',features=(0,)))
        self.assertEqual(effect['status'],'OBSERVED_EFFECT');self.assertTrue(all(effect['controls'].values()))
        self.assertGreater(effect['effect']['max_abs'],.7);self.assertEqual(effect['forward_calls'],4)
        self.assertEqual(effect['semantic_grounding'],'not_established');self.assertEqual(effect['physical_causation'],'not_established')
        no_effect=self.audit(m,Intervention('1',features=(1,)))
        self.assertEqual(no_effect['status'],'NO_OBSERVED_EFFECT')

    def test_scaling_has_known_effect_and_identity_scale_is_a_noop(self):
        m=model();zero=self.audit(m,Intervention('1',features=(0,)))
        half=self.audit(m,Intervention('1','scale',features=(0,),value=.5))
        self.assertEqual(half['status'],'OBSERVED_EFFECT')
        self.assertAlmostEqual(half['effect']['max_abs'],zero['effect']['max_abs']/2,places=7)
        identity=self.audit(m,Intervention('1','scale',features=(0,),value=1.))
        self.assertEqual(identity['status'],'NO_OBSERVED_EFFECT')

    def test_modes_rng_and_hooks_are_restored(self):
        m=model();m.train();m[1].eval();modes=[x.training for x in m.modules()]
        rng=torch.get_rng_state().clone();prng=random.getstate();nrng=np.random.get_state()
        result=self.audit(m,Intervention('1',features=(0,)),seed=42)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(prng,random.getstate())
        self.assertTrue(np.array_equal(nrng[1],np.random.get_state()[1]));self.assertEqual(nrng[2:],np.random.get_state()[2:])
        self.assertEqual(modes,[x.training for x in m.modules()]);self.assertFalse(m[1]._forward_hooks)
        self.assertTrue(result['registered_state_unchanged'])

    def test_stochastic_forward_uses_matched_python_numpy_torch_streams(self):
        class Stochastic(torch.nn.Module):
            def __init__(self):super().__init__();self.site=torch.nn.Identity()
            def forward(self,x):return self.site(x)+torch.rand_like(x)+random.random()+float(np.random.rand())
        r=self.audit(Stochastic(),Intervention('site'))
        self.assertEqual(r['status'],'OBSERVED_EFFECT');self.assertTrue(all(r['controls'].values()))

    def test_logit_changes_and_zero_margin_are_descriptive_only(self):
        m=torch.nn.Sequential(torch.nn.Identity())
        r=self.audit(m,Intervention('0',features=(1,)),output_kind='logits')
        self.assertEqual(r['effect']['argmax_changed_fraction'],1.)
        self.assertEqual(r['effect']['baseline_min_top2_margin'],1.)
        self.assertEqual(r['universal_verification'],'not_established')

    def test_exact_replay_and_changed_input_detection(self):
        m=model();r=self.audit(m,Intervention('1',features=(0,)))
        self.assertTrue(replay_intervention(m,torch.tensor([[1.,2.]]),r)['exact_replay'])
        self.assertFalse(replay_intervention(m,torch.tensor([[2.,2.]]),r)['exact_replay'])

    def test_reused_module_invocation_is_targeted_explicitly(self):
        class Shared(torch.nn.Module):
            def __init__(self):super().__init__();self.shared=torch.nn.Identity()
            def forward(self,x):return self.shared(x)+self.shared(2*x)
        m=Shared();a=self.audit(m,Intervention('shared',call_index=0));b=self.audit(m,Intervention('shared',call_index=1))
        self.assertEqual(a['status'],'OBSERVED_EFFECT');self.assertEqual(b['status'],'OBSERVED_EFFECT')
        self.assertEqual(b['effect']['max_abs'],2*a['effect']['max_abs'])

    def test_registered_state_mutation_invalidates_controls(self):
        class Stateful(torch.nn.Module):
            def __init__(self):super().__init__();self.site=torch.nn.Identity();self.register_buffer('counter',torch.zeros(1))
            def forward(self,x):self.counter.add_(1);return self.site(x)+self.counter
        m=Stateful();r=self.audit(m,Intervention('site'))
        self.assertEqual(r['status'],'INVALID_CONTROL');self.assertFalse(r['registered_state_unchanged'])
        self.assertEqual(float(m.counter),4.)  # no silent claim of hidden-state rollback

    def test_nonpersistent_registered_buffer_is_included_in_identity(self):
        from skillforge.interventions import model_identity
        m=model();m.register_buffer('scratch',torch.zeros(1),persistent=False)
        first=model_identity(m,'unit-fixture-v1')
        m.scratch.add_(1)
        self.assertNotEqual(first['sha256'],model_identity(m,'unit-fixture-v1')['sha256'])

    def test_alias_sensitive_noop_is_not_a_valid_causal_comparison(self):
        class Alias(torch.nn.Module):
            def __init__(self):super().__init__();self.site=torch.nn.Identity()
            def forward(self,x):y=self.site(x);y.add_(1);return y+x
        r=self.audit(Alias(),Intervention('site'))
        self.assertEqual(r['status'],'INVALID_CONTROL');self.assertFalse(r['controls']['noop_matches'])

    def test_invalid_site_occurrence_features_and_replacement_restore_hooks(self):
        for spec in [Intervention('missing'),Intervention('1',call_index=20),Intervention('1',features=(9,)),Intervention('1','replace',value=torch.ones(3))]:
            m=model();r=self.audit(m,spec);self.assertEqual(r['status'],'INVALID');self.assertFalse(m[1]._forward_hooks)
        with self.assertRaises(ValueError):Intervention('1',features=(0,0))

    def test_tensor_patch_payload_requires_matching_replay_data(self):
        m=model();s=Intervention('1','replace',value=torch.zeros(1,2));r=self.audit(m,s)
        self.assertEqual(r['status'],'OBSERVED_EFFECT')
        with self.assertRaises(ValueError):replay_intervention(m,torch.tensor([[1.,2.]]),r)
        self.assertTrue(replay_intervention(m,torch.tensor([[1.,2.]]),r,s)['exact_replay'])

    def test_hash_chain_tamper_anchor_and_proof_tier_rejection(self):
        r=self.audit(model(),Intervention('1',features=(0,)))
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'evidence.jsonl';head=append_evidence(path,r);self.assertEqual(len(verify_log(path,head)),1)
            append_evidence(path,r)
            with self.assertRaises(ValueError):verify_log(path,head)
            lines=path.read_text().splitlines();d=json.loads(lines[0]);d['evidence']['effect']['max_abs']=100;lines[0]=json.dumps(d);path.write_text('\n'.join(lines)+'\n')
            with self.assertRaises(ValueError):verify_log(path)
        invalid=copy.deepcopy(r);invalid['status']='PROVED'
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):append_evidence(Path(root)/'proof.jsonl',invalid)


if __name__=='__main__':unittest.main()
