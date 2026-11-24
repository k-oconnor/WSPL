import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.authoring_sft import LoRALinear,adapter_state,frozen_hash
from skillforge.robot_access import inspect_acquisition,validate_request
from skillforge.robot_authoring_access import run_session,prompt


class Tokenizer:
    def __call__(self,c,**kw):return {'input_ids':[['A','B','0','1','2','3','4','P','S'].index(c)]}
    def decode(self,ids,**kw):return str(ids[0])


class Toy(nn.Module):
    def __init__(self):
        super().__init__();torch.manual_seed(43)
        self.embed=nn.Embedding(12,4);self.model=nn.Module();self.model.layers=nn.ModuleList()
        for _ in range(2):
            block=nn.Module();block.self_attn=nn.Module();block.self_attn.q_proj=LoRALinear(nn.Linear(4,4),rank=2,alpha=4)
            self.model.layers.append(block)
        self.model.norm=nn.LayerNorm(4);self.lm_head=nn.Linear(4,12)
        self.config=SimpleNamespace(hidden_size=4);self.fail=False
        for n,p in self.named_parameters():p.requires_grad_('.lora_' in n)
    def gradient_checkpointing_disable(self):pass
    def forward(self,input_ids,**kw):
        x=self.embed(input_ids)
        for b in self.model.layers:x=x+torch.tanh(b.self_attn.q_proj(x))
        if self.fail:raise RuntimeError('Injected failure')
        return SimpleNamespace(logits=self.lm_head(self.model.norm(x))[:,-1:])


def request(**kw):
    value=dict(tool='inspect',candidate=None,rows='all',projections=[],tokens='last',weights=False);value.update(kw);return value


class AccessTests(unittest.TestCase):
    def setUp(self):
        self.model=Toy();self.tokenizer=Tokenizer()
        self.bank={k:v.numpy() for k,v in adapter_state(self.model).items()}
        self.names=[f'model.layers.{i}.self_attn.q_proj' for i in range(2)]
        self.rows=[dict(record={'ids':[1,2,3]},code='A'),dict(record={'ids':[2,3]},code='B')]
    def run_query(self,req,out,bank=None,**kw):
        return inspect_acquisition(self.model,self.tokenizer,self.rows,bank or self.bank,req,out,lambda:None,**kw)
    def test_multiple_sites_full_sequence_weights_and_restore(self):
        initial=adapter_state(self.model);base=frozen_hash(self.model)
        edited={k:v.copy() for k,v in self.bank.items()}
        for n in self.names:edited[n+'.lora_B'][:]=0.1
        with tempfile.TemporaryDirectory() as d:
            a=self.run_query(request(),Path(d)/'parent')
            b=self.run_query(request(projections=self.names,tokens='all',weights=True),Path(d)/'candidate',edited)
            arrays=np.load(Path(d)/'candidate/arrays.npz')
            self.assertEqual(arrays['row0_p0_input'].shape,(3,4));self.assertEqual(arrays['row1_p1_output'].shape,(2,4))
            self.assertTrue(np.array_equal(arrays['p0_base_weight'],self.model.get_submodule(self.names[0]).base.weight.detach().numpy()))
            arrays.close()
            self.assertNotEqual(a['mean_target_nll'],b['mean_target_nll'])
        self.assertEqual(frozen_hash(self.model),base)
        self.assertTrue(all(torch.equal(initial[k],v) for k,v in adapter_state(self.model).items()))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))
    def test_failure_restores_factors_and_hooks(self):
        self.model.fail=True;initial=adapter_state(self.model)
        edited={k:v+1 for k,v in self.bank.items()}
        with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(RuntimeError,'Injected'):
            self.run_query(request(projections=self.names),Path(d)/'bad',edited)
        self.assertTrue(all(torch.equal(initial[k],v) for k,v in adapter_state(self.model).items()))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))
    def test_rejects_outside_data_and_unknown_interfaces(self):
        for bad in (request(rows=[-1]),request(rows=[2]),request(rows=[0,0]),request(rows=[True]),request(projections=['private']),request(prompt='secret'),request(weights=1),request(tokens='future'),request(candidate={})):
            with self.subTest(bad=bad),self.assertRaises(ValueError):validate_request(bad,2,self.names)
    def test_capacity_guard_before_forward(self):
        self.model.fail=True
        with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(ValueError,'output elements'):
            self.run_query(request(projections=self.names),Path(d)/'large',max_elements=1)
    def test_score_matches_actual_full_forward(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.run_query(request(rows=[1]),Path(d)/'score')
        with torch.no_grad():logits=self.model(input_ids=torch.tensor([[2,3]])).logits[0,-1]
        self.assertAlmostEqual(result['mean_target_nll'],float(torch.logsumexp(logits,0)-logits[1]),places=5)
        self.assertEqual(result['rows'][0]['predicted_token'],int(logits.argmax()))
    def test_author_inspection_does_not_freeze_or_use_development(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);common=d/'common';common.mkdir();(common/'catalog.json').write_text('{}')
            class Workspace:
                def __init__(self,out,runtime):
                    self.inputs=out/'inputs';self.work=out/'work';self.inputs.mkdir(parents=True);self.work.mkdir()
            calls=iter([request(rows=[0]),request(rows=[0]),dict(tool='finish',method='No deployment')])
            def collect(*args,**kw):return json.dumps(next(calls)),{}
            def query(req,candidate,folder):
                self.assertIsNone(candidate);folder.mkdir();np.savez(folder/'arrays.npz',x=np.ones(1));(folder/'diagnostics.json').write_text('{}')
                return dict(examples=1,mean_target_nll=1,action_correct=0,projections={})
            cfg=dict(author_runtime='unused',native_runtime='unused',author_responses=3,inspection_attempts=1,
                cli='unused',catalog='unused',instructions='unused',author_call_seconds=1,proposal_attempts=4)
            def forbidden(*a):raise AssertionError('Development evaluation must not run')
            value=run_session('native',common,dict(adapter_id='fixture',acquisition_rows=2,projections=self.names),cfg,d/'author','m','s',lambda:None,lambda:False,lambda **k:None,forbidden,lambda s:None,collect_fn=collect,workspace_factory=Workspace,query_candidate=query)
            self.assertIsNone(value)
            tr=json.loads((d/'author/transcript.json').read_text());self.assertIn('cap exhausted',tr[3]['content']['error'])
            frozen=json.loads((d/'author/frozen.json').read_text());self.assertEqual(frozen['proposal_attempts'],0);self.assertEqual(frozen['inspection_attempts'],1)
    def test_prompt_has_no_privileged_final_projection_example(self):
        for arm in ('native','wspl'):
            value=prompt(arm,'inputs',{},[])
            self.assertNotIn('model.layers.27',value);self.assertNotIn('features.npz',value)
            self.assertIn('any subset',value);self.assertIn('24 total',value)


if __name__=='__main__':unittest.main()
