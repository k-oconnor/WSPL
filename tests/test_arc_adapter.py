import unittest
from types import SimpleNamespace
import numpy as np
import torch
from skillforge.smollm_adapter import LAYERS,initial_bank,effective_weights,edit_wspl,tensor_hash
from skillforge.arc_eval import validate_row,encode_choices,score_encoded


class AdapterTests(unittest.TestCase):
    def base(self):return {f'W{i}':np.full((576,576),.01,dtype=np.float32) for i in LAYERS}

    def test_zero_lora_and_nonzero_merge(self):
        base=self.base();bank=initial_bank(base,'lora');effective,_=effective_weights(base,bank,'lora')
        for k in base:self.assertEqual(tensor_hash(base[k]),tensor_hash(effective[k]))
        bank['A26'][0,2]=.02;bank['B26'][3,0]=.01
        effective,_=effective_weights(base,bank,'lora')
        self.assertAlmostEqual(float(effective['W26'][3,2]),.0102,places=7)
        x=np.arange(576,dtype=np.float32)/576
        np.testing.assert_allclose(effective['W26']@x,base['W26']@x+bank['B26']@(bank['A26']@x),atol=1e-5)

    def test_scope_finite_and_norm_refusals(self):
        base=self.base();bank=initial_bank(base,'lora');bank['P26'][0,0]=0
        with self.assertRaisesRegex(ValueError,'read-only'):effective_weights(base,bank,'lora')
        bank=initial_bank(base,'direct');bank['W26'][0,0]=np.nan
        with self.assertRaises(ValueError):effective_weights(base,bank,'direct')
        bank=initial_bank(base,'direct');bank['W26']*=2
        with self.assertRaisesRegex(ValueError,'norm budget'):effective_weights(base,bank,'direct')

    def test_wspl_transaction_factor_export_and_parent_identity(self):
        base=self.base();bank=initial_bank(base,'lora');before={k:tensor_hash(v) for k,v in bank.items()}
        source='''wspl 0.2;target base=artifact("parent",sha256="{{PARENT_SHA256}}");
edit base as candidate mode=fixed numeric=exact_to_storage { A26[0,2]=1/50; B26[3,0]=1/100; }'''
        updated,effective,receipt=edit_wspl(base,bank,'lora',source)
        self.assertEqual(before,{k:tensor_hash(v) for k,v in bank.items()})
        self.assertAlmostEqual(float(effective['W26'][3,2]),.0102,places=7)
        self.assertNotEqual(receipt['parent_sha256'],receipt['candidate_sha256'])


class ScoringTests(unittest.TestCase):
    def test_label_mapping_and_no_truncation(self):
        row=dict(question='Q',choices=dict(label=['1','2'],text=['one','two']),answerKey='2')
        self.assertEqual(validate_row(row),1)
        tokenizer=SimpleNamespace(encode=lambda s,**kw:list(range(1,len(s)+1)))
        encoded=encode_choices(tokenizer,row)
        self.assertEqual(len(encoded),2)
        with self.assertRaises(ValueError):encode_choices(tokenizer,row,max_length=2)
        row['answerKey']='missing'
        with self.assertRaises(ValueError):validate_row(row)

    def test_answer_only_scoring_padding_and_singleton_equivalence(self):
        class Fake:
            def __call__(self,input_ids,attention_mask,use_cache):
                # Fixed nonuniform distribution with independently calculable logprob.
                logits=torch.arange(8,dtype=torch.float32).expand(*input_ids.shape,8)
                return SimpleNamespace(logits=logits)
        rows=[([1,2,3,4],2),([1,2,5],2)]
        sums,counts=score_encoded(Fake(),rows,0,'cpu');lp=torch.log_softmax(torch.arange(8,dtype=torch.float32),0)
        self.assertEqual(counts,[2,1]);self.assertAlmostEqual(sums[0],float(lp[3]+lp[4]),places=5)
        self.assertAlmostEqual(sums[1],float(lp[5]),places=5)
        for i,row in enumerate(rows):self.assertEqual(score_encoded(Fake(),[row],0,'cpu')[0][0],sums[i])
