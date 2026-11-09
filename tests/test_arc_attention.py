import unittest
from types import SimpleNamespace
import numpy as np


class BoundaryTests(unittest.TestCase):
    def test_explicit_position_causal_and_local(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA required')
        import transformers.models.llama.modeling_llama as llama
        from skillforge.attention_probe import intervention
        torch.manual_seed(8)
        q=torch.randn(1,9,6,64,device='cuda'); k=torch.randn(1,3,6,64,device='cuda');v=torch.randn_like(k)
        mask=torch.zeros(1,1,6,6,device='cuda').masked_fill(torch.ones(6,6,device='cuda').triu(1).bool(),float('-inf'))
        module=SimpleNamespace(num_key_value_groups=3,layer_idx=26,training=False)
        bank=dict(K=np.zeros((6,3,64),np.float32),V=np.ones((6,3,64),np.float32),alpha=np.zeros((6,3),np.float32))
        bank['alpha'][2,0]=.75
        base,_=llama.eager_attention_forward(module,q,k,v,mask,1/8)
        masses=[]
        with intervention(bank,'retrieval',masses=masses,query_index=2):
            actual,_=llama.eager_attention_forward(module,q,k,v,mask,1/8)
            k2=k.clone();k2[:,:,3:]+=100
            v2=v.clone();v2[:,:,3:]-=100
            changed,_=llama.eager_attention_forward(module,q,k2,v2,mask,1/8)
        torch.testing.assert_close(actual[:,2],changed[:,2],rtol=0,atol=0)
        self.assertTrue(torch.equal(actual[:,:2],base[:,:2]))
        self.assertTrue(torch.equal(actual[:,3:],base[:,3:]))
        self.assertFalse(torch.equal(actual[:,2,:3],base[:,2,:3]))
        self.assertEqual(masses[0],masses[1])
        with self.assertRaises(ValueError),intervention(bank,'retrieval',query_index=6):
            llama.eager_attention_forward(module,q,k,v,mask,1/8)


if __name__=='__main__':unittest.main()
