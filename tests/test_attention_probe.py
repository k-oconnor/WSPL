import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from skillforge.attention_probe import validate, intervention


def bank():
    return dict(K=np.zeros((6,3,64),np.float32),V=np.zeros((6,3,64),np.float32),
                alpha=np.zeros((6,3),np.float32),K_scale=np.ones((6,3),np.float32),V_scale=np.ones((6,3),np.float32))


class ContractTests(unittest.TestCase):
    def test_constraints_and_frozen_stage(self):
        r=bank();b=bank();b['alpha'][1,1]=.5;b['V'][1,1,0]=1
        validate(b,r,'forced');f={k:v.copy() for k,v in b.items()}
        b['K'][1,1,0]=1;validate(b,r,'retrieval',f)
        b['V'][1,1,0]=2
        with self.assertRaises(ValueError):validate(b,r,'retrieval',f)
        b=bank();b['alpha'][0,:2]=1
        with self.assertRaises(ValueError):validate(b,r,'forced')
        b=bank();b['V'][0,0,0]=11
        with self.assertRaises(ValueError):validate(b,r,'forced')
        b=bank();b['K_scale'][0,0]=2
        with self.assertRaises(ValueError):validate(b,r,'forced')
        b=bank();b['K'][0,0,0]=1
        with self.assertRaises(ValueError):validate(b,r,'forced')

    def test_session_rejection_final_and_freeze(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
        from run_attention_probe import session
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r=bank();seen=[];calls=[0]
            def collector(folder,text):
                folder.mkdir();calls[0]+=1
                return ('FINAL' if calls[0]==3 else '```python\npass\n```'),dict(thread_id=str(calls[0]))
            def compiler(source,parent,turn,config):
                b=bank();b['alpha'][0,0]=1;b['V'][0,0,0]=1 if calls[0]==1 else 100
                path=turn/'candidate.npz';np.savez_compressed(path,**b)
                return path,dict(exit_code=0)
            final=session('forced',r,r,{},root/'forced','',{},collector,
                          lambda b,s:seen.append(b['V'][0,0,0]) or {},compiler,lambda *a:None)
            self.assertEqual(seen,[1]);self.assertEqual(final['V'][0,0,0],1)
            self.assertTrue((root/'forced/final.npz').exists())


class CudaAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():raise unittest.SkipTest('CUDA runtime required')

    def test_softmax_gqa_parity_and_locality(self):
        import torch
        import transformers.models.llama.modeling_llama as llama
        torch.manual_seed(4)
        q=torch.randn(2,9,5,64,device='cuda');k=torch.randn(2,3,5,64,device='cuda');v=torch.randn_like(k)
        mask=torch.zeros(2,1,5,5,device='cuda').masked_fill(torch.ones(5,5,device='cuda').triu(1).bool(),float('-inf'))
        # Non-causal padding also must be respected.
        mask[1,:,:,-1]=float('-inf')
        module=SimpleNamespace(num_key_value_groups=3,layer_idx=24,training=False)
        original=llama.eager_attention_forward
        base,_=original(module,q,k,v,mask,1/8)
        b=bank()
        with intervention(b,'forced'):
            noop,_=llama.eager_attention_forward(module,q,k,v,mask,1/8)
        self.assertTrue(torch.equal(noop,base));self.assertIs(llama.eager_attention_forward,original)
        b['alpha'][0,1]=.3;b['V'][0,1]=np.linspace(-1,1,64);b['K'][0,1]=np.linspace(1,-1,64)
        masses=[]
        with intervention(b,'retrieval',masses=masses):
            actual,_=llama.eager_attention_forward(module,q,k,v,mask,1/8)
        kk=torch.from_numpy(b['K'][0,1]).to('cuda').expand(2,1,64)
        vv=torch.from_numpy(b['V'][0,1]).to('cuda').expand(2,1,64)
        keys=torch.cat([k[:,1],kk],dim=1);values=torch.cat([v[:,1],vv],dim=1)
        scores=q[:,3:6,-1]@keys.transpose(-1,-2)/8
        scores[:,:,:5]+=mask[:,:,-1,:]
        expected=torch.softmax(scores,dim=-1)@values
        self.assertLess(float((actual[:,-1,3:6]-expected).abs().max()),1e-6)
        self.assertTrue(torch.equal(actual[:,:-1],base[:,:-1]))
        self.assertTrue(torch.equal(actual[:,-1,:3],base[:,-1,:3]))
        self.assertTrue(torch.equal(actual[:,-1,6:],base[:,-1,6:]))
        with intervention(b,'forced'):
            forced,_=llama.eager_attention_forward(module,q,k,v,mask,1/8)
        torch.testing.assert_close(forced[:,-1,3:6],.7*base[:,-1,3:6]+.3*vv)
        self.assertEqual(len(masses),1)

    def test_exception_restores_hook(self):
        import transformers.models.llama.modeling_llama as llama
        original=llama.eager_attention_forward
        with self.assertRaises(RuntimeError):
            with intervention():raise RuntimeError('fixture')
        self.assertIs(llama.eager_attention_forward,original)


if __name__=='__main__':unittest.main()
