import unittest,tempfile,random
from pathlib import Path
import torch
from skillforge.grid_copy import make_data,batch_tensors,copy_loss,checkpoint
from skillforge.grid_domain import enumerate_domain


class CopyTests(unittest.TestCase):
    def test_exclusion_and_determinism(self):
        excluded=[r['input'] for r in enumerate_domain()];data=make_data(excluded)
        self.assertEqual(data,make_data(excluded));seen={tuple(x) for x in excluded}
        for split,n in [('train',4096),('heldout',256)]:
            self.assertEqual(len(data[split]),n)
            self.assertEqual(sum(r['kind']=='sparse' for r in data[split]),n//2)
            for row in data[split]:
                self.assertEqual(set(row),{'input','kind'});self.assertNotIn(tuple(row['input']),seen);seen.add(tuple(row['input']))

    def test_answer_mask_and_shift(self):
        from types import SimpleNamespace
        batch=batch_tensors([([1,2,3,4],2),([1,2,3],1)],0,'cpu')
        self.assertEqual(batch[2].tolist(),[[-100,-100,3,4],[-100,2,3,-100]])
        logits=torch.randn(2,4,5,requires_grad=True)
        loss=copy_loss(lambda **kw:SimpleNamespace(logits=logits),batch);loss.backward()
        self.assertEqual(float(logits.grad[0,0].abs().sum()),0)
        self.assertGreater(float(logits.grad[0,1].abs().sum()),0)
        self.assertEqual(float(logits.grad[1,2:].abs().sum()),0)

    def test_checkpoint_restores_next_update(self):
        torch.manual_seed(7);model=torch.nn.Linear(3,2);opt=torch.optim.AdamW(model.parameters(),lr=.001)
        x=torch.ones(2,3)
        def update(m,o):o.zero_grad();m(x).square().sum().backward();o.step()
        update(model,opt)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'last.pt';checkpoint(p,model,opt,1,'source',[1,0],random.Random(3))
            state=torch.load(p,weights_only=False);other=torch.nn.Linear(3,2);other.load_state_dict(state['model'])
            otheropt=torch.optim.AdamW(other.parameters());otheropt.load_state_dict(state['optimizer'])
            update(model,opt);update(other,otheropt)
            for a,b in zip(model.parameters(),other.parameters()):self.assertTrue(torch.equal(a,b))
            self.assertEqual(state['manifest_sha256'],'source');self.assertFalse(p.with_suffix('.tmp').exists())

if __name__=='__main__':unittest.main()
