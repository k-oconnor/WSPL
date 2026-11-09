import unittest
import torch
from skillforge.grid_movement import prepare_data,encode_task
from skillforge.grid_domain import prepare_sets
from skillforge.grid_copy import make_data
from skillforge.relu_adapter import ResidualReLU,attached


class MovementTests(unittest.TestCase):
    def test_splits(self):
        original=prepare_sets();data=prepare_data(original,make_data([]))
        train={tuple(r['input']) for r in data['train']};self.assertEqual(len(train),168)
        for split in ('familiar','novel','copy_domain','demonstrations'):
            self.assertFalse(train.intersection(tuple(r['input']) for r in data[split]))
        self.assertTrue(all(not r['shape'].startswith('L') for r in data['train']))
        self.assertTrue(all(r['shape'].startswith('L') for r in data['novel']))
        for d in ('up','down','left','right'):
            for c in (1,2):self.assertEqual(sum(r['direction']==d and r['color']==c for r in data['train']),21)

    def test_target_mask_boundary(self):
        class Tokenizer:
            def encode(self,text,**kw):return [int(text)+10] if text in ('0','1','2','3') else [8]*20
            def decode(self,ids):return str(ids[0]-10)
        row=prepare_sets()['movement_familiar'][0]
        tokens,start=encode_task(Tokenizer(),row,'movement',[])
        self.assertEqual(start,20);self.assertEqual(tokens[start:],[x+10 for x in row['output']])
        tokens,start=encode_task(Tokenizer(),row,'copy',[])
        self.assertEqual(tokens[start:],[x+10 for x in row['input']])

    def test_frozen_base_and_hook_cleanup(self):
        from types import SimpleNamespace
        layer=torch.nn.Linear(5,5).requires_grad_(False)
        model=SimpleNamespace(model=SimpleNamespace(layers=[layer]))
        adapter=ResidualReLU(width=5,rank=3);x=torch.randn(2,5);base=layer(x).detach().clone()
        with self.assertRaises(RuntimeError):
            with attached(model,adapter,layer=0):
                self.assertTrue(torch.equal(layer(x),base));layer(x).square().sum().backward()
                self.assertIsNotNone(adapter.write.weight.grad);self.assertIsNone(layer.weight.grad)
                raise RuntimeError('exit')
        self.assertEqual(len(layer._forward_hooks),0);self.assertTrue(torch.equal(layer(x),base))

if __name__=='__main__':unittest.main()
