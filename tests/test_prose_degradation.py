import math,unittest
from types import SimpleNamespace
import torch
from run_prose_degradation import next_token_loss


class LossTests(unittest.TestCase):
    def test_shift_and_target_count(self):
        ids=torch.tensor([[0,1,2,3]])
        def model(input_ids,use_cache):
            logits=torch.zeros(1,4,5)
            for i in range(3):logits[0,i,i+1]=2
            return SimpleNamespace(logits=logits)
        total,n=next_token_loss(model,ids)
        self.assertEqual(n,3)
        self.assertAlmostEqual(total/n,math.log(math.exp(2)+4)-2,places=6)

    def test_nonfinite_rejected(self):
        def model(**kwargs):return SimpleNamespace(logits=torch.full((1,2,5),float('nan')))
        with self.assertRaises(ValueError):next_token_loss(model,torch.tensor([[0,1]]))
