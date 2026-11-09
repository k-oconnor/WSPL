"""Small additive ReLU computation over native hidden representations."""
from contextlib import contextmanager
import torch
from torch import nn


class ResidualReLU(nn.Module):
    def __init__(self,width=576,rank=32,seed=173):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.read=nn.Linear(width,rank,bias=True)
            self.write=nn.Linear(rank,width,bias=True)
            nn.init.zeros_(self.read.bias);nn.init.zeros_(self.write.weight);nn.init.zeros_(self.write.bias)

    def forward(self,h):return h+self.write(torch.relu(self.read(h)))


@contextmanager
def attached(model,adapter,layer=27):
    """Runs at every token; no task, label, position or answer-dependent dispatch."""
    def hook(module,args,output):
        if isinstance(output,tuple):return (adapter(output[0]),)+output[1:]
        return adapter(output)
    handle=model.model.layers[layer].register_forward_hook(hook)
    try:yield
    finally:handle.remove()
