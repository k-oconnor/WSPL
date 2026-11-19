"""Supplied-rule one-block adaptation and a fixed AdamW control."""
from fractions import Fraction
import hashlib
import json
import random
import time

from .controller_author import oracle,teacher,prompt as block_prompt
from .wspl_artifact import ReLUArtifact


def calibration(spec,seed=317,count=128):
    rng=random.Random(seed);rows=[]
    for i in range(count):
        bound=2. if i%2==0 else .15
        values=dict(x=round(rng.uniform(-bound,bound),6),y=round(rng.uniform(-bound,bound),6))
        rows.append(dict(inputs=values,outputs=oracle(spec,values)))
    return rows


def author_prompt(spec,rows):
    return (block_prompt(spec)+'\nThe actuator mapping has reversed both motion axes: physical displacement is '
            '(-vx,-vy). The supplied negative gain already compensates for this mapping. '
            'Only this movement block is replaced; all other neural blocks and the executive are frozen. '
            'Use a module that compiles to at most 27 scalar weights/biases, matching the existing 2-4-3 ReLU network. '
            'Implement the supplied signed gain, including its negative sign. '
            'The same calibration examples are available to the AdamW comparison. Each row is [x,y,vx,vy]; grip is zero.\n'+
            json.dumps([[r['inputs']['x'],r['inputs']['y'],r['outputs']['vx'],r['outputs']['vy']] for r in rows],separators=(',',':')))


def fit_replacement(template,rows,seed,updates=4096,guard=lambda:None):
    import torch
    from torch import nn
    started=time.monotonic()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        d=template.to_dict();widths=d['allocation']['widths']
        parts=[];linears=[]
        for i,(a,b) in enumerate(zip(widths,widths[1:])):
            layer=nn.Linear(a,b);parts.append(layer);linears.append(layer)
            with torch.no_grad():
                layer.weight.copy_(torch.tensor([[float(Fraction(v)) for v in row] for row in d['layers'][i]['weights']]))
                layer.bias.copy_(torch.tensor([float(Fraction(v)) for v in d['layers'][i]['bias']]))
            if i<len(widths)-2:parts.append(nn.ReLU())
        net=nn.Sequential(*parts)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=0,foreach=False)
        x=torch.tensor([[r['inputs'][n] for n in template.input_names] for r in rows],dtype=torch.float32)
        y=torch.tensor([[r['outputs'][n] for n in template.output_names] for r in rows],dtype=torch.float32)
        for step in range(updates):
            guard();optimizer.zero_grad(set_to_none=True)
            indices=torch.randint(len(rows),(min(32,len(rows)),))
            loss=(net(x[indices])-y[indices]).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite replacement fitting')
            loss.backward();optimizer.step()
        with torch.no_grad():final_mse=float((net(x)-y).square().mean())
        for original,layer in zip(d['layers'],linears):
            original['weights']=[[str(Fraction(float(v))) for v in row] for row in layer.weight.detach()]
            original['bias']=[str(Fraction(float(v))) for v in layer.bias.detach()]
        provenance=dict(rows=rows,seed=seed,updates=updates,lr=.003,decay=0,batch_size=32,
                        initialized_from=template.sha256)
        d['source_sha256']=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
        d['compiler']='closed_loop.adamw_export.v1';d['allocation']['mode']='imported';d['source_map']={}
        artifact=ReLUArtifact(d)
    return artifact,dict(seed=seed,updates=updates,train_mse=final_mse,seconds=time.monotonic()-started,
                         parameters=d['allocation']['parameters'],artifact_sha256=artifact.sha256,
                         initialized_from=template.sha256,batch_size=min(32,len(rows)))
