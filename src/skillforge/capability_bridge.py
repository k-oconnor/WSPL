"""Shared finite DAG-to-Torch bridge and named attention projection attachments."""
from contextlib import contextmanager
import numpy as np
from .wsp_artifact import TensorArtifact
from .wsp_graph import node,validate_graph


def projection_widths(name):
    parts=name.split('_')
    if len(parts)!=2 or not parts[0].isdigit() or not 0<=int(parts[0])<30 or parts[1] not in ('q','k','v','o'):raise ValueError('projection name')
    return 576,192 if parts[1] in ('k','v') else 576


def lowrank(name,rank=16,seed=701):
    din,dout=projection_widths(name)
    if type(rank)!=int or not 1<=rank<=64:raise ValueError('rank1..64')
    rng=np.random.default_rng(seed);arrays=dict(A=rng.normal(0,1/np.sqrt(din),(rank,din)).astype(np.float32),a=np.zeros(rank,np.float32),B=np.zeros((dout,rank),np.float32),b=np.zeros(dout,np.float32))
    graph=dict(kind='dag',inputs={'x':din},nodes=[node('down','affine',{'x':'input:x'},params={'weight':'A','bias':'a'},attrs={'layout':'out_in'}),
        node('up','affine',{'x':'node:down'},params={'weight':'B','bias':'b'},attrs={'layout':'out_in'})],outputs={'y':'node:up'})
    return TensorArtifact.from_arrays(arrays,graph=graph)


def default_plan():
    return dict(recipe=dict(lr=.001,copy_fraction=.25,changed_cell_weight=1.,steps=2048),
        branches={f'{i}_{p}':lowrank(f'{i}_{p}',seed=701+i*4+j).to_dict() for i in range(30) for j,p in enumerate(('q','k','v','o'))})


def validate_plan(plan):
    if set(plan)!={'recipe','branches'}:raise ValueError('plan keys')
    r=plan['recipe']
    if set(r)!={'lr','copy_fraction','changed_cell_weight','steps'}:raise ValueError('recipe keys')
    if not 1e-5<=r['lr']<=.005 or not .125<=r['copy_fraction']<=.75 or not 1<=r['changed_cell_weight']<=8:raise ValueError('recipe range')
    if type(r['steps'])!=int or not 128<=r['steps']<=2048:raise ValueError('steps128..2048')
    if not isinstance(plan['branches'],dict) or not 1<=len(plan['branches'])<=120:raise ValueError('branch count')
    total=0;metadata={}
    for name,document in plan['branches'].items():
        din,dout=projection_widths(name);artifact=TensorArtifact(document,max_elements=250000);d=artifact.to_dict();g=d['graph']
        meta=validate_graph(g,d['tensors'])
        if g['inputs']!={'x':din} or meta['output_widths']!={'y':dout} or meta['depth']>8 or meta['nodes']>16 or meta['live_values']>8192 or meta['unreferenced_tensors']:raise ValueError('branch interface/budget')
        total+=meta['parameters'];metadata[name]=meta
    if not 1<=total<=2500000:raise ValueError('1..2.5M trainable parameters required')
    return dict(parameters=total,branches=metadata)


def torch_branch(artifact):
    import torch
    from torch import nn
    class Branch(nn.Module):
        def __init__(self):
            super().__init__();d=artifact.to_dict();self.graph=d['graph'];self.meta=validate_graph(self.graph,d['tensors'])
            self.storage=nn.ParameterDict({k:nn.Parameter(torch.from_numpy(v.copy())) for k,v in artifact.arrays().items()})
            self.nodes={n['id']:n for n in self.graph['nodes']}
        def forward(self,x):
            values={'input:x':x}
            for key in self.meta['order']:
                n=self.nodes[key];v={p:values[s] for p,s in n['inputs'].items()};p={k:self.storage[t] for k,t in n['params'].items()};op=n['op']
                if op=='affine':
                    w=p['weight'] if n['attrs']['layout']=='out_in' else p['weight'].T
                    y=torch.nn.functional.linear(v['x'],w,p['bias'])
                elif op=='relu':y=torch.relu(v['x'])
                elif op=='tanh':y=torch.tanh(v['x'])
                elif op=='identity':y=v['x']
                elif op=='add':y=v['a']+v['b']
                elif op=='scale':y=v['x']*p['gain'][0]
                elif op=='concat':y=torch.cat((v['a'],v['b']),dim=-1)
                elif op=='slice':
                    a=n['attrs'];indices=list(range(v['x'].shape[-1]))[slice(a['start'],a['stop'],a['step'])];y=v['x'][...,indices]
                else:raise ValueError('operator')
                values['node:'+key]=y
            return values[self.graph['outputs']['y']]
        def artifact(self):
            return TensorArtifact.from_arrays({k:v.detach().cpu().numpy().copy() for k,v in self.storage.items()},graph=self.graph)
    return Branch()


def modules(plan,device='cuda'):
    from torch import nn
    validate_plan(plan)
    return nn.ModuleDict({k:torch_branch(TensorArtifact(v,max_elements=250000)) for k,v in plan['branches'].items()}).to(device)


def export(plan,bank):return dict(recipe=plan['recipe'].copy(),branches={k:m.artifact().to_dict() for k,m in bank.items()})


@contextmanager
def attached(model,bank):
    handles=[]
    try:
        for name,branch in bank.items():
            layer,projection=name.split('_');module=getattr(model.model.layers[int(layer)].self_attn,projection+'_proj')
            def hook(module,args,output,branch=branch):return output+branch(args[0])
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for h in handles:h.remove()
