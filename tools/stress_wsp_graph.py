"""Seeded engineering stress checks and archived-controller integration; no model calls."""
import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from skillforge.wsp_artifact import TensorArtifact,diff
from skillforge.wsp_editor import plan,apply,import_legacy
from skillforge.wsp_graph import (node,to_dag,GraphBudget,region,declared_region,
    replace_region,add_residual,transplant,redirect,untie)


def oracle(artifact,inputs,*,fp32=False):
    """Independent recursive traversal; no production evaluator/toposort/bit decoder."""
    d=artifact.to_dict();g=d['graph'];nodes={n['id']:n for n in g['nodes']}
    weights={k:np.array([struct.unpack('!f',bytes.fromhex(s))[0] for s in r['bits']]).reshape(r['shape']) for k,r in d['tensors'].items()}
    cache={f'input:{k}':torch.tensor(v,dtype=torch.float32) if fp32 else list(map(Q,v)) for k,v in inputs.items()}
    def get(signal):
        if signal in cache:return cache[signal]
        n=nodes[signal.split(':',1)[1]];v={k:get(s) for k,s in n['inputs'].items()};op=n['op']
        if op=='affine':
            w=weights[n['params']['weight']];b=weights[n['params']['bias']]
            if n['attrs']['layout']=='in_out':w=w.T
            z=torch.nn.functional.linear(v['x'],torch.tensor(w,dtype=torch.float32),torch.tensor(b,dtype=torch.float32)) if fp32 else [sum([Q(float(c))*x for c,x in zip(row,v['x'])],Q(float(bias))) for row,bias in zip(w,b)]
        elif op=='relu':z=torch.relu(v['x']) if fp32 else [max(Q(0),x) for x in v['x']]
        elif op=='tanh':
            if not fp32:raise ValueError('no rational tanh oracle')
            z=torch.tanh(v['x'])
        elif op=='identity':z=v['x']
        elif op=='add':z=v['a']+v['b'] if fp32 else [a+b for a,b in zip(v['a'],v['b'])]
        elif op=='scale':
            c=float(weights[n['params']['gain']][0]);z=v['x']*c if fp32 else [x*Q(c) for x in v['x']]
        elif op=='concat':z=torch.cat([v['a'],v['b']]) if fp32 else v['a']+v['b']
        elif op=='slice':
            indices=list(range(len(v['x'])))[slice(n['attrs']['start'],n['attrs']['stop'],n['attrs']['step'])]
            z=v['x'][indices] if fp32 else [v['x'][i] for i in indices]
        else:raise AssertionError(op)
        cache[signal]=z;return z
    return {k:get(s) for k,s in g['outputs'].items()}


def generated(seed):
    r=random.Random(seed);width=r.randint(2,5)
    arrays={'W':np.array([[r.randint(-2,2)/4 for _ in range(width)] for _ in range(width)],np.float32),'b':np.zeros(width,np.float32)}
    nodes=[node('a','affine',{'x':'input:x'},params={'weight':'W','bias':'b'},attrs={'layout':'in_out'}),node('b','affine',{'x':'input:x'},params={'weight':'W','bias':'b'},attrs={'layout':'in_out'})]
    signals=['input:x','node:a','node:b']
    for i in range(12):
        name=f'n{i}';op=r.choice(['affine','relu','add','identity','concat_slice','scale'])
        x=r.choice(signals)
        if op=='affine':n=node(name,op,{'x':x},params={'weight':'W','bias':'b'},attrs={'layout':r.choice(['in_out','out_in'])})
        elif op=='add':n=node(name,op,{'a':x,'b':r.choice(signals)})
        elif op=='concat_slice':
            nodes.append(node(name+'cat','concat',{'a':x,'b':r.choice(signals)}))
            n=node(name,'slice',{'x':'node:'+name+'cat'},attrs={'start':2*width-1,'stop':-2*width-1,'step':-2})
        elif op=='scale':
            arrays[name+'gain']=np.array([r.choice([-.5,.5,1.])],np.float32)
            n=node(name,op,{'x':x},params={'gain':name+'gain'})
        else:n=node(name,op,{'x':x})
        nodes.append(n);signals.append('node:'+name)
    r.shuffle(nodes)
    return TensorArtifact.from_arrays(arrays,graph=dict(kind='dag',inputs={'x':width},nodes=nodes,outputs={'y':'node:n11','keep':'node:a'}))


def verify(a,rows):
    error=0.
    for row in rows:
        actual=a.trace(row)
        for k,v in oracle(a,row).items():assert actual[k]==v,(k,actual[k],v)
        independent=oracle(a,row,fp32=True);actual=a.trace(row,semantics='numpy_f32')
        for k,v in independent.items():
            x=np.array(actual[k]);y=v.numpy();error=max(error,float(np.max(np.abs(x-y))))
            np.testing.assert_allclose(x,y,rtol=2e-5,atol=2e-6)
    return error


def randomized(count):
    records=[];budget=GraphBudget(max_nodes=256,max_parameters=10000)
    for seed in range(count):
        a=generated(41000+seed);r=random.Random(seed);w=a.to_dict()['graph']['inputs']['x']
        rows=[{'x':[Q(r.randint(-8,8),4) for _ in range(w)]} for _ in range(4)]
        maximum=verify(a,rows);changes=0;rejections=0;identities=[a.sha256]
        for step in range(12):
            before=a;old=[oracle(a,row) for row in rows];kind=step%6
            if kind==0:
                a,_=untie(a,'b.weight',f'private{step}',budget=budget)
                assert all(oracle(a,row)==v for row,v in zip(rows,old))
            elif kind==1:
                body=f'private{step-1}[0,0]+=1/8;'
                p=plan(f'wspl 0.2; target base=artifact("base",sha256="{a.sha256}"); edit base as candidate mode=fixed {{ {body} }}',parent=a)
                a=apply(p,parent=a,expected_parent_id=a.sha256)
            elif kind==2:
                cut=region(a,['b']);d=a.to_dict();n=next(n for n in d['graph']['nodes'] if n['id']=='b')
                patch=TensorArtifact.from_arrays({k:a.arrays()[k] for k in set(n['params'].values())},graph=dict(kind='dag',inputs={'x':w},nodes=[node('copy','affine',{'x':'input:x'},params=n['params'],attrs=n['attrs'])],outputs={'y':'node:copy'}))
                ins={f'i{i}':k for i,k in enumerate(cut['inputs'])};outs={f'o{i}':k for i,k in enumerate(cut['outputs'])}
                decl=declared_region(a,['b'],ins,outs,list(cut['shared_parameters']))
                a,_=replace_region(a,decl,patch,prefix=f'repl{step}',inputs={'x':next(iter(ins))},outputs={k:'y' for k in outs},budget=budget)
                assert all(oracle(a,row)==v for row,v in zip(rows,old))
            elif kind==3:
                patch=TensorArtifact.from_arrays({},graph=dict(kind='dag',inputs={'x':w},nodes=[],outputs={'y':'input:x'}))
                a,_=add_residual(a,patch,prefix=f'resid{step}',inputs={'x':'input:x'},output='y',into='output.y',scale=Q(1,8),budget=budget)
                for row,v in zip(rows,old):assert oracle(a,row)=={'y':[x+z/8 for x,z in zip(v['y'],row['x'])],'keep':v['keep']}
            elif kind==4:
                patch=TensorArtifact.from_arrays({},graph=dict(kind='dag',inputs={'x':w},nodes=[node('b','identity',{'x':'input:x'})],outputs={'y':'node:b'}))
                a,_=transplant(a,patch,prefix=f'new{step}',inputs={'x':'input:x'},budget=budget)
                assert all(oracle(a,row)==v for row,v in zip(rows,old))
                # Restore a stable editable reference for the next cycle without touching outputs.
                d=a.to_dict();d['graph']['nodes'].append(node('b','affine',{'x':'input:x'},params={'weight':'W','bias':'b'},attrs={'layout':'in_out'}));a=TensorArtifact(d)
            else:
                # A late invalid write after a valid write must not publish a candidate.
                source=f'wspl 0.2; target base=artifact("base",sha256="{a.sha256}"); edit base as candidate mode=fixed {{ b[0]=1; b[9999]=2; }}'
                try:plan(source,parent=a)
                except ValueError:rejections+=1
                else:raise AssertionError('invalid transaction accepted')
                try:redirect(a,'a.x','node:a',budget=budget)
                except ValueError:rejections+=1
                else:raise AssertionError('cycle accepted')
            assert before.sha256==identities[-1]
            maximum=max(maximum,verify(a,rows));changes+=int(before.sha256!=a.sha256);identities.append(a.sha256)
            delta=diff(before,a)['graph']
            for row in rows:
                bt=before.trace(row);at=a.trace(row)
                for key in delta['unaffected_dependency_nodes']:assert bt['node:'+key]==at['node:'+key]
        records.append(dict(seed=41000+seed,steps=12,changed_states=changes,rejected_operations=rejections,max_torch_absolute_error=maximum,artifact_ids=identities))
    return records


def controllers(output):
    records=[];rng=np.random.default_rng(9241)
    rows=np.concatenate([np.zeros((1,600),np.float32),np.ones((1,600),np.float32),rng.integers(0,2,size=(14,600)).astype(np.float32),rng.uniform(-.01,1.01,size=(16,600)).astype(np.float32)])
    for path in sorted((ROOT/'artifacts/astra_breakout_native_v2').glob('draw_*.answer.md')):
        start=time.perf_counter();original=hashlib.sha256(path.read_bytes()).hexdigest()
        dense,receipt=import_legacy(path.read_text(encoding='utf-8'));assert receipt['skipped']==0
        a=to_dag(dense);assert a.to_dict()['tensors']==dense.to_dict()['tensors']
        patch=TensorArtifact.from_arrays({},graph=dict(kind='dag',inputs={'x':6},nodes=[node('copy','identity',{'x':'input:x'})],outputs={'y':'node:copy'}))
        cut=region(a,['y']);decl=declared_region(a,['y'],{'in':'y.x'},{'out':'output.y'},[])
        src=f'''wspl 0.2; target base=artifact("base",sha256="{a.sha256}");
          region tail in base {{ nodes=[node("y")]; inputs={{v:edge("y.x")}}; outputs={{v:edge("output.y")}}; }}
          edit base as candidate mode=graph budget="audit" {{ replace tail with artifact("patch",sha256="{patch.sha256}") as new {{ inputs={{x:tail.inputs.v}}; outputs={{tail.outputs.v:y}}; }} }}'''
        p=plan(src,parent=a,artifacts={'patch':patch},budgets={'audit':GraphBudget(max_parameters=21000)})
        b=apply(p,parent=a,expected_parent_id=a.sha256)
        folder=output/path.stem;folder.mkdir();b.save(folder/'candidate.json');p.save(folder/'plan.json');b=TensorArtifact.load(folder/'candidate.json')
        assert b.to_dict()['tensors']==a.to_dict()['tensors']
        maximum=0.
        for row in rows:
            legacy=np.array(dense.trace(row,semantics='numpy_f32')['y'],np.float32)
            actual=np.array(b.trace({'x':row},semantics='numpy_f32')['y'],np.float32)
            np.testing.assert_array_equal(legacy.view(np.uint32),actual.view(np.uint32))
            other=oracle(b,{'x':row},fp32=True)['y'].numpy()
            np.testing.assert_allclose(actual,other,rtol=3e-5,atol=3e-5);maximum=max(maximum,float(np.max(np.abs(actual-other))))
        assert original==hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(dict(source=path.relative_to(ROOT).as_posix(),source_sha256=original,parent_id=a.sha256,candidate_id=b.sha256,parameters=sum(v.size for v in b.arrays().values()),samples=len(rows),tensor_bits_preserved=True,numpy_outputs_bitwise_identical=True,max_torch_absolute_error=maximum,seconds=time.perf_counter()-start))
    assert len(records)==3
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,default=100);args=p.parse_args()
    if not 1<=args.seeds<=1000:raise ValueError('seed bound')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);start=time.perf_counter()
    result=dict(kind='seeded engineering stress test; no scientific run',status='running')
    try:
        result['randomized']=randomized(args.seeds);result['controllers']=controllers(args.output);result['status']='passed'
    except BaseException as exc:result.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        result.update(seconds=time.perf_counter()-start,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_calls=0,training_updates=0,scored_episodes=0)
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('randomized','controllers')}))


if __name__=='__main__':main()
