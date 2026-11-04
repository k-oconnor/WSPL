"""Scalar WSPL JSON IR -> fresh dense ReLU MLP with signed relays.

Uses exact linear forms and algebraic ReLU lowering. This is not a universal
equivalence verifier or fixed-architecture patch allocator.
"""
import argparse
import copy
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import sys
from .wspl_core import ScalarProgram,rational
from .wspl_artifact import ReLUArtifact,canonical_bytes


class CompilationLimit(ValueError):pass


class Linear:
    def __init__(self,terms=(),bias=0):
        acc={}
        for key,value in terms:acc[key]=acc.get(key,Q(0))+value
        self.terms=tuple(sorted((k,Q(v)) for k,v in acc.items() if v));self.bias=Q(bias)

    def scale(self,c):return Linear(((k,v*c) for k,v in self.terms),self.bias*c)
    def plus(self,other):return Linear(self.terms+other.terms,self.bias+other.bias)
    def shift(self,c):return Linear(self.terms,self.bias+c)
    def record(self):return dict(terms={k:str(v) for k,v in self.terms},bias=str(self.bias))


def compile_scalar(document,*,target='fresh_mlp',max_parameters=100_000,max_layers=64,max_units=4096,max_nodes=1024):
    if target!='fresh_mlp':raise ValueError('only fresh MLP allocation is implemented; fixed/augmented modes unsupported')
    if any(type(v) is not int or v<1 for v in (max_parameters,max_layers,max_units)):raise ValueError('positive compilation limits required')
    d=copy.deepcopy(document);ScalarProgram(d,max_nodes=max_nodes)  # type/domain/scope validation, no evaluation
    env={n:Linear([(f'i{i}',Q(1))]) for i,n in enumerate(d['inputs'])}
    inputs={f'i{i}':i for i in range(len(env))};units={};depth={k:0 for k in inputs};memo={};source_nodes=[]
    def relu(form):
        if not form.terms:return Linear(bias=max(Q(0),form.bias))
        key=(form.terms,form.bias)
        if key not in memo:
            uid=f'u{len(units)}';memo[key]=uid;units[uid]=form
            depth[uid]=1+max(depth[k] for k,_ in form.terms)
            if len(units)>max_units or depth[uid]>max_layers:raise CompilationLimit('neural DAG/depth limit')
        return Linear([(memo[key],Q(1))])
    def clip(form,lo,hi):return relu(form.shift(-lo)).plus(relu(form.shift(-hi)).scale(-1)).shift(lo)
    def lower(expr,path):
        op=expr['op']
        if op=='ref':out=env[expr['name']]
        elif op=='const':out=Linear(bias=rational(expr['value']))
        elif op=='affine':
            out=Linear(bias=rational(expr['bias']))
            for i,t in enumerate(expr['terms']):out=out.plus(lower(t['value'],path+f'/terms/{i}/value').scale(rational(t['coefficient'])))
        elif op in ('relu','clip','margin','not'):
            a=lower(expr['value'],path+'/value')
            if op=='relu':out=relu(a)
            elif op=='not':out=a.scale(-1).shift(1)
            elif op=='clip':out=clip(a,rational(expr['lower']),rational(expr['upper']))
            else:
                lo,hi=rational(expr['lower']),rational(expr['upper'])
                out=clip(a.shift(-lo).scale(1/(hi-lo)),Q(0),Q(1))
        else:
            a=lower(expr['left'],path+'/left');b=lower(expr['right'],path+'/right')
            if op=='min':out=a.plus(relu(a.plus(b.scale(-1))).scale(-1))
            elif op=='max':out=b.plus(relu(a.plus(b.scale(-1))))
            elif op=='and':out=relu(a.plus(b).shift(-1))
            elif op=='or':out=a.plus(b).plus(relu(a.plus(b).shift(-1)).scale(-1))
            else:raise ValueError('unsupported operator after type validation')
        source_nodes.append(dict(path=path,operator=op,linear=out.record()))
        return out
    output_forms=[]
    for group in ('lets','outputs'):
        for i,item in enumerate(d[group]):
            form=lower(item['value'],f'/{group}/{i}/value');env[item['name']]=form
            if group=='outputs':output_forms.append(form)
    reachable=set()
    def visit(key):
        if key in inputs or key in reachable:return
        reachable.add(key)
        for k,_ in units[key].terms:visit(k)
    for form in output_forms:
        for k,_ in form.terms:visit(k)
    hidden_depth=max([depth[k] for k in reachable]+[0]);layers=[[] for _ in range(hidden_depth)]
    allocated={};relays={}
    def append(layer,terms,bias,kind,source):
        index=len(layers[layer-1]);layers[layer-1].append(dict(terms=terms,bias=bias,kind=kind,source=source))
        if sum(len(x) for x in layers)>max_units:raise CompilationLimit('allocated units including relays exceed limit')
        return index
    def combine(form,level):
        row={}
        for key,c in form.terms:
            for i,w in channels(key,level).items():row[i]=row.get(i,Q(0))+c*w
        return row
    def ensure(key):
        if key not in allocated:
            form=units[key];level=depth[key]
            row=combine(form,level-1)
            allocated[key]=(level,append(level,row,form.bias,'source_relu',key))
        return allocated[key]
    def signed_channel(key,level,sign):
        cache=(key,level,sign)
        if cache not in relays:
            prev={inputs[key]:Q(sign)} if level==1 else {signed_channel(key,level-1,sign):Q(1)}
            relays[cache]=append(level,prev,Q(0),'input_positive' if sign==1 else 'input_negative',key)
        return relays[cache]
    def channels(key,level):
        if key in inputs:
            if level==0:return {inputs[key]:Q(1)}
            return {signed_channel(key,level,1):Q(1),signed_channel(key,level,-1):Q(-1)}
        origin,index=ensure(key)
        if origin==level:return {index:Q(1)}
        cache=(key,level,0)
        if cache not in relays:relays[cache]=append(level,channels(key,level-1),Q(0),'nonnegative_relay',key)
        return {relays[cache]:Q(1)}
    for key in units:
        if key in reachable:ensure(key)
    readout=[combine(f,hidden_depth) for f in output_forms]
    widths=[len(inputs)]+[len(layer) for layer in layers]+[len(output_forms)]
    parameter_count=sum(b*(a+1) for a,b in zip(widths,widths[1:]))
    if parameter_count>max_parameters:raise CompilationLimit('dense parameter allocation limit')
    graph=[]
    for level,layer in enumerate(layers,1):
        graph.append(dict(activation='relu',weights=[[str(u['terms'].get(i,Q(0))) for i in range(widths[level-1])] for u in layer],bias=[str(u['bias']) for u in layer]))
    graph.append(dict(activation='linear',weights=[[str(row.get(i,Q(0))) for i in range(widths[-2])] for row in readout],bias=[str(f.bias) for f in output_forms]))
    source_map=dict(expressions=source_nodes,relu_units={k:dict(layer=v[0],index=v[1]) for k,v in allocated.items()},
                    allocation=[[dict(index=i,kind=u['kind'],source=u['source']) for i,u in enumerate(layer)] for layer in layers],
                    eliminated_relu_units=sorted(set(units)-reachable))
    artifact=dict(format='wspl.relu_mlp.v0',semantics='rational_real_graph',compiler='wspl.scalar_to_relu.v0',
                  source_sha256=hashlib.sha256(canonical_bytes(d)).hexdigest(),
                  inputs=[dict(name=k,type={**v,'lower':str(rational(v['lower'])),'upper':str(rational(v['upper']))}) for k,v in d['inputs'].items()],
                  outputs=[dict(name=o['name'],type={**o['type'],'lower':str(rational(o['type']['lower'])),'upper':str(rational(o['type']['upper']))}) for o in d['outputs']],
                  layers=graph,allocation=dict(mode='fresh_mlp',widths=widths,parameters=parameter_count),source_map=source_map)
    return ReLUArtifact(artifact)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        artifact=compile_scalar(json.loads(args.source.read_text(encoding='utf-8')))
        artifact.save(args.output)
        print(json.dumps(dict(output=str(args.output),sha256=artifact.sha256,allocation=artifact.to_dict()['allocation'],universal_verification='not_performed')))
    except (ValueError,OSError) as exc:
        print(str(exc),file=sys.stderr);raise SystemExit(1)


if __name__=='__main__':main()
