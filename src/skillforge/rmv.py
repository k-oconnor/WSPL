"""Scalar reference for an exported grounded RMV block, including motor readout.

No torch model, optimizer, simplifier or robotic controller is implemented here.
"""
import math


def finite(value):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Expected finite numeric value')
    return float(value)


def clip(x,lo,hi):return min(hi,max(lo,x))


class RMVReference:
    def __init__(self,graph):
        expected={'format','inputs','literal_sources','and_weights','or_weights','readout_weights','readout_bias','output_lower','output_upper'}
        if set(graph)!=expected or graph['format']!='skillforge.rmv.v1':raise ValueError('Unknown RMV graph format')
        inp=graph['inputs']
        if not isinstance(inp,dict) or set(inp)!={'names','lower','upper'}:raise ValueError('Invalid inputs')
        if not isinstance(inp['names'],list):raise ValueError('Expected input channel list')
        self.names=tuple(inp['names'])
        if not self.names or any(not isinstance(n,str) or not n for n in self.names) or len(set(self.names))!=len(self.names):raise ValueError('Invalid channel names')
        p=len(self.names)
        def vector(values,n):
            if not isinstance(values,list) or len(values)!=n:raise ValueError('Vector shape differs')
            return tuple(finite(v) for v in values)
        def matrix(values,columns):
            if not isinstance(values,list) or not values:raise ValueError('Empty matrix')
            return tuple(vector(row,columns) for row in values)
        self.lower=vector(inp['lower'],p);self.upper=vector(inp['upper'],p)
        if any(lo>=hi for lo,hi in zip(self.lower,self.upper)):raise ValueError('Normalization interval must have positive width')
        sources=graph['literal_sources']
        if not isinstance(sources,list) or not sources:raise ValueError('Expected grounded literals')
        literals=[]
        for source in sources:
            if not isinstance(source,dict) or set(source)!={'input','negated'}:raise ValueError('Invalid literal source')
            i,negated=source['input'],source['negated']
            if type(i) is not int or not 0<=i<p or type(negated) is not bool:raise ValueError('Invalid literal grounding')
            literals.append((i,negated))
        if len(set(literals))!=len(literals):raise ValueError('Duplicate literal source')
        self.literals=tuple(literals)
        self.w=matrix(graph['and_weights'],len(self.literals));self.v=matrix(graph['or_weights'],len(self.w))
        if any(x<0 for matrix_ in (self.w,self.v) for row in matrix_ for x in row):raise ValueError('Logical weights must be nonnegative')
        self.a=matrix(graph['readout_weights'],len(self.v));outputs=len(self.a)
        self.b=vector(graph['readout_bias'],outputs)
        self.out_lower=vector(graph['output_lower'],outputs);self.out_upper=vector(graph['output_upper'],outputs)
        if any(lo>=hi for lo,hi in zip(self.out_lower,self.out_upper)):raise ValueError('Invalid output bounds')

    def evaluate(self,observation):
        if set(observation)!=set(self.names):raise ValueError('Observation channels differ')
        x=[finite(observation[n]) for n in self.names]
        predicates=[clip((v-lo)/(hi-lo),0,1) for v,lo,hi in zip(x,self.lower,self.upper)]
        literals=[1-predicates[i] if negated else predicates[i] for i,negated in self.literals]
        clauses=[clip(1-sum(w*(1-l) for w,l in zip(row,literals)),0,1) for row in self.w]
        heads=[clip(sum(w*c for w,c in zip(row,clauses)),0,1) for row in self.v]
        commands=[clip(sum(w*h for w,h in zip(row,heads))+b,lo,hi) for row,b,lo,hi in zip(self.a,self.b,self.out_lower,self.out_upper)]
        return dict(predicates=predicates,literals=literals,clauses=clauses,heads=heads,commands=commands)
