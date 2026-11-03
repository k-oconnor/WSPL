"""Small-alphabet LTLf derivative DFA with canonical Boolean residuals.

Reduced ordered BDDs canonicalize residual Boolean functions over the fixed
formula closure. A nonempty-suffix marker distinguishes strong/weak next at
the last position. Construction limits raise explicitly, never 'unsatisfiable'.
"""
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import json
from .ltlf import Expr,parse


class CompilationLimit(RuntimeError):pass


class _BDD:
    def __init__(self,max_nodes):
        self.nodes=[None,None];self.unique={};self.max_nodes=max_nodes
        self.negate=lru_cache(None)(self._negate)
        self.apply=lru_cache(None)(self._apply)

    def node(self,var,low,high):
        if low==high:return low
        key=(var,low,high)
        if key not in self.unique:
            if len(self.nodes)>=self.max_nodes:raise CompilationLimit('BDD node limit exceeded')
            self.unique[key]=len(self.nodes);self.nodes.append(key)
        return self.unique[key]

    def _negate(self,u):
        if u<2:return 1-u
        v,lo,hi=self.nodes[u]
        return self.node(v,self.negate(lo),self.negate(hi))

    def _apply(self,op,u,v):
        if op=='and':
            if u==0 or v==0:return 0
            if u==1:return v
            if v==1 or u==v:return u
        elif op=='or':
            if u==1 or v==1:return 1
            if u==0:return v
            if v==0 or u==v:return u
        else:raise ValueError('Unsupported Boolean operation')
        a=self.nodes[u][0];b=self.nodes[v][0];top=min(a,b)
        ulo,uhi=self.nodes[u][1:] if a==top else (u,u)
        vlo,vhi=self.nodes[v][1:] if b==top else (v,v)
        return self.node(top,self.apply(op,ulo,vlo),self.apply(op,uhi,vhi))

    def ite(self,c,yes,no):
        return self.apply('or',self.apply('and',c,yes),self.apply('and',self.negate(c),no))


@dataclass(frozen=True)
class DFA:
    formula: Expr
    alphabet: tuple[str,...]
    transitions: tuple[tuple[int,...],...]
    accepting: frozenset[int]
    completable: frozenset[int]
    bdd_nodes: int
    initial: int = 0

    def step(self,state,valuation):
        if type(state) is not int or not 0<=state<len(self.transitions):raise ValueError('Unknown DFA state')
        if not isinstance(valuation,dict) or any(a not in valuation or type(valuation[a]) is not bool for a in self.alphabet):
            raise ValueError('Every proposition needs an explicit Boolean value')
        mask=sum(int(valuation[a])<<i for i,a in enumerate(self.alphabet))
        return self.transitions[state][mask]

    def accepts(self,trace):
        if not trace:raise ValueError('Expected nonempty finite trace')
        state=self.initial
        for valuation in trace:state=self.step(state,valuation)
        return state in self.accepting

    def to_dict(self):
        return dict(schema_version=1,kind='ltlf_dfa',formula=self.formula.to_dict(),alphabet=list(self.alphabet),
            initial=self.initial,accepting=sorted(self.accepting),completable=sorted(self.completable),
            transitions=[list(row) for row in self.transitions],bdd_nodes=self.bdd_nodes,
            letter_encoding='bit i corresponds to alphabet[i]',empty_trace_accepted=False)


def compile_dfa(formula,max_atoms=8,max_states=512,max_bdd_nodes=50000):
    if not isinstance(formula,Expr):formula=parse(formula)
    if min(max_atoms,max_states,max_bdd_nodes)<1:raise ValueError('Expected positive construction limits')
    alphabet=tuple(sorted(formula.atoms))
    if len(alphabet)>max_atoms:raise CompilationLimit('Explicit alphabet limit exceeded')
    closure=set()
    def collect(f):
        if f.op in ('ap','X','Xw','F','G','U'):closure.add(f)
        for a in f.args:collect(a)
    collect(formula)
    variables=tuple(sorted(closure,key=lambda f:json.dumps(f.to_dict(),sort_keys=True,separators=(',',':'))))
    indices={f:i for i,f in enumerate(variables)};nonempty=len(variables);bdd=_BDD(max_bdd_nodes)
    @lru_cache(None)
    def encode(f):
        if f.op in ('true','false'):return int(f.op=='true')
        if f in indices:return bdd.node(indices[f],0,1)
        if f.op=='not':return bdd.negate(encode(f.args[0]))
        if f.op=='implies':return bdd.apply('or',bdd.negate(encode(f.args[0])),encode(f.args[1]))
        value=int(f.op=='and')
        for a in f.args:value=bdd.apply(f.op,value,encode(a))
        return value
    suffix_exists=bdd.node(nonempty,0,1)
    @lru_cache(None)
    def nullable(u):
        if u<2:return bool(u)
        v,lo,hi=bdd.nodes[u]
        empty_value=v!=nonempty and variables[v].op in ('true','G','Xw')
        return nullable(hi if empty_value else lo)
    atom_index={a:i for i,a in enumerate(alphabet)}
    @lru_cache(None)
    def derivative_var(v,mask):
        if v==nonempty:return 1
        f=variables[v];op=f.op;a=f.args
        if op=='ap':return (mask>>atom_index[f.name])&1
        if op=='X':return bdd.apply('and',encode(a[0]),suffix_exists)
        if op=='Xw':return bdd.apply('or',encode(a[0]),bdd.negate(suffix_exists))
        if op=='F':return bdd.apply('or',derivative(encode(a[0]),mask),encode(f))
        if op=='G':return bdd.apply('and',derivative(encode(a[0]),mask),encode(f))
        if op=='U':return bdd.apply('or',derivative(encode(a[1]),mask),bdd.apply('and',derivative(encode(a[0]),mask),encode(f)))
        raise ValueError('Unknown closure variable')
    @lru_cache(None)
    def derivative(u,mask):
        if u<2:return u
        v,lo,hi=bdd.nodes[u]
        return bdd.ite(derivative_var(v,mask),derivative(hi,mask),derivative(lo,mask))
    initial=(encode(formula),False);states=[initial];ids={initial:0};rows=[];accepting=set()
    for root,seen in states:
        index=len(rows)
        if seen and nullable(root):accepting.add(index)
        row=[]
        for mask in range(1<<len(alphabet)):
            target=(derivative(root,mask),True)
            if target not in ids:
                if len(states)>=max_states:raise CompilationLimit('DFA state limit exceeded')
                ids[target]=len(states);states.append(target)
            row.append(ids[target])
        rows.append(tuple(row))
    reverse=[set() for _ in rows]
    for source,row in enumerate(rows):
        for target in row:reverse[target].add(source)
    completable=set(accepting);queue=deque(accepting)
    while queue:
        for previous in reverse[queue.popleft()]-completable:completable.add(previous);queue.append(previous)
    return DFA(formula,alphabet,tuple(rows),frozenset(accepting),frozenset(completable),len(bdd.nodes))
