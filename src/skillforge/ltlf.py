"""Typed LTLf AST and direct semantics for completed, nonempty finite traces.

This is a reference evaluator, not a DFA compiler, online verdict monitor,
realizability checker or controller synthesizer. Prefix false is not failure.
"""
from dataclasses import dataclass
from functools import lru_cache
import re

ARITY={'true':0,'false':0,'ap':0,'not':1,'X':1,'Xw':1,'F':1,'G':1,'U':2,'implies':2}


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple['Expr', ...] = ()
    name: str | None = None

    def __post_init__(self):
        if self.op not in ARITY and self.op not in ('and','or'):raise ValueError('Unsupported operator')
        if not isinstance(self.args,tuple) or any(not isinstance(a,Expr) for a in self.args):raise ValueError('Expected expression tuple')
        if self.op in ('and','or'):
            if len(self.args)<2:raise ValueError('Expected at least two arguments')
        elif len(self.args)!=ARITY[self.op]:raise ValueError('Invalid arity')
        if self.op=='ap':
            if not isinstance(self.name,str) or not re.fullmatch(r'[a-z][a-z0-9_]*',self.name):raise ValueError('Invalid proposition')
        elif self.name is not None:raise ValueError('Only propositions have names')

    @property
    def atoms(self):
        return frozenset([self.name]) if self.op=='ap' else frozenset().union(*(a.atoms for a in self.args))

    def to_dict(self):
        if self.op=='ap':return dict(op=self.op,name=self.name)
        if self.op in ('true','false'):return dict(op=self.op)
        return dict(op=self.op,args=[a.to_dict() for a in self.args])


def parse(data,max_nodes=256,max_depth=32):
    count=0
    def visit(node,depth):
        nonlocal count
        count+=1
        if count>max_nodes or depth>max_depth:raise ValueError('Formula exceeds declared limits')
        if not isinstance(node,dict) or not isinstance(node.get('op'),str):raise ValueError('Expected AST node')
        op=node['op']
        if op=='ap':
            if set(node)!={'op','name'} or not isinstance(node['name'],str) or not re.fullmatch(r'[a-z][a-z0-9_]*',node['name']):
                raise ValueError('Invalid proposition')
            return Expr(op,name=node['name'])
        if op in ('true','false'):
            if set(node)!={'op'}:raise ValueError('Unexpected constant fields')
            return Expr(op)
        if op not in ARITY and op not in ('and','or'):raise ValueError('Unsupported operator')
        if set(node)!={'op','args'} or not isinstance(node['args'],list):raise ValueError('Expected arguments')
        n=len(node['args'])
        if (op in ('and','or') and n<2) or (op not in ('and','or') and n!=ARITY[op]):raise ValueError('Invalid arity')
        return Expr(op,tuple(visit(a,depth+1) for a in node['args']))
    return visit(data,0)


def evaluate_closed_trace(formula,trace):
    """Boolean satisfaction at position zero after the trace has ended."""
    if not isinstance(formula,Expr):formula=parse(formula)
    if not trace:raise ValueError('This interface requires a nonempty closed trace')
    atoms=formula.atoms
    for valuation in trace:
        if not isinstance(valuation,dict) or any(a not in valuation or type(valuation[a]) is not bool for a in atoms):
            raise ValueError('Every referenced proposition needs an explicit Boolean observation')
    @lru_cache(None)
    def truth(f,i):
        op=f.op;a=f.args
        if op=='true':return True
        if op=='false':return False
        if op=='ap':return trace[i][f.name]
        if op=='not':return not truth(a[0],i)
        if op=='and':return all(truth(x,i) for x in a)
        if op=='or':return any(truth(x,i) for x in a)
        if op=='implies':return not truth(a[0],i) or truth(a[1],i)
        if op=='X':return i+1<len(trace) and truth(a[0],i+1)
        if op=='Xw':return i+1==len(trace) or truth(a[0],i+1)
        if op=='F':return any(truth(a[0],j) for j in range(i,len(trace)))
        if op=='G':return all(truth(a[0],j) for j in range(i,len(trace)))
        if op=='U':return any(truth(a[1],j) and all(truth(a[0],k) for k in range(i,j)) for j in range(i,len(trace)))
        raise ValueError('Unsupported operator')
    return truth(formula,0)
