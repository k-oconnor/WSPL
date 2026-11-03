"""Supplied reusable task hierarchies and macro-summary monitor lowering.

This expands/verifies given decompositions, not hierarchy discovery. Sequenced
summaries denote ordered outcome witnesses, not a hierarchical execution stack.
Unordered branches are supported only at the root, avoiding ambiguous nesting.
"""
from collections import Counter
import hashlib
import json
import re
from .ltlf import Expr


def _name(value):
    if not isinstance(value,str) or not re.fullmatch(r'[a-z][a-z0-9_]*',value):raise ValueError('Invalid symbolic identifier')
    return value


def _bind(pattern,bindings):
    if not isinstance(pattern,str):raise ValueError('Expected binding string')
    def substitute(match):
        key=match.group(1)
        if key not in bindings:raise ValueError('Unbound template parameter: '+key)
        return bindings[key]
    value=re.sub(r'\{([a-z][a-z0-9_]*)\}',substitute,pattern)
    return _name(value)


def _combine(op,parts):
    if not parts:raise ValueError('Empty task expression')
    return parts[0] if len(parts)==1 else Expr(op,tuple(parts))


def _word(events):
    if not events:raise ValueError('Empty outcome sequence')
    tail=Expr('F',(Expr('ap',name=events[-1]),))
    for event in reversed(events[:-1]):tail=Expr('F',(Expr('and',(Expr('ap',name=event),Expr('X',(tail,)))),))
    return tail


class Hierarchy:
    def __init__(self,templates,max_depth=16,max_primitives=256):
        if not isinstance(templates,dict) or not templates or min(max_depth,max_primitives)<1:raise ValueError('Invalid template library/limits')
        self.templates=json.loads(json.dumps(templates));self.max_depth=max_depth;self.max_primitives=max_primitives
        for name,template in self.templates.items():
            _name(name)
            if set(template)-{'parameters','body','summary_event'} or not {'parameters','body'}<=set(template):raise ValueError('Invalid template fields')
            params=template['parameters']
            if not isinstance(params,list) or any(_name(p)!=p for p in params) or len(set(params))!=len(params):raise ValueError('Invalid parameters')

    def _walk(self,root,cut=False):
        calls=Counter();primitives=[];max_seen=0
        def visit(node,bindings,stack):
            nonlocal max_seen
            if len(stack)>self.max_depth:raise ValueError('Hierarchy depth limit exceeded')
            max_seen=max(max_seen,len(stack))
            if not isinstance(node,dict) or 'op' not in node:raise ValueError('Expected hierarchy node')
            kind=node['op']
            if kind=='call':
                if set(node)!={'op','template','bindings'}:raise ValueError('Invalid call fields')
                name=node['template']
                if name not in self.templates:raise ValueError('Unknown subtask template')
                if name in stack:raise ValueError('Recursive subtask cycle')
                template=self.templates[name];arguments=node['bindings']
                if not isinstance(arguments,dict) or set(arguments)!=set(template['parameters']):raise ValueError('Call parameter mismatch')
                bound={p:_bind(v,bindings) for p,v in arguments.items()};calls[name]+=1
                if cut and 'summary_event' in template:return dict(op='summary',event=_bind(template['summary_event'],bound))
                return dict(op='subtask',template=name,bindings=bound,body=visit(template['body'],bound,stack+(name,)))
            if kind=='skill':
                if set(node)!={'op','name','bindings'} or not isinstance(node['bindings'],dict):raise ValueError('Invalid primitive fields')
                if cut:raise ValueError('Every macro-cut primitive requires a summary-bearing template')
                result=dict(op='skill',name=_name(node['name']),bindings={_name(k):_bind(v,bindings) for k,v in node['bindings'].items()})
                primitives.append(result)
                if len(primitives)>self.max_primitives:raise ValueError('Primitive expansion limit exceeded')
                return result
            if kind not in ('sequence','all') or set(node)!={'op','children'} or not isinstance(node['children'],list) or not node['children']:raise ValueError('Invalid hierarchy operator')
            return dict(op=kind,children=[visit(child,bindings,stack) for child in node['children']])
        expanded=visit(root,{},())
        return expanded,dict(template_calls=dict(calls),primitive_calls=len(primitives),
            primitive_skill_names=sorted({p['name'] for p in primitives}),maximum_template_depth=max_seen)

    def expand(self,root):return self._walk(root)

    def outcome_words(self,root):
        # Full expansion validates cycles and primitive structure even below a cut.
        self.expand(root);cut,_=self._walk(root,True)
        def word(node):
            if node['op']=='summary':return [node['event']]
            if node['op']=='subtask':return word(node['body'])
            if node['op']=='sequence':return sum((word(c) for c in node['children']),[])
            raise ValueError('Unordered branches may appear only at the hierarchy root')
        return [word(c) for c in cut['children']] if cut['op']=='all' else [word(cut)]

    def formula(self,root,terminal_goals,avoid=('collision',)):
        if not terminal_goals:raise ValueError('Declare physical terminal goals')
        words=self.outcome_words(root);clauses=[_word(w) for w in words]
        end_condition=_combine('and',[Expr('ap',name=_name(p)) for p in terminal_goals])
        last=Expr('not',(Expr('X',(Expr('true'),)),))
        clauses.append(Expr('F',(Expr('and',(last,end_condition)),)))
        clauses.extend(Expr('G',(Expr('not',(Expr('ap',name=_name(p)),)),)) for p in avoid)
        return _combine('and',clauses)

    @property
    def identity(self):return hashlib.sha256(json.dumps(self.templates,sort_keys=True,separators=(',',':')).encode()).hexdigest()
