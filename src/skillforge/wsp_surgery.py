"""Bind graph-edit syntax to host-owned budgets and pinned module artifacts."""
import ast
from dataclasses import asdict
from fractions import Fraction

from .wsp_artifact import digest
from .wsp_syntax import expression
from .wsp_values import scalar, numeric
from .wsp_graph import (GraphBudget, identifier, declared_region, transplant,
                        replace_region, redirect, untie, add_residual)


KINDS = {'replace','transplant','add_residual','redirect','untie'}


def name(text):
    n=expression(text).body
    value=n.id if isinstance(n,ast.Name) else n.value if isinstance(n,ast.Constant) else None
    if not identifier(value):raise ValueError('identifier mapping key/value required')
    return value


def call_string(text, function):
    n=expression(text).body
    if (not isinstance(n,ast.Call) or not isinstance(n.func,ast.Name) or n.func.id!=function
            or n.keywords or len(n.args)!=1 or not isinstance(n.args[0],ast.Constant) or not isinstance(n.args[0].value,str)):
        raise ValueError(f'{function}("literal") required')
    return n.args[0].value


def sequence(text, transform):
    tree=expression(text).body
    if not isinstance(tree,ast.List):raise ValueError('literal list required')
    return [transform(ast.get_source_segment(text,n)) for n in tree.elts]


def mapping(values, key, value):
    if not isinstance(values,dict):raise ValueError('explicit graph mapping required')
    out={}
    for k,v in values.items():
        k=key(k)
        if k in out:raise ValueError('duplicate normalized mapping key')
        out[k]=value(v)
    return out


def boundary(text, region, side):
    n=expression(text).body
    if (not isinstance(n,ast.Attribute) or not isinstance(n.value,ast.Attribute)
            or not isinstance(n.value.value,ast.Name) or n.value.value.id!=region or n.value.attr!=side):
        raise ValueError(f'{region}.{side}.alias required')
    return n.attr


class GraphContext:
    def __init__(self, parent, tree, artifacts, budgets):
        if parent.to_dict()['graph']['kind']!='dag':
            raise ValueError('graph mode needs an explicit graphify conversion first')
        self.parent=parent;self.artifacts=dict(artifacts or {});self.used={};self.reports=[]
        self.budget_name=tree['edit']['budget']
        if self.budget_name not in (budgets or {}) or not isinstance(budgets[self.budget_name],GraphBudget):
            raise ValueError('missing host-owned graph budget')
        self.budget=budgets[self.budget_name];self.budget.check(parent)
        self.regions={}
        for key,spec in tree['regions'].items():
            if spec['target']!=tree['target']['name']:raise ValueError('region target differs from parent')
            f=spec['fields']
            if not {'nodes','inputs','outputs'}<=set(f) or set(f)-{'nodes','inputs','outputs','shared_parameters'}:
                raise ValueError('region fields differ')
            nodes=sequence(f['nodes'],lambda s:call_string(s,'node'))
            inputs=mapping(f['inputs'],name,lambda s:call_string(s,'edge'))
            outputs=mapping(f['outputs'],name,lambda s:call_string(s,'edge'))
            shared=sequence(f.get('shared_parameters','[]'),name)
            self.regions[key]=declared_region(parent,nodes,inputs,outputs,shared)

    def module(self, text):
        n=expression(text).body
        if (not isinstance(n,ast.Call) or not isinstance(n.func,ast.Name) or n.func.id!='artifact'
                or len(n.args)!=1 or not isinstance(n.args[0],ast.Constant) or not isinstance(n.args[0].value,str)
                or len(n.keywords)!=1 or n.keywords[0].arg!='sha256' or not isinstance(n.keywords[0].value,ast.Constant)):
            raise ValueError('pinned artifact(name,sha256=...) required')
        key=n.args[0].value
        if key not in self.artifacts or self.artifacts[key].sha256!=n.keywords[0].value.value:
            raise ValueError('module artifact identity differs')
        self.used[key]=self.artifacts[key]
        return self.artifacts[key]

    def execute(self, builder, row):
        before=builder.current();kind=row['kind']
        if kind=='redirect':
            candidate,report=redirect(before,call_string(row['edge'],'edge'),call_string(row['signal'],'signal'),budget=self.budget)
        elif kind=='untie':
            candidate,report=untie(before,call_string(row['parameter'],'parameter'),row['name'],budget=self.budget)
        else:
            patch=self.module(row['artifact']);f=row['fields']
            allowed={'inputs','share'} if kind=='transplant' else {'inputs','outputs','share','drop_inputs'} if kind=='replace' else {'inputs','output','into','scale','share'}
            required={'inputs'} if kind=='transplant' else {'inputs','outputs'} if kind=='replace' else {'inputs','output','into','scale'}
            if set(f)-allowed or not required<=set(f):raise ValueError('graph operation fields differ')
            share=mapping(f.get('share',{}),name,name)
            if kind=='replace':
                key=row['region']
                if key not in self.regions:raise ValueError('unknown declared region')
                inputs=mapping(f['inputs'],name,lambda s:boundary(s,key,'inputs'))
                outputs=mapping(f['outputs'],lambda s:boundary(s,key,'outputs'),name)
                candidate,report=replace_region(before,self.regions[key],patch,prefix=row['prefix'],inputs=inputs,outputs=outputs,
                                                drop_inputs=sequence(f.get('drop_inputs','[]'),name),share=share,budget=self.budget)
            else:
                inputs=mapping(f['inputs'],name,lambda s:call_string(s,'signal'))
                if kind=='transplant':
                    candidate,report=transplant(before,patch,prefix=row['prefix'],inputs=inputs,share=share,budget=self.budget)
                else:
                    scale=scalar(numeric(builder.arithmetic.run(f['scale'])))
                    if not isinstance(scale,Fraction):raise ValueError('rational residual scale required')
                    candidate,report=add_residual(before,patch,prefix=row['prefix'],inputs=inputs,output=name(f['output']),
                                                  into=call_string(f['into'],'edge'),scale=scale,share=share,budget=self.budget)
        self.budget.check(candidate,self.parent)
        if set(candidate.to_dict()['tensors']) & set(builder.arithmetic.constants):
            raise ValueError('new graph tensor shadows a construction constant')
        report['source_span']=row['span'];self.reports.append(report)
        builder.base=candidate;builder.arrays=candidate.arrays(bits=True);builder.arithmetic.arrays=builder.arrays

    def receipt(self):
        budget=asdict(self.budget)
        return dict(budget_name=self.budget_name,budget=budget,budget_id=digest(budget),regions=self.regions,
                    modules={k:v.to_dict() for k,v in self.used.items()},operations=self.reports)
