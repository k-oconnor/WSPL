"""Matched direct/structured neural graph repair screen; no live collection here."""
from dataclasses import asdict
from fractions import Fraction
import json
import re

import numpy as np

from .wsp_artifact import TensorArtifact,digest,round_f32,diff
from .wsp_editor import plan,apply
from .wsp_evidence import describe,check
from .wsp_graph import GraphBudget,node
from .wsp_syntax import parse


ARMS=('direct','wspl')
TASKS=('shared_gain','clipped_drive','isolated_deadband')
BUDGET=GraphBudget(max_nodes=24,max_edges=48,max_parameters=128,max_depth=12,max_live_values=128,
                   allowed_ops=('affine','relu','identity','add','scale','concat','slice'))
MAX_REPLY=200_000


def strict_json(text):
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    return json.loads(text,object_pairs_hook=pairs,parse_float=str,
                      parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def tensor(record):
    if not isinstance(record,dict) or set(record) not in ({'shape','values'},{'shape','bits'}):raise ValueError('tensor shape and values OR bits required')
    shape=record['shape']
    if not isinstance(shape,list) or not 1<=len(shape)<=8 or any(type(n) is not int or n<1 for n in shape):raise ValueError('tensor shape invalid')
    from math import prod
    if prod(shape)>BUDGET.max_parameters:raise ValueError('tensor too large')
    if 'bits' in record:words=record['bits']
    else:
        if not isinstance(record['values'],list) or len(record['values'])!=prod(shape):raise ValueError('flat tensor values differ')
        words=[]
        for v in record['values']:
            if type(v) not in (str,int) or len(str(v))>1000:raise ValueError('rational string or integer required')
            words.append(f'{round_f32(Fraction(v)):08x}')
    # Validate through the common artifact loader, including bit count/nonfinites.
    d=dict(format='wsp.tensor_artifact.v1',graph={'kind':'opaque','label':'literal'},tensors={'value':dict(shape=shape,dtype='f32',bits=words)})
    return TensorArtifact(d).to_dict()['tensors']['value']


def module(document):
    if not isinstance(document,dict) or set(document)!={'graph','tensors'} or not isinstance(document['tensors'],dict):raise ValueError('module requires graph/tensors')
    return TensorArtifact(dict(format='wsp.tensor_artifact.v1',graph=document['graph'],tensors={k:tensor(v) for k,v in document['tensors'].items()}),max_elements=BUDGET.max_parameters)


def fixed_source(parent,body):
    return f'wspl 0.2; target base=artifact("base",sha256="{parent.sha256}"); edit base as candidate mode=fixed {{ {body} }}'


def raw_edit(parent,raw):
    if not isinstance(raw,dict) or set(raw)-{'put_tensors','drop_tensors','put_nodes','drop_nodes','outputs','tensor_source'}:raise ValueError('raw fields differ')
    d=parent.to_dict();report={}
    for key in ('put_tensors','outputs'):
        if not isinstance(raw.get(key,{}),dict):raise ValueError('raw map required')
    for key in ('drop_tensors','drop_nodes'):
        v=raw.get(key,[])
        if not isinstance(v,list) or any(not isinstance(x,str) for x in v) or len(v)!=len(set(v)):raise ValueError('unique raw deletion names required')
    if set(raw.get('drop_tensors',[]))&set(raw.get('put_tensors',{})):raise ValueError('ambiguous tensor replace/delete')
    for key in raw.get('drop_tensors',[]):del d['tensors'][key]
    d['tensors'].update({k:tensor(v) for k,v in raw.get('put_tensors',{}).items()})
    existing={n['id']:n for n in d['graph']['nodes']};added=raw.get('put_nodes',[])
    if not isinstance(added,list) or any(not isinstance(n,dict) or not isinstance(n.get('id'),str) for n in added):raise ValueError('raw node list required')
    if len({n['id'] for n in added})!=len(added) or {n['id'] for n in added}&set(raw.get('drop_nodes',[])):raise ValueError('ambiguous node replace/delete')
    for key in raw.get('drop_nodes',[]):del existing[key]
    existing.update({n['id']:n for n in added});d['graph']['nodes']=list(existing.values());d['graph']['outputs'].update(raw.get('outputs',{}))
    candidate=TensorArtifact(d,max_elements=BUDGET.max_parameters);BUDGET.check(candidate,parent)
    body=raw.get('tensor_source','')
    if not isinstance(body,str):raise ValueError('tensor_source must be text')
    if body:
        source=fixed_source(candidate,body);tree=parse(source)
        if (tree['constants'] or tree['circuits'] or tree['regions'] or tree['descriptions'] or tree['edit']['mode']!='fixed'
                or tree['edit']['numeric']!='exact_to_storage'):raise ValueError('raw source only supports direct tensor construction')
        p=plan(source,parent=candidate);candidate=apply(p,parent=candidate,expected_parent_id=candidate.sha256);report['tensor_plan']=p.to_dict()
    BUDGET.check(candidate,parent)
    return candidate,report


def task(task_id):
    if task_id not in TASKS:raise ValueError('unknown task')
    # Explicit supplied parents, not trained policies. Node IDs are neutral.
    nodes=[node('s0','slice',{'x':'input:x'},attrs={'start':0,'stop':1,'step':1}),node('s1','slice',{'x':'input:x'},attrs={'start':1,'stop':2,'step':1})]
    arrays={'W':np.array([[2 if task_id!='isolated_deadband' else 1]],np.float32),'b':np.array([0],np.float32)}
    nodes.extend([node('n0','affine',{'x':'node:s0'},params={'weight':'W','bias':'b'},attrs={'layout':'in_out'}),node('n1','affine',{'x':'node:s1'},params={'weight':'W','bias':'b'},attrs={'layout':'in_out'})])
    if task_id=='shared_gain':
        outputs={'drive':'node:n0','keep':'node:n1'};repair='3*x[0]';preserve='2*x[1]'
        request='Change the first channel gain from two to three while keeping the second channel gain exactly two. The current affine branches share their weight and bias.'
    elif task_id=='clipped_drive':
        outputs={'drive':'node:n0','keep':'node:s1'};repair='min(1/2,max(-1/2,2*x[0]))';preserve='x[1]'
        request='Retain slope two near zero but saturate the drive output at minus/plus one half. Preserve the other input channel exactly.'
    else:
        nodes.extend([node('n2','add',{'a':'node:n0','b':'node:n1'}),node('n3','relu',{'x':'node:n1'})])
        outputs={'drive':'node:n2','keep':'node:n3'};repair='relu(x[0]-1/4)-relu(-x[0]-1/4)';preserve='relu(x[1])'
        request='Remove cross-channel interference and give the drive an odd deadband of radius one quarter, with unit slope outside it. Preserve the second channel rectifier.'
    parent=TensorArtifact.from_arrays(arrays,graph=dict(kind='dag',inputs={'x':2},nodes=nodes,outputs=outputs))
    claims=f'''describe candidate {{
      claim repair semantics=real_graph {{ assume x[0] in [-1,1] and x[1] in [-1,1]; ensure drive[0]=={repair}; }}
      claim preservation semantics=real_graph {{ assume x[0] in [-1,1] and x[1] in [-1,1]; ensure keep[0]=={preserve}; }}
    }}'''
    return dict(id=task_id,parent=parent,requirements=claims,request=request)


def numerical_target(task_id,x):
    a,b=map(float,x)
    if task_id=='shared_gain':return {'drive':[3*a],'keep':[2*b]}
    if task_id=='clipped_drive':return {'drive':[max(-.5,min(.5,2*a))],'keep':[b]}
    return {'drive':[max(0,a-.25)-max(0,-a-.25)],'keep':[max(0,b)]}


class RepairStudy:
    def construct(self,task_id,arm,answer):
        if arm not in ARMS:raise ValueError('unknown arm')
        if not isinstance(answer,str) or len(answer.encode('utf-8'))>MAX_REPLY:raise ValueError('reply size limit')
        blocks=re.findall(r'```json\s*\n(.*?)```',answer,re.S)
        if len(blocks)!=1:raise ValueError('exactly one fenced JSON proposal required')
        reply=strict_json(blocks[0])
        allowed={'raw'} if arm=='direct' else {'raw','modules','wspl'}
        if not isinstance(reply,dict) or set(reply)-allowed:raise ValueError('proposal fields not allowed for arm')
        t=task(task_id);parent=t['parent'];base,report=raw_edit(parent,reply.get('raw',{}))
        report.update(raw_parent_id=parent.sha256,raw_candidate_id=base.sha256)
        if 'wspl' in reply:
            if not isinstance(reply['wspl'],str) or not isinstance(reply.get('modules',{}),dict) or len(reply.get('modules',{}))>8:raise ValueError('WSPL source/module limits')
            modules={k:module(v) for k,v in reply.get('modules',{}).items()}
            source=reply['wspl'].replace('{{base_sha256}}',base.sha256)
            for k,a in modules.items():
                if not re.fullmatch('[A-Za-z_][A-Za-z_0-9]*',k):raise ValueError('module alias invalid')
                source=source.replace('{{module_sha256:'+k+'}}',a.sha256)
            tree=parse(source)
            if tree['descriptions']:raise ValueError('requirements belong to evaluator')
            if tree['edit'] and tree['edit']['numeric']!='exact_to_storage':raise ValueError('common exact-to-FP32 numeric profile required')
            p=plan(source,parent=base,artifacts=modules,budgets={'repair':BUDGET})
            base=apply(p,parent=base,expected_parent_id=base.sha256);report['wspl_plan']=p.to_dict()
            report['module_ids']={k:v.sha256 for k,v in modules.items()}
        elif reply.get('modules'):raise ValueError('unused module bundle')
        BUDGET.check(base,parent);report.update(candidate_id=base.sha256,diff=diff(parent,base),resources=BUDGET.check(base))
        return base,report

    def evaluate(self,task_id,candidate):
        t=task(task_id);BUDGET.check(candidate,t['parent'])
        description=describe(candidate,t['requirements'],origin='external_requirement')
        proof=check(candidate,description,timeout_ms=5000,max_parameters=128)
        # Fixed dyadic panel includes thresholds and both sides; not a universal IEEE proof.
        grid=[-1,-.5,-.25,-.25-2**-16,-.25+2**-16,0,.25-2**-16,.25,.25+2**-16,.5,1]
        samples=[{'x':[a,b]} for a in grid for b in grid]
        rng=np.random.default_rng(814230)
        samples += [{'x':[int(a)/1024,int(b)/1024]} for a,b in rng.integers(-1024,1025,size=(128,2))]
        maximum=0.;finite=True;completed=0;failure=None
        for row in samples:
            try:
                actual=candidate.trace(row,semantics='numpy_f32');wanted=numerical_target(task_id,row['x'])
                values=[abs(float(actual[k][0])-v[0]) for k,v in wanted.items()]
                if not np.isfinite(values).all():finite=False;failure='nonfinite output';break
                maximum=max(maximum,*values);completed+=1
            except (ValueError,ArithmeticError,TypeError) as exc:finite=False;failure=str(exc);break
        passed=finite and maximum<=5e-6
        return dict(success=proof['status']=='PROVED' and passed,formal=proof,description=description.to_dict(),
                    numerical=dict(status='PASS' if passed else 'FAIL',samples=len(samples),completed_samples=completed,failure=failure,max_absolute_error=maximum,
                        tolerance=5e-6,fixture_id=digest(samples),universal_ieee_claim=False),
                    scope='supplied synthetic stateless circuit repair; no learned/physical controller claim')


def public_task(task_id):
    t=task(task_id);a=t['parent'];d=a.to_dict()
    return dict(id=task_id,request=t['request'],input_domain={'x':['[-1,1]','[-1,1]']},parent_id=a.sha256,
        parent_graph=d['graph'],parent_tensors={k:dict(shape=list(v.shape),values=[str(float(x)) for x in v.flat]) for k,v in a.arrays().items()},
        requirements=t['requirements'],budget=asdict(BUDGET))


def prompt(task_id,arm,guide):
    if arm not in ARMS:raise ValueError('unknown arm')
    return ('Repair the supplied neural graph through the selected interface. One response; no tools, retries or feedback. '
        'Return a short explanation and exactly one fenced json proposal. Success requires BOTH external claims and the common numerical check. '
        'All valid repairs within the common budget are allowed; topology augmentation is available to both arms.\n\n'
        +guide+'\n\nTASK\n'+json.dumps(public_task(task_id),indent=2)+'\n\nSELECTED ARM: '+arm+'\n')
