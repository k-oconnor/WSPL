"""Scoped QF_LRA checks of serialized ReLU graph I/O contracts.

Trusted solver/encoding, not independently proof-carrying or IEEE/robot proof.
Counterexamples are replayed with independent rational artifact evaluation.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from .wspl_artifact import ReLUArtifact,canonical_bytes


def _q(x):
    if type(x) is int or isinstance(x,F):return F(x)
    if not isinstance(x,str):raise ValueError('exact rational contract constants required')
    try:return F(x)
    except (ValueError,ZeroDivisionError) as exc:raise ValueError('invalid contract constant') from exc


def _parse_formula(document,units):
    count=0
    def linear(d):
        if not isinstance(d,dict) or set(d)!={'terms','bias','unit'} or not isinstance(d['terms'],dict) or not isinstance(d['unit'],str):
            raise ValueError('explicit affine form and unit required')
        terms=[]
        for key,coefficient in d['terms'].items():
            c=_q(coefficient)
            if key not in units:raise ValueError('unknown numeric port: '+str(key))
            if units[key]!=d['unit']:raise ValueError('contract units differ; normalize explicitly')
            terms.append((key,c))
        if not d['unit']:raise ValueError('empty contract unit')
        return (tuple(terms),_q(d['bias']),d['unit'])
    def visit(d,depth=0):
        nonlocal count
        count+=1
        if count>256 or depth>32 or not isinstance(d,dict):raise ValueError('invalid/oversized contract formula')
        op=d.get('op')
        if op in ('true','false'):
            if set(d)!={'op'}:raise ValueError('invalid Boolean constant')
            return (op,)
        if op in ('le','lt','ge','gt','eq'):
            if set(d)!={'op','left','right'}:raise ValueError('invalid comparison')
            a,b=linear(d['left']),linear(d['right'])
            if a[2]!=b[2]:raise ValueError('comparison units differ')
            return (op,a,b)
        if op not in ('and','or','not','implies') or set(d)!={'op','args'} or not isinstance(d['args'],list):
            raise ValueError('unsupported Boolean contract operator')
        n=len(d['args'])
        if (op in ('and','or') and n<2) or (op=='not' and n!=1) or (op=='implies' and n!=2):raise ValueError('invalid Boolean arity')
        return (op,*(visit(x,depth+1) for x in d['args']))
    return visit(document)


def _evaluate(formula,ports):
    def value(expr):return sum((c*ports[k] for k,c in expr[0]),expr[1])
    op=formula[0]
    if op=='true':return True
    if op=='false':return False
    if op in ('le','lt','ge','gt','eq'):
        a,b=value(formula[1]),value(formula[2])
        return {'le':a<=b,'lt':a<b,'ge':a>=b,'gt':a>b,'eq':a==b}[op]
    args=[_evaluate(x,ports) for x in formula[1:]]
    if op=='and':return all(args)
    if op=='or':return any(args)
    if op=='not':return not args[0]
    if op=='implies':return not args[0] or args[1]
    raise ValueError('invalid replay operator')


def verify_contract(artifact,contract,*,timeout_ms=5000,max_parameters=10000):
    started=time.monotonic()
    result=dict(status='INVALID',scope='stored_weight_rational_real_graph',
                quantifier='forall inputs in artifact domain: assumptions imply ensures',
                numerical_execution='not_established',source_equivalence='not_checked',
                physical_trajectory='not_established',independently_checked_proof_trace=False,
                trust=['contract parser','real-graph encoder','Z3','rational witness replay'],timeout_ms=timeout_ms,
                logic='QF_LRA',verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                evaluator_source_sha256=hashlib.sha256(Path(__file__).with_name('wspl_artifact.py').read_bytes()).hexdigest())
    def finish(status,**fields):
        result.update(status=status,**fields);result['wall_seconds']=time.monotonic()-started
        return result
    try:
        if not isinstance(artifact,ReLUArtifact):artifact=ReLUArtifact(artifact)
        d=artifact.to_dict();result['artifact_sha256']=artifact.sha256
        if (not isinstance(contract,dict) or set(contract)!={'format','name','artifact_sha256','semantics','assumptions','ensures'} or
            contract['format']!='wspl.io_contract.v0' or not isinstance(contract['name'],str) or not contract['name']):
            raise ValueError('invalid I/O contract schema')
        result['name']=contract['name'];result['contract_sha256']=hashlib.sha256(canonical_bytes(contract)).hexdigest()
        if contract['artifact_sha256']!=artifact.sha256:raise ValueError('contract artifact identity differs')
        if contract['semantics']!='rational_real_graph':return finish('UNKNOWN',reason='unsupported numerical semantics')
        units={prefix+'.'+x['name']:x['type']['unit'] for prefix,key in [('input','inputs'),('output','outputs')] for x in d[key]}
        assumption=_parse_formula(contract['assumptions'],units);conclusion=_parse_formula(contract['ensures'],units)
        if type(timeout_ms) is not int or timeout_ms<0 or type(max_parameters) is not int or max_parameters<1:raise ValueError('invalid verification budget')
        if timeout_ms==0:return finish('UNKNOWN',reason='budget_exhausted_before_query')
        if d['allocation']['parameters']>max_parameters:return finish('UNKNOWN',reason='verification capacity limit')
        try:import z3
        except ImportError:return finish('UNKNOWN',reason='optional Z3 backend unavailable')
        result['backend']='Z3';result['backend_version']=z3.get_version_string()
        def real(q):return z3.RealVal(str(_q(q)))
        ports={};base=[];previous=[]
        for i,port in enumerate(d['inputs']):
            x=z3.Real('input_'+str(i));ports['input.'+port['name']]=x;previous.append(x)
            base.extend([x>=real(port['type']['lower']),x<=real(port['type']['upper'])])
        for layer_id,layer in enumerate(d['layers']):
            current=[]
            for i,(weights,bias) in enumerate(zip(layer['weights'],layer['bias'])):
                s=sum((real(w)*x for w,x in zip(weights,previous)),real(bias));v=z3.Real(f'layer_{layer_id}_{i}')
                base.append(v==(z3.If(s>=0,s,real(0)) if layer['activation']=='relu' else s));current.append(v)
            previous=current
        for port,value in zip(d['outputs'],previous):ports['output.'+port['name']]=value
        def encode(formula):
            def linear(expr):return sum((real(c)*ports[k] for k,c in expr[0]),real(expr[1]))
            op=formula[0]
            if op in ('true','false'):return z3.BoolVal(op=='true')
            if op in ('le','lt','ge','gt','eq'):
                a,b=linear(formula[1]),linear(formula[2])
                return {'le':a<=b,'lt':a<b,'ge':a>=b,'gt':a>b,'eq':a==b}[op]
            args=[encode(x) for x in formula[1:]]
            if op=='and':return z3.And(*args)
            if op=='or':return z3.Or(*args)
            if op=='not':return z3.Not(args[0])
            return z3.Implies(*args)
        def query(extra):
            remaining=timeout_ms-int(1000*(time.monotonic()-started))
            if remaining<=0:return None,'unknown','total query budget exhausted'
            solver=z3.SolverFor('QF_LRA');solver.set(timeout=remaining);solver.add(*base,*extra)
            status=solver.check()
            return solver,str(status),solver.reason_unknown() if status==z3.unknown else None
        solver,domain,reason=query([encode(assumption)])
        result['domain_query']=domain
        if solver is not None:result['domain_query_sha256']=hashlib.sha256(solver.sexpr().encode()).hexdigest()
        if domain=='unknown':return finish('UNKNOWN',reason=reason)
        if domain=='unsat':return finish('INVALID',reason='vacuous assumption domain',nonvacuous=False)
        def witness(solver):
            model=solver.model();values={}
            for port in d['inputs']:
                v=model.eval(ports['input.'+port['name']],model_completion=True)
                if not z3.is_rational_value(v):raise ValueError('nonrational witness unsupported')
                values[port['name']]=F(v.numerator_as_long(),v.denominator_as_long())
            outputs=artifact.evaluate(values)
            replay={**{'input.'+k:v for k,v in values.items()},**{'output.'+k:v for k,v in outputs.items()}}
            return values,outputs,replay
        domain_inputs,_,replay=witness(solver)
        if not _evaluate(assumption,replay):return finish('UNKNOWN',reason='nonvacuity witness replay failed')
        result.update(nonvacuous=True,domain_witness={k:str(v) for k,v in domain_inputs.items()})
        solver,violation,reason=query([encode(assumption),z3.Not(encode(conclusion))])
        result['violation_query']=violation
        if solver is not None:result['violation_query_sha256']=hashlib.sha256(solver.sexpr().encode()).hexdigest()
        if violation=='unknown':return finish('UNKNOWN',reason=reason)
        if violation=='unsat':return finish('PROVED',reason='violation query UNSAT under declared real-graph semantics')
        values,outputs,replay=witness(solver)
        if not _evaluate(assumption,replay) or _evaluate(conclusion,replay):return finish('UNKNOWN',reason='violating witness replay failed')
        return finish('REFUTED',reason='SAT violation independently replayed',witness_validated=True,
                      counterexample=dict(inputs={k:str(v) for k,v in values.items()},outputs={k:str(v) for k,v in outputs.items()}))
    except (ValueError,TypeError,KeyError,ZeroDivisionError) as exc:return finish('INVALID',reason=str(exc))
    except Exception as exc:return finish('UNKNOWN',reason=f'backend/encoding failure: {type(exc).__name__}: {exc}')


def verify_suite(artifact,suite,**kwargs):
    if not isinstance(suite,dict) or set(suite)!={'format','artifact_sha256','contracts'} or suite['format']!='wspl.io_contract_suite.v0' or not isinstance(suite['contracts'],list) or not suite['contracts']:
        return dict(status='INVALID',reason='invalid contract suite',results=[])
    results=[]
    for c in suite['contracts']:
        if not isinstance(c,dict) or set(c)!={'name','assumptions','ensures'}:return dict(status='INVALID',reason='invalid suite member',results=results)
        results.append(verify_contract(artifact,dict(format='wspl.io_contract.v0',semantics='rational_real_graph',artifact_sha256=suite['artifact_sha256'],**c),**kwargs))
    status=next((s for s in ['INVALID','REFUTED','UNKNOWN'] if any(r['status']==s for r in results)),'PROVED')
    return dict(status=status,scope='stored_weight_rational_real_graph',results=results)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('artifact',type=Path);p.add_argument('contracts',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout-ms',type=int,default=5000)
    args=p.parse_args()
    try:
        result=verify_suite(ReLUArtifact.load(args.artifact),json.loads(args.contracts.read_text(encoding='utf-8')),timeout_ms=args.timeout_ms)
        with args.output.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps(dict(status=result['status'],output=str(args.output))))
        raise SystemExit({'PROVED':0,'REFUTED':2,'UNKNOWN':3,'INVALID':1}[result['status']])
    except (OSError,ValueError) as exc:print(str(exc),file=sys.stderr);raise SystemExit(1)


if __name__=='__main__':main()
