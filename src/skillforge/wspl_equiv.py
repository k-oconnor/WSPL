"""Scoped QF_LRA check that a serialized ReLU artifact equals its scalar IR source.

Encodes the scalar IR directly from its documented semantics, independently of
the compiler, and compares it with the stored artifact weights over the declared
input box. Trusted solver/encoders, not a proof-carrying certificate, IEEE
execution proof or physical guarantee. Differences are replayed through both
exact rational interpreters.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from .wspl_core import ScalarProgram,rational
from .wspl_artifact import ReLUArtifact,canonical_bytes

STATUS_EXIT={'EQUIVALENT':0,'DIFFERENT':2,'UNKNOWN':3,'INVALID':1}


def _port_signature(name,ty):
    return (name,ty['kind'],ty['unit'],rational(ty['lower']),rational(ty['upper']))


def _check_interface(source,artifact_document):
    expected_inputs=[_port_signature(k,v) for k,v in source['inputs'].items()]
    actual_inputs=[_port_signature(p['name'],p['type']) for p in artifact_document['inputs']]
    if expected_inputs!=actual_inputs:raise ValueError('artifact input ports differ from source declarations')
    expected_outputs=[_port_signature(o['name'],o['type']) for o in source['outputs']]
    actual_outputs=[_port_signature(p['name'],p['type']) for p in artifact_document['outputs']]
    if expected_outputs!=actual_outputs:raise ValueError('artifact output ports differ from source declarations')


def _encode_source(z3,source,inputs,constraints):
    """Encode validated scalar IR JSON as Z3 terms; mirrors ScalarProgram.evaluate."""
    def real(q):return z3.RealVal(str(F(q)))
    def clip(a,lo,hi):return z3.If(a<lo,lo,z3.If(a>hi,hi,a))
    env=dict(inputs)
    def term(expr):
        op=expr['op']
        if op=='ref':return env[expr['name']]
        if op=='const':return real(rational(expr['value']))
        if op=='affine':return sum((real(rational(t['coefficient']))*term(t['value']) for t in expr['terms']),real(rational(expr['bias'])))
        if op in ('relu','clip','margin','not'):
            a=term(expr['value'])
            if op=='relu':return z3.If(a>=0,a,real(0))
            if op=='not':return real(1)-a
            lo,hi=rational(expr['lower']),rational(expr['upper'])
            if op=='clip':return clip(a,real(lo),real(hi))
            return clip((a-real(lo))*real(1/(hi-lo)),real(0),real(1))
        a,b=term(expr['left']),term(expr['right'])
        if op=='min':return z3.If(a<=b,a,b)
        if op=='max':return z3.If(a>=b,a,b)
        if op=='and':return z3.If(a+b-1>=0,a+b-1,real(0))
        if op=='or':return z3.If(a+b<=1,a+b,real(1))
        raise ValueError('unsupported scalar operator after validation')
    outputs=[]
    for group in ('lets','outputs'):
        for declaration in source[group]:
            v=z3.Real('source_'+declaration['name'])
            constraints.append(v==term(declaration['value']));env[declaration['name']]=v
            if group=='outputs':outputs.append(v)
    return outputs


def _encode_artifact(z3,artifact_document,inputs,constraints):
    """Encode stored hidden-ReLU/final-linear layers as exact rational Z3 terms."""
    def real(q):return z3.RealVal(str(F(q)))
    previous=list(inputs)
    for layer_id,layer in enumerate(artifact_document['layers']):
        current=[]
        for i,(weights,bias) in enumerate(zip(layer['weights'],layer['bias'])):
            s=sum((real(w)*x for w,x in zip(weights,previous)),real(bias));v=z3.Real(f'artifact_{layer_id}_{i}')
            constraints.append(v==(z3.If(s>=0,s,real(0)) if layer['activation']=='relu' else s));current.append(v)
        previous=current
    return previous


def check_equivalence(source,artifact,*,timeout_ms=5000,max_parameters=10000,max_source_nodes=1024):
    started=time.monotonic();here=Path(__file__)
    result=dict(status='INVALID',scope='source_ir_vs_stored_weight_rational_real_graph',
                quantifier='forall inputs in declared domain: every source output equals the artifact output',
                numerical_execution='not_established',physical_trajectory='not_established',
                independently_checked_proof_trace=False,
                trust=['scalar IR validator','source encoder','artifact graph encoder','Z3','rational replay through both interpreters'],
                timeout_ms=timeout_ms,logic='QF_LRA',
                max_source_nodes=max_source_nodes,max_parameters=max_parameters,
                checker_source_sha256=hashlib.sha256(here.read_bytes()).hexdigest(),
                source_interpreter_sha256=hashlib.sha256(here.with_name('wspl_core.py').read_bytes()).hexdigest(),
                artifact_evaluator_sha256=hashlib.sha256(here.with_name('wspl_artifact.py').read_bytes()).hexdigest())
    def finish(status,**fields):
        result.update(status=status,**fields);result['wall_seconds']=time.monotonic()-started
        return result
    try:
        program=ScalarProgram(source,max_nodes=max_source_nodes)
        if not isinstance(artifact,ReLUArtifact):artifact=ReLUArtifact(artifact)
        d=artifact.to_dict()
        result['source_sha256']=hashlib.sha256(canonical_bytes(source)).hexdigest();result['artifact_sha256']=artifact.sha256
        if d['source_sha256']!=result['source_sha256']:raise ValueError('artifact is not bound to this source document')
        _check_interface(source,d)
        if type(timeout_ms) is not int or timeout_ms<0 or type(max_parameters) is not int or max_parameters<1:raise ValueError('invalid verification budget')
        if timeout_ms==0:return finish('UNKNOWN',reason='budget_exhausted_before_query')
        if d['allocation']['parameters']>max_parameters:return finish('UNKNOWN',reason='verification capacity limit')
        try:import z3
        except ImportError:return finish('UNKNOWN',reason='optional Z3 backend unavailable')
        result['backend']='Z3';result['backend_version']=z3.get_version_string()
        names=list(source['inputs']);xs=[z3.Real('input_'+str(i)) for i in range(len(names))]
        constraints=[]
        for x,(key,ty) in zip(xs,source['inputs'].items()):
            constraints.extend([x>=z3.RealVal(str(rational(ty['lower']))),x<=z3.RealVal(str(rational(ty['upper'])))])
        source_outputs=_encode_source(z3,source,dict(zip(names,xs)),constraints)
        artifact_outputs=_encode_artifact(z3,d,xs,constraints)
        difference=z3.Or(*(a!=b for a,b in zip(source_outputs,artifact_outputs)))
        solver=z3.SolverFor('QF_LRA')
        remaining=timeout_ms-int(1000*(time.monotonic()-started))
        if remaining<=0:return finish('UNKNOWN',reason='total query budget exhausted')
        solver.set(timeout=remaining);solver.add(*constraints,difference)
        result['difference_query_sha256']=hashlib.sha256(solver.sexpr().encode()).hexdigest()
        status=solver.check();result['difference_query']=str(status)
        if status==z3.unknown:return finish('UNKNOWN',reason=solver.reason_unknown())
        if status==z3.unsat:return finish('EQUIVALENT',reason='difference query UNSAT under declared rational real semantics')
        model=solver.model();values={}
        for key,x in zip(names,xs):
            v=model.eval(x,model_completion=True)
            if not z3.is_rational_value(v):return finish('UNKNOWN',reason='nonrational witness unsupported')
            values[key]=F(v.numerator_as_long(),v.denominator_as_long())
        expected=program.evaluate(values);actual=artifact.evaluate(values)
        differing=[k for k in artifact.output_names if expected[k]!=actual[k]]
        if not differing:return finish('UNKNOWN',reason='difference witness replay failed')
        return finish('DIFFERENT',reason='SAT difference independently replayed',witness_validated=True,
                      counterexample=dict(inputs={k:str(v) for k,v in values.items()},
                                          source_outputs={k:str(v) for k,v in expected.items()},
                                          artifact_outputs={k:str(v) for k,v in actual.items()},
                                          differing_outputs=differing))
    except (ValueError,TypeError,KeyError,ZeroDivisionError) as exc:return finish('INVALID',reason=str(exc))
    except Exception as exc:return finish('UNKNOWN',reason=f'backend/encoding failure: {type(exc).__name__}: {exc}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('artifact',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout-ms',type=int,default=5000)
    args=p.parse_args()
    try:
        result=check_equivalence(json.loads(args.source.read_text(encoding='utf-8')),ReLUArtifact.load(args.artifact),timeout_ms=args.timeout_ms)
        with args.output.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps(dict(status=result['status'],output=str(args.output))))
        raise SystemExit(STATUS_EXIT[result['status']])
    except (OSError,ValueError) as exc:print(str(exc),file=sys.stderr);raise SystemExit(1)


if __name__=='__main__':main()
