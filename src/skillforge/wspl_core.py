"""Experimental scalar WSPL IR and exact rational reference interpreter.

Not the textual WSPL parser, compiler, solver, deployment runtime or scientific
authoring harness. Interval typing is sound/conservative for this scalar subset;
it is not a universal contract verifier. No Python proposal code is executed.
"""
import copy
from dataclasses import dataclass
from fractions import Fraction
import re


def rational(value):
    if type(value) is int or isinstance(value, Fraction): return Fraction(value)
    if not isinstance(value, str) or not re.fullmatch(r'[+-]?(?:\d+/\d+|\d+(?:\.\d*)?|\.\d+)', value):
        raise ValueError('exact integer, decimal string or fraction string required; floats are not implicit exact constants')
    try: return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc: raise ValueError('invalid rational') from exc


def name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', value): raise ValueError('invalid name')
    return value


@dataclass(frozen=True)
class ScalarType:
    kind: str
    unit: str
    lower: Fraction
    upper: Fraction

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) != {'kind', 'unit', 'lower', 'upper'}:
            raise ValueError('explicit scalar kind/unit/bounds required')
        lo, hi = rational(data['lower']), rational(data['upper'])
        if lo > hi or data['kind'] not in ('Real', 'Truth') or not isinstance(data['unit'], str) or not data['unit']:
            raise ValueError('invalid scalar type')
        if data['kind'] == 'Truth' and (data['unit'] != '1' or lo < 0 or hi > 1):
            raise ValueError('Truth must be dimensionless and bounded in [0,1]')
        return cls(data['kind'], data['unit'], lo, hi)


class ScalarProgram:
    def __init__(self, document):
        d = copy.deepcopy(document)
        if not isinstance(d, dict) or set(d) != {'format', 'inputs', 'lets', 'outputs'} or d['format'] != 'wspl.scalar_ir.v0':
            raise ValueError('unsupported scalar IR document')
        if not isinstance(d['inputs'], dict) or not d['inputs'] or not isinstance(d['lets'], list) or not isinstance(d['outputs'], list) or not d['outputs']:
            raise ValueError('explicit inputs, lets and outputs required')
        self._inputs = {name(k): ScalarType.parse(v) for k, v in d['inputs'].items()}
        environment = dict(self._inputs)
        nodes = []
        count = 0
        def lower(expr, depth=0):
            nonlocal count
            count += 1
            if count > 1024 or depth > 32 or not isinstance(expr, dict): raise ValueError('invalid/oversized expression')
            op = expr.get('op')
            if op == 'ref':
                if set(expr) != {'op', 'name'} or expr['name'] not in environment: raise ValueError('unknown/forward reference')
                return ('ref', expr['name']), environment[expr['name']]
            if op == 'const':
                if set(expr) != {'op', 'value', 'unit'} or not isinstance(expr['unit'], str) or not expr['unit']:
                    raise ValueError('constant needs explicit value and unit')
                v = rational(expr['value'])
                return ('const', v), ScalarType('Real', expr['unit'], v, v)
            if op == 'affine':
                if set(expr) != {'op', 'terms', 'bias', 'unit'} or not isinstance(expr['terms'], list) or not expr['terms']:
                    raise ValueError('invalid affine expression')
                bias = rational(expr['bias']); lo = hi = bias; terms = []
                for term in expr['terms']:
                    if not isinstance(term, dict) or set(term) != {'coefficient', 'value'}: raise ValueError('invalid affine term')
                    coefficient = rational(term['coefficient']); tree, ty = lower(term['value'], depth+1)
                    if ty.unit != expr['unit']: raise ValueError('affine terms require identical units; normalize explicitly')
                    a, b = coefficient*ty.lower, coefficient*ty.upper
                    lo += min(a, b); hi += max(a, b); terms.append((coefficient, tree))
                if not isinstance(expr['unit'], str) or not expr['unit']: raise ValueError('invalid affine unit')
                return ('affine', tuple(terms), bias), ScalarType('Real', expr['unit'], lo, hi)
            if op in ('relu', 'clip', 'margin', 'not'):
                expected = {'op', 'value'} | ({'lower', 'upper'} if op in ('clip', 'margin') else set())
                if set(expr) != expected: raise ValueError('unexpected unary fields')
                tree, ty = lower(expr['value'], depth+1)
                if op == 'relu': return ('relu', tree), ScalarType('Real', ty.unit, max(0,ty.lower), max(0,ty.upper))
                if op == 'not':
                    if ty.kind != 'Truth': raise ValueError('graded not requires Truth, not arbitrary Real/Bool')
                    return ('not', tree), ScalarType('Truth','1',1-ty.upper,1-ty.lower)
                a, b = rational(expr['lower']), rational(expr['upper'])
                if a > b or (op == 'margin' and a == b): raise ValueError('invalid clipping/normalization interval')
                def clip(v): return min(b,max(a,v))
                if op == 'clip': return ('clip',tree,a,b), ScalarType('Real',ty.unit,clip(ty.lower),clip(ty.upper))
                def margin(v): return min(Fraction(1),max(Fraction(0),(v-a)/(b-a)))
                return ('margin',tree,a,b), ScalarType('Truth','1',margin(ty.lower),margin(ty.upper))
            if op in ('min', 'max', 'and', 'or'):
                if set(expr) != {'op','left','right'}: raise ValueError('invalid binary expression')
                left, a = lower(expr['left'],depth+1); right, b = lower(expr['right'],depth+1)
                if a.unit != b.unit: raise ValueError('incompatible units')
                if op in ('and','or'):
                    if a.kind != 'Truth' or b.kind != 'Truth': raise ValueError('graded connectives require explicit Truth operands')
                    fn = (lambda x,y:max(Fraction(0),x+y-1)) if op=='and' else (lambda x,y:min(Fraction(1),x+y))
                    ty=ScalarType('Truth','1',fn(a.lower,b.lower),fn(a.upper,b.upper))
                else:
                    fn=min if op=='min' else max
                    ty=ScalarType('Real',a.unit,fn(a.lower,b.lower),fn(a.upper,b.upper))
                return (op,left,right),ty
            raise ValueError('unsupported scalar operator/profile')
        self._outputs=[]
        for group in ('lets','outputs'):
            for declaration in d[group]:
                if not isinstance(declaration,dict) or set(declaration)!={'name','type','value'}: raise ValueError('invalid declaration')
                key=name(declaration['name'])
                if key in environment: raise ValueError('duplicate declaration')
                declared=ScalarType.parse(declaration['type']); tree,inferred=lower(declaration['value'])
                if declared.unit!=inferred.unit: raise ValueError('declared unit differs from expression')
                # Real[0,1] -> Truth must be explicit via margin or a declared
                # Truth result with an independently inferred [0,1] range.
                if declared.kind=='Truth' and (inferred.lower<0 or inferred.upper>1): raise ValueError('unproved Truth range')
                if inferred.lower<declared.lower or inferred.upper>declared.upper:
                    raise ValueError('range obligation unresolved by conservative interval typing')
                environment[key]=declared;nodes.append((key,tree))
                if group=='outputs':self._outputs.append(key)
        self._nodes=tuple(nodes);self._outputs=tuple(self._outputs)

    def evaluate(self, inputs):
        if not isinstance(inputs,dict) or set(inputs)!=set(self._inputs): raise ValueError('input record differs')
        env={}
        for key,ty in self._inputs.items():
            v=rational(inputs[key])
            if not ty.lower<=v<=ty.upper: raise ValueError('input outside declared domain: '+key)
            env[key]=v
        def run(node):
            op=node[0]
            if op=='ref':return env[node[1]]
            if op=='const':return node[1]
            if op=='affine':return sum((c*run(v) for c,v in node[1]),node[2])
            a=run(node[1])
            if op=='relu':return max(Fraction(0),a)
            if op=='not':return 1-a
            if op=='clip':return min(node[3],max(node[2],a))
            if op=='margin':return min(Fraction(1),max(Fraction(0),(a-node[2])/(node[3]-node[2])))
            b=run(node[2])
            if op=='min':return min(a,b)
            if op=='max':return max(a,b)
            if op=='and':return max(Fraction(0),a+b-1)
            if op=='or':return min(Fraction(1),a+b)
            raise ValueError('unsupported internal node')
        for key,tree in self._nodes:env[key]=run(tree)
        return {key:env[key] for key in self._outputs}
