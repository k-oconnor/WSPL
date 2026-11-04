"""Explicit scalar affine/ReLU circuit placement, with no allocator or relays."""
import ast
from dataclasses import dataclass
from fractions import Fraction
import re

import numpy as np

from .wsp_artifact import fraction
from .wsp_syntax import expression
from .wsp_values import integer, scalar, numeric


@dataclass
class Form:
    bias: Fraction
    terms: dict

    def scale(self, q):
        return Form(self.bias*q, {k: v*q for k, v in self.terms.items() if v*q})

    def add(self, other):
        terms = dict(self.terms)
        for key, value in other.terms.items():
            terms[key] = terms.get(key, Fraction(0))+value
        return Form(self.bias+other.bias, {k: v for k, v in terms.items() if v})


def signal(node, graph, arithmetic, text, *, producer=False):
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        raise ValueError('signal must be a scalar x/hN/y index')
    name = node.value.id
    if name == 'x': level = 0
    elif name == 'y': level = len(graph['widths'])-1
    elif re.fullmatch(r'h[1-9]\d*', name): level = int(name[1:])
    else: raise ValueError('unknown signal name')
    if level >= len(graph['widths']) or (name.startswith('h') and level == len(graph['widths'])-1):
        raise ValueError('hidden signal level out of range')
    if producer and level == 0:
        raise ValueError('input is not a writable producer')
    index = integer(arithmetic.eval(node.slice, text))
    width = graph['widths'][level]
    if not -width <= index < width:
        raise ValueError('signal index out of bounds')
    return level, index % width


def affine(node, graph, arithmetic, text, env, *, allow_target=True):
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.Subscript):
        if not allow_target:
            raise ValueError('circuit bodies use declared ports/locals; bind target signals at place')
        return Form(Fraction(0), {signal(node, graph, arithmetic, text): Fraction(1)})
    if isinstance(node, ast.Constant) or (isinstance(node, ast.Name) and node.id in arithmetic.constants):
        q = scalar(numeric(arithmetic.eval(node, text)))
        if not isinstance(q, Fraction):
            raise ValueError('signal coefficients must be rational values')
        return Form(q, {})
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        result = affine(node.operand, graph, arithmetic, text, env, allow_target=allow_target)
        return result.scale(Fraction(-1)) if isinstance(node.op, ast.USub) else result
    if isinstance(node, ast.BinOp):
        a = affine(node.left, graph, arithmetic, text, env, allow_target=allow_target)
        b = affine(node.right, graph, arithmetic, text, env, allow_target=allow_target)
        if isinstance(node.op, ast.Add): return a.add(b)
        if isinstance(node.op, ast.Sub): return a.add(b.scale(Fraction(-1)))
        if isinstance(node.op, ast.Mult):
            if not a.terms: return b.scale(a.bias)
            if not b.terms: return a.scale(b.bias)
        if isinstance(node.op, ast.Div) and not b.terms and b.bias:
            return a.scale(1/b.bias)
    raise ValueError('placement requires affine expressions or one named outer relu; expand clip/min/max into explicit gates')


def place(row, circuit, graph, arithmetic, write, reserved):
    if graph['kind'] != 'dense':
        raise ValueError('UNSUPPORTED: placement requires a dense graph adapter')
    call = expression('f('+row['arguments']+')').body
    if call.args or any(k.arg is None for k in call.keywords) or len({k.arg for k in call.keywords}) != len(call.keywords):
        raise ValueError('unique named circuit input bindings required')
    if {k.arg for k in call.keywords} != set(circuit['inputs']):
        raise ValueError('circuit input names differ')
    assigned = {a['name'] for a in circuit['assignments']}
    if set(row['placements']) != assigned:
        raise ValueError('place every assigned circuit node/output exactly once')
    locations = {k: signal(expression(v).body, graph, arithmetic, v, producer=True) for k, v in row['placements'].items()}
    if len(set(locations.values())) != len(locations) or set(locations.values()) & reserved:
        raise ValueError('two placements cannot overwrite the same producer')
    env = {k.arg: affine(k.value, graph, arithmetic, 'f('+row['arguments']+')', {}) for k in call.keywords}
    rows = []; refinement_obligations = []
    for name, ty in list(circuit['inputs'].items())+list(circuit['outputs'].items()):
        if ty['kind'] == 'Truth' or 'lower' in ty:
            if ty['kind'] == 'Truth': lo, hi = Fraction(0), Fraction(1)
            else: lo, hi = scalar(arithmetic.run(ty['lower'])), scalar(arithmetic.run(ty['upper']))
            if not isinstance(lo, Fraction) or not isinstance(hi, Fraction) or lo > hi:
                raise ValueError('invalid circuit refinement interval')
            refinement_obligations.append(dict(name=name, lower=str(lo), upper=str(hi), status='UNVERIFIED', used_for_lowering=False))
    for assignment in circuit['assignments']:
        if assignment['type'] is not None and assignment['type'] != {'kind': 'Real'}:
            raise ValueError('UNSUPPORTED: intermediate refinements; use a separate claim')
        name, text = assignment['name'], assignment['expression']
        node = expression(text).body
        activation = 'linear'
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'relu' and len(node.args) == 1 and not node.keywords:
            activation = 'relu'; node = node.args[0]
        form = affine(node, graph, arithmetic, text, env, allow_target=False)
        level, index = locations[name]
        if graph['activations'][level-1] != activation:
            raise ValueError(f'INVALID_PLACEMENT: {name} needs {activation}, destination uses {graph["activations"][level-1]}')
        missing = [list(k) for k in form.terms if k[0] != level-1]
        if missing:
            raise ValueError(f'INVALID_PLACEMENT: {name} requires unavailable edges {missing}; no implicit relays')
        weights = np.full((graph['widths'][level-1],), Fraction(0), dtype=object)
        for (_, source), coefficient in form.terms.items():
            weights[source] = coefficient
        rows.append((name, level, index, weights, form.bias))
        env[name] = Form(Fraction(0), {(level, index): Fraction(1)})
    outside = []
    for _, level, index, _, _ in rows:
        if level < len(graph['activations']):
            w = arithmetic.arrays[f'W{level+1}']
            values = w[index, :] if graph['layout'] == 'in_out' else w[:, index]
            for consumer, bits in enumerate(values):
                if fraction(bits) != 0 and (level+1, consumer) not in locations.values():
                    outside.append(dict(source=[level, index], consumer=[level+1, consumer], edge_preserved=True))
    for _, level, index, weights, bias in rows:
        selection = (slice(None), index) if graph['layout'] == 'in_out' else (index, slice(None))
        write(f'W{level}', selection, weights, '=', row['span'], f'place {row["name"]}')
        write(f'b{level}', (index,), np.array(bias, dtype=object), '=', row['span'], f'place {row["name"]}')
    reserved.update(locations.values())
    return dict(circuit=row['name'], placements={k:list(v) for k, v in locations.items()}, external_consumers=outside,
                refinement_obligations=refinement_obligations, source_equivalence='not_checked',
                later_edits_may_invalidate=False)
