"""Exact, bounded construction arithmetic; AST nodes are interpreted as data."""
import ast
from dataclasses import dataclass, asdict
from fractions import Fraction
import functools
import math
import re

import numpy as np

from .wsp_artifact import Bits, fraction, digest
from .wsp_syntax import expression


@dataclass(frozen=True)
class Limits:
    max_elements: int = 1_000_000
    max_operations: int = 4_000_000
    max_statements: int = 10_000
    max_loop_iterations: int = 10_000
    max_integer_bits: int = 4096

    def __post_init__(self):
        if any(type(v) is not int or not 1 <= v <= 100_000_000 for v in asdict(self).values()):
            raise ValueError('positive bounded resource limits required')


BUILTINS = {'copy', 'transpose', 'reshape', 'zeros', 'full', 'eye', 'f32bits', 'tensor'}


def scalar(value):
    a = np.asarray(value, dtype=object)
    if a.shape:
        raise ValueError('scalar value required')
    return a.item()


def integer(value):
    q = scalar(value)
    if isinstance(q, Bits):
        q = fraction(q.value)
    if not isinstance(q, Fraction) or q.denominator != 1:
        raise ValueError('integer required')
    return q.numerator


def numeric(value):
    a = np.asarray(value, dtype=object)
    return np.array([fraction(x.value) if isinstance(x, Bits) else x for x in a.flat], dtype=object).reshape(a.shape)


def pack(value):
    a = np.asarray(value, dtype=object)
    return dict(shape=list(a.shape), values=[['bits', f'{x.value:08x}'] if isinstance(x, Bits) else ['q', str(x)] for x in a.flat])


def unpack(document, limits=Limits()):
    if not isinstance(document, dict) or set(document) != {'shape', 'values'}:
        raise ValueError('invalid construction binding')
    shape = document['shape']
    if not isinstance(shape, list) or len(shape) > 8 or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError('invalid binding shape')
    if math.prod(shape) > limits.max_elements or len(document['values']) != math.prod(shape):
        raise ValueError('binding element limit/shape mismatch')
    values = []
    for row in document['values']:
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[1], str) or len(row[1]) > 2500:
            raise ValueError('invalid binding value')
        if row[0] == 'bits':
            if not re.fullmatch('[0-9a-f]{8}', row[1]):
                raise ValueError('invalid exact bits')
            value = Bits(int(row[1], 16))
        elif row[0] == 'q':
            value = Fraction(row[1])
            if max(value.numerator.bit_length(), value.denominator.bit_length()) > limits.max_integer_bits:
                raise ValueError('binding precision limit')
        else:
            raise ValueError('unknown binding type')
        values.append(value)
    return np.array(values, dtype=object).reshape(shape)


@functools.lru_cache(maxsize=65536)
def _literal(text, kind, lineno, col_offset, end_lineno, end_col_offset):
    """Exact value of one numeric literal; cached because loop bodies repeat it.

    Depends only on the source text and the node's position and value type.
    Errors are not cached, so they are raised again on every evaluation.
    """
    node = ast.Constant(value=None, lineno=lineno, col_offset=col_offset, end_lineno=end_lineno, end_col_offset=end_col_offset)
    raw = ast.get_source_segment(text, node)
    if kind not in (int, float) or raw is None or not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', raw):
        raise ValueError('decimal/scientific numeric literal required')
    # Bound exponent before Fraction can allocate a gigantic power of ten.
    exponent = re.search(r'[eE]([+-]?\d+)', raw)
    if len(raw) > 1500 or (exponent and (len(exponent[1]) > 5 or abs(int(exponent[1])) > 1200)):
        raise ValueError('numeric literal precision limit')
    return Fraction(raw)


def _exact(item):
    return fraction(item.value) if isinstance(item, Bits) else item


class Arithmetic:
    def __init__(self, arrays, parent, constants, bindings, limits=Limits()):
        self.arrays, self.parent, self.constants, self.bindings = arrays, parent, constants, bindings
        self.limits = limits
        self.operations = 0

    def charge(self, work=1):
        self.operations += work
        if self.operations > self.limits.max_operations:
            raise ValueError('construction operation budget exhausted')

    def bounded(self, value):
        a = np.asarray(value, dtype=object)
        if a.size > self.limits.max_elements or a.ndim > 8:
            raise ValueError('construction tensor size/rank limit')
        for q in a.flat:
            if isinstance(q, Bits):
                continue
            if not isinstance(q, Fraction):
                raise ValueError('construction values must be rationals or exact bits')
            if max(q.numerator.bit_length(), q.denominator.bit_length()) > self.limits.max_integer_bits:
                raise ValueError('construction rational precision limit')
        return a

    def run(self, text):
        return self.eval(expression(text).body, text)

    def literal(self, node, text):
        value = _literal(text, type(node.value), node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
        return self.bounded(np.array(value, dtype=object))

    def selection(self, node, shape, text):
        entries = node.elts if isinstance(node, ast.Tuple) else [node]
        if len(entries) != len(shape):
            raise ValueError('explicit tensor rank required')
        result = []; clipped = False
        for entry, width in zip(entries, shape):
            if isinstance(entry, ast.Slice):
                lo, hi, step = [None if x is None else integer(self.eval(x, text)) for x in (entry.lower, entry.upper, entry.step)]
                if step == 0:
                    raise ValueError('zero slice stride')
                clipped |= any(x is not None and (x < -width or x > width) for x in (lo, hi))
                result.append(slice(lo, hi, step))
            else:
                index = integer(self.eval(entry, text))
                if not -width <= index < width:
                    raise ValueError('tensor index out of bounds')
                result.append(index)
        return tuple(result), clipped

    def read_tensor(self, name, parent=False):
        arrays = self.parent if parent else self.arrays
        if name not in arrays:
            raise ValueError(f'unknown tensor {name}')
        a = arrays[name]
        self.charge(a.size)
        return np.array([Bits(int(x)) for x in a.flat], dtype=object).reshape(a.shape)

    def eval(self, node, text):
        self.charge()
        if isinstance(node, ast.Constant):
            return self.literal(node, text)
        if isinstance(node, ast.Name):
            if node.id in self.constants:
                return self.constants[node.id].copy()
            return self.read_tensor(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'parent':
            return self.read_tensor(node.attr, True)
        if isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > self.limits.max_elements:
                raise ValueError('list size limit')
            items = [self.eval(x, text) for x in node.elts]
            if sum(x.size for x in items) > self.limits.max_elements:
                raise ValueError('list element limit')
            try:
                return self.bounded(np.stack(items) if items else np.empty((0,), dtype=object))
            except (ValueError, TypeError) as exc:
                raise ValueError('rectangular numeric array required') from exc
        if isinstance(node, ast.Subscript):
            # Preserve statement snapshots while boxing only the requested slice.
            name=None;parent=False
            if isinstance(node.value,ast.Name) and node.value.id in self.arrays and node.value.id not in self.constants:
                name=node.value.id
            elif isinstance(node.value,ast.Attribute) and isinstance(node.value.value,ast.Name) and node.value.value.id=='parent':
                name=node.value.attr;parent=True
            if name is not None:
                bank=self.parent if parent else self.arrays
                if name not in bank:raise ValueError('unknown tensor '+name)
                selection,_=self.selection(node.slice,bank[name].shape,text)
                a=np.asarray(bank[name][selection]);self.charge(a.size+1)
                return np.array([Bits(int(x)) for x in a.flat],dtype=object).reshape(a.shape)
            a = self.eval(node.value, text)
            selection, _ = self.selection(node.slice, a.shape, text)
            return np.asarray(a[selection], dtype=object).copy()
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            a = self.eval(node.operand, text)
            if a.ndim == 0:  # Scalar index/coefficient arithmetic; same result as numeric().
                x = _exact(a.item())
                return self.bounded(np.array(x if isinstance(node.op, ast.UAdd) else -x, dtype=object))
            a = numeric(a)
            return self.bounded(a if isinstance(node.op, ast.UAdd) else -a)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.MatMult)):
            a, b = self.eval(node.left, text), self.eval(node.right, text)
            if a.ndim == 0 and b.ndim == 0 and not isinstance(node.op, ast.MatMult):
                # Scalar fast path: identical charge (broadcast size 1) and exact result.
                x, y = _exact(a.item()), _exact(b.item())
                self.charge(1)
                if 1 > self.limits.max_elements:
                    raise ValueError('arithmetic result exceeds element limit')
                op = node.op
                out = x+y if isinstance(op, ast.Add) else x-y if isinstance(op, ast.Sub) else x*y if isinstance(op, ast.Mult) else x/y
                return self.bounded(np.array(out, dtype=object))
            a, b = numeric(a), numeric(b)
            if isinstance(node.op, ast.MatMult):
                if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
                    raise ValueError('matrix multiply needs compatible rank-two tensors')
                size = a.shape[0]*b.shape[1]
                self.charge(size*max(1, a.shape[1]))
            else:
                size = math.prod(np.broadcast_shapes(a.shape, b.shape))
                self.charge(size)
            if size > self.limits.max_elements:
                raise ValueError('arithmetic result exceeds element limit')
            if isinstance(node.op, ast.Add): out = a+b
            elif isinstance(node.op, ast.Sub): out = a-b
            elif isinstance(node.op, ast.Mult): out = a*b
            elif isinstance(node.op, ast.Div): out = a/b
            else: out = a@b
            return self.bounded(out)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BUILTINS:
            name = node.func.id
            if name == 'f32bits':
                if node.keywords or len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or type(node.args[0].value) is not int:
                    raise ValueError('f32bits needs one integer literal')
                return np.array(Bits(node.args[0].value), dtype=object)
            if name == 'tensor':
                if (len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str)
                        or len(node.keywords) != 1 or node.keywords[0].arg != 'sha256'
                        or not isinstance(node.keywords[0].value, ast.Constant)):
                    raise ValueError('tensor needs pinned name and sha256')
                key = node.args[0].value
                if key not in self.bindings or digest(pack(self.bindings[key])) != node.keywords[0].value.value:
                    raise ValueError('construction binding identity differs')
                return self.bindings[key].copy()
            if node.keywords:
                raise ValueError('unexpected construction keyword')
            args = [self.eval(a, text) for a in node.args]
            if name in ('copy', 'transpose'):
                if len(args) != 1 or (name == 'transpose' and args[0].ndim != 2):
                    raise ValueError('copy/transpose arity or rank differs')
                return args[0].copy() if name == 'copy' else args[0].T.copy()
            if name == 'eye':
                if len(args) != 1:
                    raise ValueError('eye needs one width')
                n = integer(args[0]); shape = (n, n)
            else:
                required = 2 if name in ('reshape', 'full') else 1
                if len(args) != required:
                    raise ValueError('constructor arity differs')
                dims = args[1] if name == 'reshape' else args[0]
                if dims.ndim > 1:
                    raise ValueError('shape must be a scalar or vector')
                shape = tuple(integer(x) for x in dims.flat)
            if not 1 <= len(shape) <= 8 or any(n < 0 for n in shape) or math.prod(shape) > self.limits.max_elements:
                raise ValueError('constructor dimensions exceed limit')
            self.charge(math.prod(shape))
            if name == 'reshape':
                return args[0].reshape(shape).copy()
            if name == 'eye':
                out = np.full(shape, Fraction(0), dtype=object)
                np.fill_diagonal(out, Fraction(1)); return out
            fill = scalar(args[1]) if name == 'full' else Fraction(0)
            return np.full(shape, fill, dtype=object)
        raise ValueError(f'unsupported construction expression: {type(node).__name__}')
