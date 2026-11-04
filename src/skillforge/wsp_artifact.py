"""Immutable FP32 tensor artifacts for the WSP editor (no source compilation).

This is deliberately separate from the historical rational ReLU artifact format.
NumPy is an optional dependency required by this editor, not by the scalar core.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import re
import struct

import numpy as np


FORMAT = 'wsp.tensor_artifact.v1'
NAME = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def fraction(bits):
    return Fraction(struct.unpack('<f', struct.pack('<I', int(bits)))[0])


def round_f32(value):
    """Round a rational directly to binary32, nearest/even, with no float64 hop."""
    q = Fraction(value)
    sign = 0x80000000 if q < 0 else 0
    n, d = abs(q.numerator), q.denominator
    if n == 0:
        return sign
    exponent = n.bit_length() - d.bit_length()
    below_power = n < (d << exponent) if exponent >= 0 else (n << -exponent) < d
    if below_power:
        exponent -= 1
    shift = 149 if exponent < -126 else 23 - exponent
    a, b = (n << shift, d) if shift >= 0 else (n, d << -shift)
    mantissa, remainder = divmod(a, b)
    if 2 * remainder > b or (2 * remainder == b and mantissa % 2):
        mantissa += 1
    if exponent < -126:
        return sign | mantissa  # Includes rounded zero and the minimum normal.
    if mantissa == 1 << 24:
        exponent += 1
        mantissa >>= 1
    if exponent > 127:
        raise ValueError('FP32 overflow; nonfinite storage is unsupported')
    return sign | ((exponent + 127) << 23) | (mantissa - (1 << 23))


@dataclass(frozen=True)
class Bits:
    value: int

    def __post_init__(self):
        if type(self.value) is not int or not 0 <= self.value < 2**32:
            raise ValueError('f32bits requires an unsigned 32-bit integer')
        if (self.value & 0x7f800000) == 0x7f800000:
            raise ValueError('nonfinite FP32 bits are unsupported')


@dataclass(frozen=True, init=False)
class TensorArtifact:
    _text: str

    def __init__(self, document, *, max_elements=1_000_000):
        if not isinstance(document, dict) or set(document) != {'format', 'graph', 'tensors'}:
            raise ValueError('invalid tensor artifact fields')
        if document['format'] != FORMAT or not isinstance(document['tensors'], dict):
            raise ValueError('unsupported tensor artifact')
        total = 0
        for name, record in document['tensors'].items():
            if not NAME.fullmatch(name) or name in {'parent', 'x', 'y'} or re.fullmatch(r'h\d+', name):
                raise ValueError('invalid or reserved tensor name')
            if not isinstance(record, dict) or set(record) != {'shape', 'dtype', 'bits'}:
                raise ValueError('tensor requires shape, dtype and exact bits')
            shape = record['shape']
            if not isinstance(shape, list) or not 1 <= len(shape) <= 8 or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError('positive bounded tensor dimensions required')
            count = math.prod(shape)
            total += count
            if total > max_elements:
                raise ValueError('tensor element limit exceeded')
            if record['dtype'] != 'f32' or not isinstance(record['bits'], list) or len(record['bits']) != count:
                raise ValueError('tensor dtype or element count differs')
            for word in record['bits']:
                if not isinstance(word, str) or not re.fullmatch('[0-9a-f]{8}', word):
                    raise ValueError('canonical eight-digit FP32 hex bits required')
                Bits(int(word, 16))
        graph = document['graph']
        if not isinstance(graph, dict):
            raise ValueError('graph description required')
        if graph.get('kind') == 'dense':
            if set(graph) != {'kind', 'widths', 'activations', 'layout'}:
                raise ValueError('dense graph fields differ')
            widths, acts = graph['widths'], graph['activations']
            if (not isinstance(widths, list) or not 2 <= len(widths) <= 65
                    or any(type(n) is not int or n < 1 for n in widths)
                    or not isinstance(acts, list) or len(acts) != len(widths)-1
                    or any(a not in ('relu', 'linear', 'tanh') for a in acts)
                    or graph['layout'] not in ('in_out', 'out_in')):
                raise ValueError('unsupported dense graph')
            expected = {}
            for i, (a, b) in enumerate(zip(widths, widths[1:]), 1):
                expected[f'W{i}'] = [a, b] if graph['layout'] == 'in_out' else [b, a]
                expected[f'b{i}'] = [b]
            if set(expected) != set(document['tensors']) or any(document['tensors'][k]['shape'] != v for k, v in expected.items()):
                raise ValueError('tensor shapes do not match dense graph')
        elif graph.get('kind') == 'dag':
            from .wsp_graph import validate_graph
            validate_graph(graph, document['tensors'])
        elif graph.get('kind') == 'opaque':
            if set(graph) != {'kind', 'label'} or not isinstance(graph['label'], str) or len(graph['label']) > 4096:
                raise ValueError('opaque graph needs a bounded label; no execution adapter')
        else:
            raise ValueError('unsupported graph; tied storage/state require a new adapter')
        object.__setattr__(self, '_text', canonical(document))

    @classmethod
    def dense(cls, widths, activations, *, layout='in_out', storage='f32', init='zeros', max_elements=1_000_000):
        if storage != 'f32' or init != 'zeros':
            raise ValueError('first editor profile supports f32 and explicit zero initialization')
        if (not isinstance(widths, (list, tuple)) or not 2 <= len(widths) <= 65
                or any(type(n) is not int or n < 1 for n in widths)
                or layout not in ('in_out', 'out_in')):
            raise ValueError('invalid dense dimensions/layout')
        if sum(a*b+b for a, b in zip(widths, widths[1:])) > max_elements:
            raise ValueError('tensor element limit exceeded before allocation')
        tensors = {}
        for i, (a, b) in enumerate(zip(widths, widths[1:]), 1):
            shape = [a, b] if layout == 'in_out' else [b, a]
            tensors[f'W{i}'] = dict(shape=shape, dtype='f32', bits=['00000000']*(a*b))
            tensors[f'b{i}'] = dict(shape=[b], dtype='f32', bits=['00000000']*b)
        return cls(dict(format=FORMAT, graph=dict(kind='dense', widths=list(widths), activations=list(activations), layout=layout), tensors=tensors), max_elements=max_elements)

    @classmethod
    def from_arrays(cls, arrays, *, graph=None, max_elements=1_000_000):
        values = list(arrays.values())
        if any(not isinstance(a, np.ndarray) or a.dtype != np.dtype('float32') for a in values):
            raise ValueError('explicit float32 arrays required; no silent dtype conversion')
        if sum(a.size for a in values) > max_elements:
            raise ValueError('tensor element limit exceeded before serialization')
        for i, a in enumerate(values):
            if any(np.shares_memory(a, b) for b in values[:i]):
                raise ValueError('shared storage is unsupported; do not silently untie tensors')
        tensors = {k: dict(shape=list(a.shape), dtype='f32', bits=[f'{int(b):08x}' for b in np.ascontiguousarray(a).view(np.uint32).flat]) for k, a in arrays.items()}
        return cls(dict(format=FORMAT, graph=graph or dict(kind='opaque', label='uninterpreted tensor bank'), tensors=tensors), max_elements=max_elements)

    @property
    def sha256(self):
        return hashlib.sha256(self._text.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self._text)

    def arrays(self, *, bits=False):
        result = {}
        for name, record in self.to_dict()['tensors'].items():
            a = np.array([int(s, 16) for s in record['bits']], dtype=np.uint32).reshape(record['shape'])
            result[name] = a if bits else a.view(np.float32)
        return result  # Copies, never live views of artifact storage.

    def with_bits(self, arrays, *, max_elements=1_000_000):
        d = self.to_dict()
        if set(arrays) != set(d['tensors']):
            raise ValueError('cannot add/remove tensors in fixed mode')
        for name, a in arrays.items():
            if a.dtype != np.uint32 or list(a.shape) != d['tensors'][name]['shape']:
                raise ValueError('cannot change tensor shape/dtype in fixed mode')
            d['tensors'][name]['bits'] = [f'{int(b):08x}' for b in a.flat]
        return TensorArtifact(d, max_elements=max_elements)

    def save(self, path):
        with open(path, 'x', encoding='utf-8', newline='\n') as stream:
            stream.write(self._text + '\n')

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as stream:
            data = stream.read(32_000_001)
        if len(data) > 32_000_000:
            raise ValueError('artifact file too large')
        return cls(json.loads(data))

    def trace(self, inputs, *, semantics='real_graph'):
        graph = self.to_dict()['graph']
        if graph['kind'] == 'dag':
            from .wsp_graph import trace_graph
            return trace_graph(self, inputs, semantics)
        if graph['kind'] != 'dense':
            raise ValueError('UNSUPPORTED: opaque execution adapter')
        if len(inputs) != graph['widths'][0]:
            raise ValueError('input width differs')
        if semantics not in ('real_graph', 'numpy_f32'):
            raise ValueError('unknown execution semantics')
        if semantics == 'real_graph' and 'tanh' in graph['activations']:
            raise ValueError('UNSUPPORTED: rational tanh execution')
        arrays = self.arrays(bits=semantics == 'real_graph')
        values = [Fraction(x) for x in inputs] if semantics == 'real_graph' else np.array(inputs, dtype=np.float32)
        if semantics == 'numpy_f32' and not np.isfinite(values).all():
            raise ValueError('finite inputs required')
        taps = {'x': list(values)}
        for i, act in enumerate(graph['activations'], 1):
            w, b = arrays[f'W{i}'], arrays[f'b{i}']
            if graph['layout'] == 'in_out':
                w = w.T
            if semantics == 'real_graph':
                pre = [sum((fraction(c)*x for c, x in zip(row, values)), fraction(bias)) for row, bias in zip(w, b)]
                values = [max(Fraction(0), x) for x in pre] if act == 'relu' else pre
            else:
                with np.errstate(over='raise', invalid='raise'):
                    pre = w @ values + b
                    values = np.maximum(pre, np.float32(0)) if act == 'relu' else np.tanh(pre) if act == 'tanh' else pre
            name = 'y' if i == len(graph['activations']) else f'h{i}'
            taps[f'pre_{name}'], taps[name] = list(pre), list(values)
        return taps


def inspect(artifact, *, tensor=None, offset=0, limit=32):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 0 <= limit <= 4096:
        raise ValueError('invalid page')
    d = artifact.to_dict()
    result = dict(artifact_id=artifact.sha256, graph=d['graph'], tensors={k: dict(shape=v['shape'], dtype=v['dtype'], elements=len(v['bits'])) for k, v in d['tensors'].items()}, aliases=[], immutable=True,
                  capabilities=dict(tensor_edit=True, fixed_placement=d['graph']['kind']=='dense', numpy_execution=d['graph']['kind']=='dense', real_graph=d['graph']['kind']=='dense' and 'tanh' not in d['graph'].get('activations', []), graph_surgery=False))
    if d['graph']['kind'] == 'dag':
        from .wsp_graph import validate_graph, edges
        meta = validate_graph(d['graph'], d['tensors'])
        result.update(graph_analysis=meta, edges=edges(d['graph']),
                      aliases=[dict(tensor=k, references=v) for k,v in meta['shared_parameters'].items()])
        result['capabilities'].update(numpy_execution=True, real_graph=meta['formal_supported'], graph_surgery=True)
    if tensor is not None:
        if tensor not in d['tensors']:
            raise ValueError('unknown tensor')
        words = d['tensors'][tensor]['bits']
        result['page'] = dict(tensor=tensor, offset=offset, total=len(words), values=[dict(flat_index=i, bits=words[i], exact=str(fraction(int(words[i], 16)))) for i in range(offset, min(offset+limit, len(words)))])
    return result


def diff(parent, candidate, *, offset=0, limit=64):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 0 <= limit <= 4096:
        raise ValueError('invalid diff page')
    a, b = parent.to_dict(), candidate.to_dict()
    if (a['graph'] != b['graph'] or set(a['tensors']) != set(b['tensors']) or a['graph']['kind']=='dag'
            or any(a['tensors'][k]['shape']!=b['tensors'][k]['shape'] for k in a['tensors'])):
        from .wsp_graph import graph_diff
        return graph_diff(parent, candidate, offset=offset, limit=limit)
    changed = []; count = 0; tensors = {}
    for name in sorted(a['tensors']):
        ar, br = a['tensors'][name], b['tensors'][name]
        if ar['shape'] != br['shape']:
            raise ValueError('shape changed')
        local = 0
        for i, (old, new) in enumerate(zip(ar['bits'], br['bits'])):
            if old == new:
                continue
            if offset <= count < offset + limit:
                changed.append(dict(tensor=name, flat_index=i, index=list(map(int, np.unravel_index(i, ar['shape']))), before_bits=old, after_bits=new))
            count += 1; local += 1
        if local:
            tensors[name] = local
    return dict(parent_id=parent.sha256, candidate_id=candidate.sha256, status='NO_CHANGE' if count == 0 else 'CHANGED', changed_elements=count, changed_by_tensor=tensors, offset=offset, changes=changed,
                behavior_preservation='not_established', evidence_reuse='requires_dependency_recheck')
