"""Import frozen MLP tensors as an exact rational ReLU artifact (G3a).

Reads a framework-neutral `wspl.mlp_import.v0` bundle whose numbers are hex float
strings, checks dtype representability and explicit tensor orientation, and
stores exactly those tensors in an ordinary ReLUArtifact (allocation mode
`imported`). For imported artifacts `source_sha256` is the canonical bundle hash,
not a scalar-IR hash. Pre/postprocessing composition, output patches and
transactions are not implemented. No training or IEEE execution claim is made.
"""
import argparse
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
from .wspl_artifact import ReLUArtifact, canonical_bytes

FORMAT = 'wspl.mlp_import.v0'
IMPORTER = 'wspl.mlp_import.v0'
DTYPES = ('float32', 'float64')
LAYOUTS = ('out_by_in', 'in_by_out')
_HEX = re.compile(r'[+-]?0x[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?p[+-]?\d+')
_FIELDS = {'format', 'dtype', 'tensor_layout', 'inputs', 'outputs', 'layers',
           'preprocessing', 'postprocessing', 'provenance'}


def _exact(text, dtype, where):
    """Hex float string -> exact Fraction, rejecting decimals, NaN/inf and inexact float32."""
    if not isinstance(text, str) or not _HEX.fullmatch(text):
        raise ValueError(f'{where}: 0x-prefixed hex float string required (float(x).hex()), got {text!r}')
    try:
        value = float.fromhex(text)
    except OverflowError as exc:
        raise ValueError(f'{where}: value overflows float64') from exc
    if not math.isfinite(value):
        raise ValueError(f'{where}: non-finite value')
    if dtype == 'float32':
        try:
            rounded = struct.unpack('<f', struct.pack('<f', value))[0]
        except OverflowError as exc:
            raise ValueError(f'{where}: value overflows float32') from exc
        if rounded != value:
            raise ValueError(f'{where}: value is not exactly representable in float32')
    return Fraction(value)


def _tensor_sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def import_bundle(bundle):
    """Return a ReLUArtifact storing exactly the bundle's tensors."""
    b = copy.deepcopy(bundle)
    if not isinstance(b, dict) or set(b) != _FIELDS or b['format'] != FORMAT:
        raise ValueError(f'{FORMAT} bundle with fields {sorted(_FIELDS)} required')
    if b['dtype'] not in DTYPES:
        raise ValueError(f'dtype must be one of {DTYPES}')
    if b['tensor_layout'] not in LAYOUTS:
        raise ValueError(f'tensor_layout must be one of {LAYOUTS}; orientation is never guessed')
    if b['preprocessing'] != [] or b['postprocessing'] != []:
        raise ValueError('non-identity pre/postprocessing composition is not implemented (G3b)')
    if not isinstance(b['provenance'], dict) or not b['provenance']:
        raise ValueError('nonempty provenance record required')
    if not isinstance(b['layers'], list) or not 1 <= len(b['layers']) <= 65:
        raise ValueError('1 to 65 layers required')
    if not isinstance(b['inputs'], list) or not isinstance(b['outputs'], list):
        raise ValueError('input and output port lists required')
    width = len(b['inputs'])
    layers, hashes = [], []
    for index, layer in enumerate(b['layers']):
        where = f'layer {index}'
        if not isinstance(layer, dict) or set(layer) != {'activation', 'weight', 'bias'}:
            raise ValueError(f'{where}: activation, weight and bias required')
        expected = 'linear' if index == len(b['layers']) - 1 else 'relu'
        if layer['activation'] != expected:
            raise ValueError(f'{where}: only hidden relu and a final linear layer are supported')
        weight, bias = layer['weight'], layer['bias']
        if not isinstance(weight, list) or not weight or any(not isinstance(r, list) or not r for r in weight):
            raise ValueError(f'{where}: nonempty 2-D weight tensor required')
        if len({len(r) for r in weight}) != 1:
            raise ValueError(f'{where}: ragged weight tensor')
        rows = weight if b['tensor_layout'] == 'out_by_in' else [list(c) for c in zip(*weight)]
        if len(rows[0]) != width:
            raise ValueError(f'{where}: weight expects {len(rows[0])} inputs, previous width is {width}; check tensor_layout')
        if not isinstance(bias, list) or len(bias) != len(rows):
            raise ValueError(f'{where}: bias length differs from output width')
        layers.append({'activation': expected,
                       'weights': [[str(_exact(v, b['dtype'], f'{where} weight')) for v in r] for r in rows],
                       'bias': [str(_exact(v, b['dtype'], f'{where} bias')) for v in bias]})
        hashes.append({'weight': _tensor_sha(weight), 'bias': _tensor_sha(bias)})
        width = len(rows)
    widths = [len(b['inputs'])] + [len(l['bias']) for l in layers]
    parameters = sum(o * (i + 1) for i, o in zip(widths, widths[1:]))
    ports = {key: [{'name': p.get('name') if isinstance(p, dict) else None,
                    'type': _port_type(p)} for p in b[key]] for key in ('inputs', 'outputs')}
    artifact = {'format': 'wspl.relu_mlp.v0', 'semantics': 'rational_real_graph', 'compiler': IMPORTER,
                'source_sha256': hashlib.sha256(canonical_bytes(b)).hexdigest(),
                'inputs': ports['inputs'], 'outputs': ports['outputs'], 'layers': layers,
                'allocation': {'mode': 'imported', 'widths': widths, 'parameters': parameters},
                'source_map': {'import': {'bundle_format': FORMAT, 'dtype': b['dtype'],
                                          'tensor_layout': b['tensor_layout'],
                                          'stored_layers': 'exact imported tensors' + (
                                              ', transposed from in_by_out' if b['tensor_layout'] == 'in_by_out' else ''),
                                          'tensor_sha256': hashes, 'preprocessing': [], 'postprocessing': [],
                                          'provenance': b['provenance']}}}
    return ReLUArtifact(artifact)


def _port_type(port):
    if not isinstance(port, dict) or set(port) != {'name', 'type'} or not isinstance(port['type'], dict):
        raise ValueError('ports need name and type')
    ty = port['type']
    if set(ty) != {'kind', 'unit', 'lower', 'upper'}:
        raise ValueError('port type needs kind, unit, lower and upper')
    if not isinstance(ty['lower'], str) or not isinstance(ty['upper'], str):
        raise ValueError('port bounds must be exact rational strings')
    try:
        lower, upper = Fraction(ty['lower']), Fraction(ty['upper'])
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError('exact rational port bounds required') from exc
    return {'kind': ty['kind'], 'unit': ty['unit'], 'lower': str(lower), 'upper': str(upper)}


def bundle_from_artifact(artifact, *, provenance, dtype='float32', tensor_layout='out_by_in'):
    """Re-express an existing artifact's stored weights as an import bundle.

    Used for import-fidelity fixtures. The result is not a learned network unless
    the artifact was one; record that in `provenance`.
    """
    if not isinstance(artifact, ReLUArtifact):
        artifact = ReLUArtifact(artifact)
    d = artifact.to_dict()

    def encode(text):
        value = Fraction(text)
        stored = float(value)
        if Fraction(stored) != value:
            raise ValueError(f'{value} is not exactly representable in float64')
        hex_text = stored.hex()
        _exact(hex_text, dtype, 'export')
        return hex_text
    layers = []
    for layer in d['layers']:
        rows = [[encode(v) for v in r] for r in layer['weights']]
        layers.append({'activation': layer['activation'],
                       'weight': rows if tensor_layout == 'out_by_in' else [list(c) for c in zip(*rows)],
                       'bias': [encode(v) for v in layer['bias']]})
    return {'format': FORMAT, 'dtype': dtype, 'tensor_layout': tensor_layout,
            'inputs': d['inputs'], 'outputs': d['outputs'], 'layers': layers,
            'preprocessing': [], 'postprocessing': [], 'provenance': copy.deepcopy(provenance)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        artifact = import_bundle(json.loads(args.bundle.read_text(encoding='utf-8')))
        artifact.save(args.output)
        print(json.dumps({'output': str(args.output), 'sha256': artifact.sha256,
                          'allocation': artifact.to_dict()['allocation'], 'universal_verification': 'not_performed'}))
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
