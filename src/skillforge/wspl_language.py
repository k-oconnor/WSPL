"""WSPL 0.3: typed definitions, identity-bound editing and task composition.

Legacy frontends remain unchanged. This frontend lowers to their existing IRs;
it never evaluates proposal Python or supplies a task solver.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re

import numpy as np

from .control_dsl import Call
from .wspl_parse import parse_module
from .wspl_compile import compile_scalar
from .wsp_artifact import TensorArtifact, round_f32, fraction
from .wsp_editor import plan_and_apply
from .wsp_syntax import Reader


SIGNATURES = {'move_to': ('ObjectOrPoint',), 'grasp': ('Object',),
              'release': ('Object',), 'transfer': ('Object', 'Point')}
BLOCK_PORTS = {'move': (('x','y'), ('vx','vy','grip')),
               'near': (('x','y'), ('score',)), 'holding': (('h',), ('score',)),
               'close': (('x','y'), ('vx','vy','grip')),
               'open': (('x','y'), ('vx','vy','grip')),
               'safe': (('x','y'), ('score',))}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def library_document(blocks, specs):
    """The host links observed-state skills to exact typed module artifacts."""
    if set(blocks) != set(BLOCK_PORTS) or set(specs) != set(BLOCK_PORTS):
        raise ValueError('Complete planar library required')
    records = {}
    for name, block in blocks.items():
        if (block.input_names, block.output_names) != BLOCK_PORTS[name]:
            raise ValueError('Library port mismatch: '+name)
        records[name] = dict(sha256=block.sha256, inputs=block.to_dict()['inputs'],
                             outputs=block.to_dict()['outputs'], spec=specs[name])
    return dict(format='wspl.library.v1', runtime='planar.observed_state.v1',
                signatures={k:list(v) for k,v in SIGNATURES.items()}, modules=records)


def tensor_from_module(module):
    """Explicit rational -> FP32 boundary; report all coefficient rounding."""
    d = module.to_dict()
    target = TensorArtifact.dense(d['allocation']['widths'],
                                 [x['activation'] for x in d['layers']], layout='out_in')
    arrays = target.arrays(bits=True)
    rounded = 0
    maximum = Fraction(0)
    for i, layer in enumerate(d['layers'], 1):
        for name, values in [(f'W{i}', layer['weights']), (f'b{i}', layer['bias'])]:
            values = np.array(values, dtype=object)
            for index in np.ndindex(values.shape):
                q = Fraction(values[index]); word = round_f32(q)
                error = abs(q-fraction(word))
                rounded += bool(error); maximum = max(maximum, error)
                arrays[name][index] = word
    artifact = target.with_bits(arrays)
    return artifact, dict(source_module_sha256=module.sha256,
                         stored_artifact_sha256=artifact.sha256,
                         rounded_coefficients=rounded, max_coefficient_error=str(maximum),
                         source_equivalence='not_checked', contracts='not_transferred')


@dataclass(frozen=True)
class Compilation:
    modules: dict
    module_contracts: dict
    imports: dict
    calls: tuple
    task_name: str | None
    edit_plan: object
    candidate: object
    provenance: dict


def _declaration(reader):
    """Consume one balanced top-level declaration while preserving source text."""
    start = reader.tok.start
    depth = 0
    seen_block = False
    while reader.tok.text != '<EOF>':
        token = reader.take()
        if token == '{':
            depth += 1; seen_block = True
            if depth > 64: reader.fail('Declaration nesting limit')
        elif token == '}':
            depth -= 1
            if depth < 0: reader.fail('Unbalanced declaration')
            if seen_block and depth == 0:
                return reader.text[start:reader.tokens[reader.i-1].end]
        elif token == ';' and depth == 0:
            return reader.text[start:reader.tokens[reader.i-1].end]
    reader.fail('Unterminated declaration')


def compile_source(source, *, libraries=None, objects=(), points=(), limit=128,
                   parent=None, bindings=None, artifacts=None, budgets=None):
    """Compile one unified source. Libraries are host data, never disk imports."""
    if type(limit) is not int or limit < 1: raise ValueError('Positive call budget required')
    if set(objects) & set(points): raise ValueError('Object/point namespaces overlap')
    reader = Reader(source)
    reader.take('wspl'); reader.take('0.3'); reader.take(';')
    modules = {}; contracts = {}; imports = {}; editor = []; calls = []
    task_name = None; phase = 0; names = set()
    library_map = libraries or {}
    while reader.tok.text != '<EOF>':
        kind = reader.tok.text
        if kind in ('module','use'):
            if phase: reader.fail('Definitions and imports must precede editing/tasks')
            text = _declaration(reader)
            r = Reader(text); r.take(kind); name = r.name()
            if name in names: raise ValueError('Duplicate declaration: '+name)
            names.add(name)
            if kind == 'module':
                parsed = parse_module(text)
                modules[name] = compile_scalar(parsed.program)
                contracts[name] = parsed.contracts
            else:
                r.take('='); r.take('library'); r.take('(')
                key = json.loads(r.take()); r.take(','); r.take('sha256'); r.take('=')
                expected = json.loads(r.take()); r.take(')'); r.take(';')
                if r.tok.text != '<EOF>': r.fail('Trailing import syntax')
                if not isinstance(key,str) or key not in library_map:
                    raise ValueError('Unknown host library')
                actual = library_map[key]
                if not isinstance(expected,str) or expected != digest(actual):
                    raise ValueError('Library identity mismatch')
                if (actual.get('format') != 'wspl.library.v1' or
                    actual.get('runtime') != 'planar.observed_state.v1' or
                    actual.get('signatures') != {k:list(v) for k,v in SIGNATURES.items()}):
                    raise ValueError('Unsupported library ABI')
                imports[name] = dict(key=key, sha256=expected)
        elif kind in ('target','const','circuit','region','edit','describe'):
            if phase == 2: reader.fail('Editor declarations must precede task')
            phase = 1; text = _declaration(reader)
            if kind != 'describe':
                r=Reader(text);r.take(kind);name=r.name()
                if kind=='edit':r.take('as');name=r.name()
                if name in names:raise ValueError('Duplicate declaration: '+name)
                names.add(name)
            editor.append(text)
        elif kind == 'task':
            if task_name is not None: reader.fail('One task per source')
            phase = 2; reader.take('task'); task_name = reader.name()
            if task_name in names: reader.fail('Duplicate task name')
            reader.take('{')
            while reader.tok.text != '}':
                alias = reader.name(); reader.take('.'); op = reader.name(); reader.take('(')
                args = []
                if reader.tok.text != ')':
                    args.append(reader.name())
                    while reader.tok.text == ',': reader.take(','); args.append(reader.name())
                reader.take(')'); reader.take(';')
                if alias not in imports or op not in SIGNATURES:
                    raise ValueError('Unknown imported skill')
                if len(args) != len(SIGNATURES[op]): raise ValueError('Skill arity mismatch')
                for name, ty in zip(args, SIGNATURES[op]):
                    allowed = set(objects) if ty == 'Object' else set(points) if ty == 'Point' else set(objects)|set(points)
                    if name not in allowed: raise ValueError('Skill argument type mismatch: '+name)
                if op == 'transfer':
                    calls.extend((Call('move_to',args[0]), Call('grasp',args[0]),
                                  Call('move_to',args[1]), Call('release',args[0])))
                else: calls.append(Call(op,args[0]))
                if len(calls) > limit: raise ValueError('Expanded instruction budget exceeded')
            reader.take('}')
            if not calls: raise ValueError('Empty task')
            if reader.tok.text != '<EOF>': reader.fail('Task must be last')
        else: reader.fail('Unknown unified declaration')
    if not modules and not editor and task_name is None: raise ValueError('Empty WSPL program')
    edit_plan = candidate = None; bridge = None
    for i, text in enumerate(editor):
        match = re.fullmatch(r'target\s+(\w+)\s*=\s*compiled\s*\(\s*(\w+)\s*\)\s*;', text)
        if match:
            alias, name = match.groups()
            if name not in modules or parent is not None: raise ValueError('Invalid compiled module binding')
            parent, bridge = tensor_from_module(modules[name])
            editor[i] = f'target {alias}=artifact("{name}",sha256="{parent.sha256}");'
    if editor:
        edit_plan, candidate = plan_and_apply('wspl 0.2;\n'+'\n'.join(editor), parent=parent,
                                             bindings=bindings, artifacts=artifacts, budgets=budgets)
    # A task cannot silently select between two different motor libraries.
    used_hashes = {v['sha256'] for v in imports.values()}
    if task_name and len(used_hashes) != 1: raise ValueError('Task requires one exact runtime library')
    provenance = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      modules={k:v.sha256 for k,v in modules.items()}, imports=imports,
                      conversion=bridge, editor_plan_sha256=edit_plan.sha256 if edit_plan else None)
    return Compilation(modules, contracts, imports, tuple(calls), task_name,
                       edit_plan, candidate, provenance)
