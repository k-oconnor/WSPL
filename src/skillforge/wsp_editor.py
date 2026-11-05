"""First-pass WSP editor: plan/apply, immutable tensors and explicit placement.

CLI and Python API only. No model calls, training, deployment or admission.
"""
import argparse
import ast
from dataclasses import dataclass, asdict
from fractions import Fraction
import functools
import hashlib
import io
import json
from pathlib import Path
import re
import tokenize

import numpy as np

from .wsp_artifact import TensorArtifact, Bits, canonical, digest, fraction, round_f32, inspect, diff
from .wsp_values import Arithmetic, Limits, BUILTINS, integer, scalar, pack, unpack, numeric
from .wsp_syntax import parse, expression
from .wsp_circuit import place


@functools.lru_cache(maxsize=None)
def implementation_id():
    paths = ('wsp_artifact.py', 'wsp_syntax.py', 'wsp_values.py', 'wsp_circuit.py',
             'wsp_editor.py', 'wsp_evidence.py', 'wsp_graph.py', 'wsp_surgery.py',
             'native_weight_edits.py', 'csail_breakout.py')
    return digest({p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in paths})


class EditError(ValueError):
    def __init__(self, message, span=None):
        self.diagnostic = dict(status='INVALID', reason=str(message), source_span=span, parent_unchanged=True)
        super().__init__(f'{span}: {message}' if span else str(message))


def _config(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.List):
        return [_config(x) for x in node.elts]
    raise ValueError('literal target configuration required')


def _target(record, parent, limits):
    node = expression(record['expression']).body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError('dense(...) or artifact(...) target required')
    kwargs = {}
    for kw in node.keywords:
        if kw.arg is None or kw.arg in kwargs:
            raise ValueError('unique target keywords required')
        kwargs[kw.arg] = _config(kw.value)
    if node.func.id == 'dense':
        if node.args or set(kwargs)-{'widths', 'activations', 'layout', 'storage', 'init'} or not {'widths', 'activations'} <= set(kwargs):
            raise ValueError('invalid dense constructor arguments')
        declared = TensorArtifact.dense(**kwargs, max_elements=limits.max_elements)
        if parent is not None and parent.sha256 != declared.sha256:
            raise ValueError('provided parent differs from declared zero target')
        return declared
    if node.func.id == 'artifact':
        if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str) or set(kwargs) != {'sha256'}:
            raise ValueError('artifact requires a name and exact sha256')
        if parent is None or parent.sha256 != kwargs['sha256']:
            raise ValueError('missing parent or artifact identity differs')
        return parent.within(limits.max_elements)  # Immutable; only the element limit can differ.
    raise ValueError('unsupported target constructor')


def selected_flat_indices(shape, selection):
    """Row-major indices for basic indexing, without materializing the tensor."""
    strides=[];stride=1
    for width in reversed(shape):
        strides.append(stride);stride*=width
    strides.reverse()
    axes=[];offset=0
    for width,index,stride in zip(shape,selection,strides):
        if isinstance(index,slice):
            start,stop,step=index.indices(width)
            axes.append(np.arange(start,stop,step,dtype=np.int64)*stride)
        else:offset+=(index%width)*stride
    result=np.array(offset,dtype=np.int64)
    for axis,values in enumerate(axes):
        dims=[1]*len(axes);dims[axis]=len(values)
        result=result+values.reshape(dims)
    return result


class Builder:
    def __init__(self, parent, bindings, limits):
        self.parent = parent
        self.base = parent
        self.parent_graph = parent.graph
        self.graph_context = None
        self.arrays = parent.arrays(bits=True)
        self.arithmetic = Arithmetic(self.arrays, parent.arrays(bits=True), {}, bindings, limits)
        self.limits = limits
        self.writes = []; self.rounding = []; self.rounded_count = 0
        self.placements = []; self.reserved = set(); self.touched = {}
        self.statements = 0; self.loop_iterations = 0

    def current(self):
        return self.base.with_bits(self.arrays,max_elements=self.limits.max_elements)

    def write(self, name, selection, value, op, span, label='raw'):
        if name not in self.arrays:
            raise ValueError('unknown writable tensor')
        target = np.asarray(self.arrays[name][selection])
        # Basic-index coordinate arithmetic scales with selected elements/rank.
        self.arithmetic.charge(max(1,target.size)*(len(selection)+1))
        indices = selected_flat_indices(self.arrays[name].shape,selection)
        self.arithmetic.charge(target.size)
        rhs = np.asarray(value)
        try:
            value = np.broadcast_to(value, target.shape)
        except ValueError as exc:
            raise ValueError('RHS does not broadcast to selected shape') from exc
        fast = self._fast_words(op, target, rhs)
        if fast is None:
            converted = self._reference_words(name, op, target, value, indices)
        else:
            converted, rounded, requested = fast
            positions = np.flatnonzero(rounded)
            self.rounded_count += len(positions)
            for pos in positions[:max(0, 64-len(self.rounding))].tolist():
                word = int(converted.flat[pos])
                self.rounding.append(dict(tensor=name, flat_index=int(np.asarray(indices).flat[pos]), requested=requested(pos),
                                          stored=str(fraction(word)), bits=f'{word:08x}'))
        self.arrays[name][selection] = np.asarray(converted, dtype=np.uint32).reshape(target.shape)
        self.touched.setdefault(name, []).append(np.asarray(indices, dtype=np.int64).reshape(-1))
        self.writes.append(dict(tensor=name, flat_indices=np.asarray(indices).reshape(-1).tolist(), operator=op, source_span=span, origin=label, selected_shape=list(target.shape), empty=target.size==0))

    def _reference_words(self, name, op, target, value, indices):
        """Original element-wise exact path; also the fallback for every edge case."""
        converted = []
        for flat, old, item in zip(np.asarray(indices).flat, target.flat, value.flat):
            if op != '=':
                a, b = fraction(old), fraction(item.value) if isinstance(item, Bits) else item
                item = a+b if op == '+=' else a-b if op == '-=' else a*b
            if isinstance(item, Bits):
                word = item.value
            else:
                self.arithmetic.bounded(np.array(item, dtype=object))
                word = round_f32(item)
                if fraction(word) != item:
                    self.rounded_count += 1
                    if len(self.rounding) < 64:
                        self.rounding.append(dict(tensor=name, flat_index=int(flat), requested=str(item), stored=str(fraction(word)), bits=f'{word:08x}'))
            converted.append(word)
        return converted

    def _fast_words(self, op, target, rhs):
        """Same words, rounding flags and requested values as _reference_words.

        Returns None whenever exactness or an error cannot be decided cheaply;
        the caller then runs the reference path, which raises any error itself.
        Nothing here mutates builder state.
        """
        # Every float64 is a dyadic rational with < 1100-bit numerator/denominator,
        # so the reference precision check cannot fire on float64-exact values.
        if self.limits.max_integer_bits < 1100 or target.size == 0:
            return None
        if rhs.dtype != object:
            return None
        # Work per RHS element (before broadcasting), then broadcast the results.
        base = rhs
        items = base.reshape(-1)
        cache = {}
        if op == '=':
            words = np.empty(items.size, np.uint32); rounded = np.zeros(items.size, bool); exact = []
            for i, item in enumerate(items.tolist()):
                hit = cache.get((type(item), item))  # Fraction(1) == 1 == True; keep types apart.
                if hit is None:
                    if isinstance(item, Bits):
                        hit = (item.value, False)
                    elif type(item) is Fraction:
                        try:
                            self.arithmetic.bounded(np.array(item, dtype=object))
                            word = round_f32(item)
                        except (ValueError, OverflowError):
                            return None
                        hit = (word, fraction(word) != item)
                    else:
                        return None
                    cache[(type(item), item)] = hit
                words[i], rounded[i] = hit
                exact.append(item)
            exact = np.array(exact, dtype=object)
            words = np.broadcast_to(words.reshape(base.shape), target.shape).reshape(-1)
            rounded = np.broadcast_to(rounded.reshape(base.shape), target.shape).reshape(-1)
            exact = np.broadcast_to(exact.reshape(base.shape), target.shape).reshape(-1)
            return words.copy(), rounded, lambda pos: str(exact[pos])
        # Augmented ops: exact float64 arithmetic where provably exact.
        rhs = np.empty(items.size, np.float64)
        for i, item in enumerate(items.tolist()):
            hit = cache.get((type(item), item))  # Fraction(1) == 1 == True; keep types apart.
            if hit is None:
                if isinstance(item, Bits):
                    q = fraction(item.value)
                elif type(item) is Fraction:
                    q = item
                else:
                    return None
                try:
                    f = float(q)
                except OverflowError:
                    return None
                # Products need a 24-bit RHS to stay exact; sums need an exact float64 RHS.
                with np.errstate(over='ignore'):
                    single = float(np.float32(f))
                if Fraction(f) != q or (op == '*=' and single != f):
                    return None
                hit = cache[(type(item), item)] = f
            rhs[i] = hit
        b = np.broadcast_to(rhs.reshape(base.shape), target.shape).reshape(-1)
        a = np.ascontiguousarray(target).reshape(-1).view(np.float32).astype(np.float64)
        with np.errstate(over='ignore', invalid='ignore'):
            if op == '*=':
                result = a*b; inexact = np.zeros(result.size, bool)
            else:
                b = b if op == '+=' else -b
                result = a+b
                back = result-a  # TwoSum: error is zero iff the float64 sum is exact.
                inexact = ((a-(result-back))+(b-back)) != 0
            single = result.astype(np.float32)
        if not np.isfinite(result).all() or not np.isfinite(single).all():
            return None
        words = single.view(np.uint32).copy()
        words[result == 0] = 0  # Exact rational zero is +0, as Fraction rounding gives.
        rounded = single.astype(np.float64) != result
        slow = np.flatnonzero(inexact)
        exact_slow = {}
        if len(slow):
            olds = np.ascontiguousarray(target).reshape(-1)
            spread = np.broadcast_to(base, target.shape).reshape(-1)
        for pos in slow.tolist():
            old, item = olds[pos], spread[pos]
            x, y = fraction(old), fraction(item.value) if isinstance(item, Bits) else item
            q = x+y if op == '+=' else x-y
            try:
                self.arithmetic.bounded(np.array(q, dtype=object))
                word = round_f32(q)
            except (ValueError, OverflowError):
                return None
            words[pos] = word; rounded[pos] = fraction(word) != q; exact_slow[pos] = q
        return words, rounded, lambda pos: str(exact_slow[pos] if pos in exact_slow else Fraction(float(result[pos])))

    def raw(self, row, profile):
        text = row['text']; tree = expression(text, mode='exec')
        if len(tree.body) != 1:
            raise ValueError('one assignment per statement required')
        node = tree.body[0]
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, op = node.targets[0], '='
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            target, op = node.target, {ast.Add:'+=', ast.Sub:'-=', ast.Mult:'*='}[type(node.op)]
        else:
            raise ValueError('only tensor assignment and +=/-=/*= supported')
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name) or target.value.id not in self.arrays:
            raise ValueError('assignment needs a declared tensor slice')
        name = target.value.id
        selected, clipped = self.arithmetic.selection(target.slice, self.arrays[name].shape, text)
        if profile == 'exact_to_storage':
            value = self.arithmetic.eval(node.value, text)
            self.write(name, selected, value, op, row['span'])
        else:
            from .native_weight_edits import value as native_value, index as native_index
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if tok.type == tokenize.NUMBER and not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)', tok.string):
                    raise ValueError('native profile requires plain decimal literals')
            entries = target.slice.elts if isinstance(target.slice, ast.Tuple) else [target.slice]
            selected = tuple(native_index(x) for x in entries)
            literal = native_value(node.value)
            a = self.arrays[name].view(np.float32).copy()
            with np.errstate(over='raise', invalid='raise'):
                if op == '=': a[selected] = literal
                elif op == '+=': a[selected] += literal
                elif op == '-=': a[selected] -= literal
                else: a[selected] *= literal
            if not np.isfinite(a).all() or (np.abs(a) > 1e6).any():
                raise ValueError('edited tensor outside native limits')
            words = np.asarray(a[selected]).copy().view(np.uint32)
            value = np.array([Bits(int(x)) for x in words.flat], dtype=object).reshape(words.shape)
            self.write(name, selected, value, '=', row['span'], 'native_numpy_v1')
            self.writes[-1]['operator'] = op
        self.writes[-1]['clipped_slice'] = bool(clipped)
        if self.parent_graph['kind']=='dag':
            from .wsp_graph import parameter_uses
            self.writes[-1]['parameter_users']=parameter_uses(self.base.graph).get(name,[])
        for p in self.placements:
            p['later_edits_may_invalidate'] = True  # Conservative until dependency analysis.

    def run(self, rows, circuits, profile):
        for row in rows:
            self.statements += 1
            if self.statements > self.limits.max_statements:
                raise EditError('statement expansion budget exhausted', row['span'])
            try:
                if row['kind'] == 'write':
                    self.raw(row, profile)
                elif row['kind'] == 'for':
                    if profile != 'exact_to_storage':
                        raise ValueError('native profile supports literal statements only')
                    name = row['name']
                    if name in self.arithmetic.constants or name in self.arrays or name in BUILTINS or name in {'parent','x','y'} or re.fullmatch(r'h\d+', name):
                        raise ValueError('loop variable shadows a binding')
                    call = expression('range('+row['arguments']+')').body
                    if call.keywords or not 1 <= len(call.args) <= 3:
                        raise ValueError('range needs one to three constant arguments')
                    args = [integer(self.arithmetic.eval(n, 'range('+row['arguments']+')')) for n in call.args]
                    iterations = range(*args)
                    count = len(iterations)
                    self.loop_iterations += count
                    if self.loop_iterations > self.limits.max_loop_iterations:
                        raise ValueError('construction loop budget exceeded before expansion')
                    try:
                        for value in iterations:
                            self.arithmetic.constants[name] = np.array(Fraction(value), dtype=object)
                            self.run(row['body'], circuits, profile)
                    finally:
                        self.arithmetic.constants.pop(name, None)
                elif row['kind'] in ('replace','transplant','add_residual','redirect','untie'):
                    if self.graph_context is None:raise ValueError('structural edits require graph mode and a host budget')
                    self.graph_context.execute(self,row)
                else:
                    if profile != 'exact_to_storage' or row['name'] not in circuits:
                        raise ValueError('placement needs exact profile and a declared circuit')
                    for old in self.placements:
                        old['later_edits_may_invalidate'] = True
                    report = place(row, circuits[row['name']], self.parent_graph, self.arithmetic, self.write, self.reserved)
                    self.placements.append(report)
            except EditError:
                raise
            except (ValueError, TypeError, IndexError, KeyError, OverflowError, ZeroDivisionError, FloatingPointError) as exc:
                raise EditError(str(exc), row['span']) from exc


@dataclass(frozen=True)
class EditPlan:
    _text: str

    @property
    def sha256(self):
        return hashlib.sha256(self._text.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self._text)

    def save(self, path):
        with open(path, 'x', encoding='utf-8', newline='\n') as stream:
            stream.write(canonical(dict(plan_id=self.sha256, body=self.to_dict()))+'\n')

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as stream:
            text = stream.read(64_000_001)
        if len(text) > 64_000_000:
            raise ValueError('plan file limit exceeded')
        d = json.loads(text)
        if not isinstance(d, dict) or set(d) != {'plan_id','body'} or not isinstance(d['body'],dict) or digest(d['body']) != d['plan_id']:
            raise ValueError('plan identity differs')
        return cls(canonical(d['body']))


def plan(source, *, parent=None, bindings=None, limits=Limits(), artifacts=None, budgets=None):
    tree = parse(source)
    parent = _target(tree['target'], parent, limits)
    bound = {k: unpack(pack(v), limits) for k, v in (bindings or {}).items()}
    builder = Builder(parent, bound, limits)
    if tree['edit'] and tree['edit']['mode']=='graph':
        from .wsp_surgery import GraphContext
        if tree['edit']['numeric']!='exact_to_storage':raise ValueError('graph mode requires exact_to_storage')
        builder.graph_context=GraphContext(parent,tree,artifacts,budgets)
    elif tree['regions']:
        raise ValueError('region declarations require a graph edit transaction')
    for const in tree['constants']:
        if const['name'] in builder.arrays or const['name'] in BUILTINS or const['name'] in {'relu','clip','min','max'}:
            raise EditError('constant shadows a reserved name')
        builder.arithmetic.constants[const['name']] = builder.arithmetic.run(const['expression'])
    profile = tree['edit']['numeric'] if tree['edit'] else 'exact_to_storage'
    if profile == 'native_numpy_v1' and (tree['constants'] or tree['circuits']):
        raise EditError('native compatibility profile supports literal edits only')
    if tree['edit']:
        builder.run(tree['edit']['body'], tree['circuits'], profile)
    candidate = builder.current()
    if builder.graph_context:builder.graph_context.budget.check(candidate,parent)
    from .wsp_evidence import describe
    descriptions = []
    for d in tree['descriptions']:
        artifact = parent if d['target'] == tree['target']['name'] else candidate
        descriptions.append(describe(artifact, d).to_dict())
    changed = diff(parent, candidate, limit=0)
    body = dict(format='wsp.edit_plan.v1', implementation_id=implementation_id(), source=source,
                parent=parent.document(), parent_id=parent.sha256, candidate=candidate.document(), candidate_id=candidate.sha256,
                limits=asdict(limits), bindings={k:pack(v) for k,v in bound.items()}, numeric_profile=profile,
                native_runtime=dict(numpy=np.__version__) if profile=='native_numpy_v1' else None,
                writes=builder.writes, placements=builder.placements, descriptions=descriptions,
                rounding=dict(changed=builder.rounded_count, examples=builder.rounding),
                resources=dict(statements=builder.statements, loop_iterations=builder.loop_iterations, operations=builder.arithmetic.operations),
                touched={k:np.unique(np.concatenate(v)).tolist() if v else [] for k,v in builder.touched.items()}, diff=changed,
                graph_edits=builder.graph_context.receipt() if builder.graph_context else None,
                status='NO_CHANGE' if candidate.sha256==parent.sha256 else 'STAGED', admission='not_performed')
    staged = EditPlan(canonical(body))
    object.__setattr__(staged, '_candidate', candidate)  # Cache only; identity is the text.
    return staged


def apply(edit_plan, *, parent, expected_parent_id):
    d = edit_plan.to_dict()
    if d.get('format') != 'wsp.edit_plan.v1' or d.get('implementation_id') != implementation_id():
        raise ValueError('stale or unsupported plan implementation')
    if parent.sha256 != expected_parent_id or parent.sha256 != d['parent_id']:
        raise ValueError('stale parent identity')
    limits = Limits(**d['limits'])
    graph=d.get('graph_edits');artifacts={};budgets={}
    if graph:
        from .wsp_graph import GraphBudget
        artifacts={k:TensorArtifact(v) for k,v in graph['modules'].items()}
        budgets={graph['budget_name']:GraphBudget(**graph['budget'])}
    replay = plan(d['source'], parent=parent, bindings={k:unpack(v, limits) for k,v in d['bindings'].items()}, limits=limits,
                  artifacts=artifacts,budgets=budgets)
    if replay.sha256 != edit_plan.sha256:
        raise ValueError('plan replay differs; source/receipt/candidate tampering')
    # Equal plan text implies this is exactly the recorded candidate document.
    return replay._candidate.within(limits.max_elements)


def import_legacy(answer):
    """Historical skip-invalid import; isolated and explicitly labelled."""
    from .native_weight_edits import NativeEditedPolicy, NAMES
    policy = NativeEditedPolicy()
    receipt = policy.apply_answer(answer)
    target = TensorArtifact.dense([600,32,32,6], ['tanh','tanh','linear'])
    artifact = TensorArtifact.from_arrays({k:policy.weights[v].copy() for k,v in NAMES.items()}, graph=target.to_dict()['graph'])
    return artifact, dict(receipt, artifact_id=artifact.sha256, profile='legacy_skip_invalid_native_numpy_v1', numpy=np.__version__)


def _write_json(path, value):
    with open(path, 'x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan'); p.add_argument('source', type=Path); p.add_argument('--parent', type=Path); p.add_argument('--output', type=Path, required=True); p.add_argument('--module', action='append', default=[]); p.add_argument('--budget', action='append', default=[])
    p = sub.add_parser('apply'); p.add_argument('plan', type=Path); p.add_argument('--parent', type=Path); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('inspect'); p.add_argument('artifact', type=Path); p.add_argument('--tensor'); p.add_argument('--offset', type=int, default=0); p.add_argument('--limit', type=int, default=32)
    p = sub.add_parser('diff'); p.add_argument('parent', type=Path); p.add_argument('candidate', type=Path)
    p = sub.add_parser('check'); p.add_argument('artifact', type=Path); p.add_argument('description', type=Path); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('describe'); p.add_argument('artifact', type=Path); p.add_argument('--source', type=Path); p.add_argument('--output', type=Path); p.add_argument('--layer', type=int, default=1); p.add_argument('--node'); p.add_argument('--offset', type=int, default=0); p.add_argument('--limit', type=int, default=16); p.add_argument('--origin', choices=['exploratory','external_requirement'], default='exploratory')
    p = sub.add_parser('probe'); p.add_argument('artifact', type=Path); p.add_argument('inputs', type=Path); p.add_argument('--semantics', choices=['real_graph','numpy_f32'], default='real_graph'); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('graphify'); p.add_argument('artifact', type=Path); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('boundary'); p.add_argument('artifact', type=Path); p.add_argument('--nodes', nargs='+', required=True); p.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'plan':
            from .wsp_graph import GraphBudget
            modules={};budgets={}
            for entries,destination,loader in ((args.module,modules,TensorArtifact.load),
                                              (args.budget,budgets,lambda p:GraphBudget(**json.loads(Path(p).read_text(encoding='utf-8'))))):
                for entry in entries:
                    name,separator,path=entry.partition('=')
                    if not separator or not name or name in destination:raise ValueError('unique name=path binding required')
                    destination[name]=loader(path)
            planned = plan(args.source.read_text(encoding='utf-8'), parent=TensorArtifact.load(args.parent) if args.parent else None,
                           artifacts=modules,budgets=budgets)
            planned.save(args.output); d = planned.to_dict()
            print(canonical(dict(plan_id=planned.sha256, parent_id=d['parent_id'], candidate_id=d['candidate_id'], status=d['status'], diff=d['diff'], admission=d['admission'])))
        elif args.command == 'apply':
            planned = EditPlan.load(args.plan)
            parent = TensorArtifact.load(args.parent) if args.parent else TensorArtifact(planned.to_dict()['parent'])
            candidate = apply(planned, parent=parent, expected_parent_id=parent.sha256)
            candidate.save(args.output); print(canonical(dict(candidate_id=candidate.sha256, status='STAGED', admission='not_performed')))
        elif args.command == 'inspect':
            print(json.dumps(inspect(TensorArtifact.load(args.artifact), tensor=args.tensor, offset=args.offset, limit=args.limit), indent=2))
        elif args.command == 'diff':
            print(json.dumps(diff(TensorArtifact.load(args.parent), TensorArtifact.load(args.candidate)), indent=2))
        elif args.command == 'describe':
            from .wsp_evidence import describe, equations
            artifact = TensorArtifact.load(args.artifact)
            if args.source:
                if not args.output:
                    raise ValueError('description source needs an exclusive --output file')
                description = describe(artifact, args.source.read_text(encoding='utf-8'), origin=args.origin)
                description.save(args.output)
                print(canonical(dict(description_id=description.sha256, artifact_id=artifact.sha256, executable_changes=0)))
            else:
                result = equations(artifact, layer=args.layer, node_id=args.node, offset=args.offset, limit=args.limit)
                if args.output:
                    _write_json(args.output, result)
                print(json.dumps(result, indent=2))
        elif args.command == 'probe':
            from .wsp_evidence import probe
            if args.inputs.stat().st_size > 1_000_000:
                raise ValueError('input fixture byte limit exceeded')
            result = probe(TensorArtifact.load(args.artifact), json.loads(args.inputs.read_text(encoding='utf-8')), semantics=args.semantics)
            _write_json(args.output, result)
            print(canonical(dict(status=result['status'], samples=result['samples'], universal_claim=False)))
        elif args.command == 'graphify':
            from .wsp_graph import to_dag
            parent=TensorArtifact.load(args.artifact);candidate=to_dag(parent);candidate.save(args.output)
            print(canonical(dict(parent_id=parent.sha256,candidate_id=candidate.sha256,tensor_storage_unchanged=True,admission='not_performed')))
        elif args.command == 'boundary':
            from .wsp_graph import region
            result=region(TensorArtifact.load(args.artifact),args.nodes)
            if args.output:_write_json(args.output,result)
            print(json.dumps(result,indent=2))
        else:
            from .wsp_evidence import Description, check
            result = check(TensorArtifact.load(args.artifact), Description.load(args.description))
            _write_json(args.output, result); print(canonical(dict(status=result['status'], scope=result['scope'])))
    except (ValueError, OSError, TypeError) as exc:
        print(canonical(dict(status='INVALID', reason=str(exc), parent_unchanged=True)))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
