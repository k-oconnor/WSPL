"""Finite-domain authoring experiment kernel. No provider calls or proposal exec.

The kernel enforces candidate/feedback boundaries, not OS/process isolation.
Tasks describe supplied mechanisms; this is authoring, not causal discovery.
"""
import ast
import copy
from fractions import Fraction as Q
import hashlib
import itertools
import json
import random
import time

from .wspl_artifact import ReLUArtifact, canonical_bytes
from .wspl_compile import compile_scalar
from .wspl_core import rational
from .wspl_parse import parse_module

ARMS = ('weights_off', 'weights_on', 'code_off', 'code_on', 'wspl_off', 'wspl_on')
INPUTS = ('switch_a', 'switch_b', 'power', 'lock')
HIDDEN_WIDTH = 12
WIDTHS = (4, 12, 12, 12, 12, 1)
PARAMETERS = sum((a + 1) * b for a, b in zip(WIDTHS, WIDTHS[1:]))
TYPE = dict(kind='Real', unit='1', lower='0', upper='1')


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def mechanism(seed):
    """Two distinct two-literal conjunctions; independent Boolean task oracle."""
    rng = random.Random(seed)
    clauses = []
    while len(clauses) < 2:
        clause = sorted([(i, rng.choice([False, True])) for i in rng.sample(range(4), 2)])
        if clause not in clauses:
            clauses.append(clause)
    return dict(format='wspl.boolean_mechanism.v0', inputs=list(INPUTS),
                clauses=clauses, output='next_open',
                domain='all 16 Boolean input assignments; no continuous-domain claim')


def target(task, inputs):
    return int(any(all(bool(inputs[INPUTS[i]]) == positive for i, positive in clause)
                   for clause in task['clauses']))


def cases():
    return [dict(zip(INPUTS, bits)) for bits in itertools.product((0, 1), repeat=4)]


def ref(key):
    return dict(op='ref', name=key)


def affine(terms, bias=0):
    return dict(op='affine', terms=[dict(coefficient=str(c), value=v) for c, v in terms],
                bias=str(bias), unit='1')


def program(expression):
    return dict(format='wspl.scalar_ir.v0', inputs={k: dict(TYPE) for k in INPUTS},
                lets=[], outputs=[dict(name='next_open', type=dict(TYPE), value=expression)])


def reference_program(task):
    """Feasibility/test helper. Must never be included in proposer requests."""
    conjunctions = []
    for clause in task['clauses']:
        literals = [ref(INPUTS[i]) if positive else affine([(-1, ref(INPUTS[i]))], 1)
                    for i, positive in clause]
        conjunctions.append(dict(op='relu', value=affine([(1, x) for x in literals], 1-len(literals))))
    return program(dict(op='clip', value=affine([(1, x) for x in conjunctions]), lower='0', upper='1'))


def parse_code(text):
    """Parse one ordinary PWA Python expression; never compile/eval/exec it."""
    if not isinstance(text, str) or len(text) > 12000:
        raise ValueError('one bounded expression required')
    tree = ast.parse(text, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise ValueError('code expression too large')
    def constant(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Q(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return constant(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return constant(node.left) / constant(node.right)
        raise ValueError('numeric coefficient must be integer or rational constant')
    def visit(node):
        if isinstance(node, ast.Name) and node.id in INPUTS:
            return ref(node.id)
        try:
            return dict(op='const', value=str(constant(node)), unit='1')
        except ValueError:
            pass
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return affine([(-1 if isinstance(node.op, ast.USub) else 1, visit(node.operand))])
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return affine([(1, visit(node.left)), (-1 if isinstance(node.op, ast.Sub) else 1, visit(node.right))])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            try:
                return affine([(constant(node.left), visit(node.right))])
            except ValueError:
                return affine([(constant(node.right), visit(node.left))])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return affine([(1/constant(node.right), visit(node.left))])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            if node.func.id == 'clip' and len(node.args) == 3:
                return dict(op='clip', value=visit(node.args[0]), lower=str(constant(node.args[1])), upper=str(constant(node.args[2])))
            if node.func.id in ('min', 'max') and len(node.args) == 2:
                return dict(op=node.func.id, left=visit(node.args[0]), right=visit(node.args[1]))
        raise ValueError('unsupported expression; no imports, attributes, indexing, loops or Python execution')
    return program(visit(tree.body))


def fixed_artifact(artifact):
    """Zero pad nonnegative hidden states and append identity ReLU layers."""
    d = artifact.to_dict()
    hidden = d['layers'][:-1]
    if len(hidden) > 4 or any(len(layer['bias']) > HIDDEN_WIDTH for layer in hidden):
        raise ValueError('compiled program exceeds fixed 4x12 target')
    result = []
    previous = 4
    for layer in hidden:
        rows = [row + ['0'] * (previous - len(row)) for row in layer['weights']]
        rows += [['0'] * previous for _ in range(HIDDEN_WIDTH-len(rows))]
        result.append(dict(activation='relu', weights=rows, bias=layer['bias'] + ['0'] * (HIDDEN_WIDTH-len(layer['bias']))))
        previous = HIDDEN_WIDTH
    # A graph with no hidden state needs signed input relays; do not silently
    # clip it. All task inputs are nonnegative, so identity relays are valid here.
    while len(result) < 4:
        result.append(dict(activation='relu', weights=[['1' if i == j else '0' for j in range(previous)] for i in range(HIDDEN_WIDTH)], bias=['0']*HIDDEN_WIDTH))
        previous = HIDDEN_WIDTH
    last = d['layers'][-1]
    result.append(dict(activation='linear', weights=[row + ['0']*(HIDDEN_WIDTH-len(row)) for row in last['weights']], bias=last['bias']))
    d.update(layers=result, allocation=dict(mode='fresh_mlp', widths=list(WIDTHS), parameters=PARAMETERS))
    # Original source map refers to pre-padding allocation, not final neurons.
    d['source_map'] = dict(original_compilation=d['source_map'], padding='fixed nonnegative hidden relays')
    return ReLUArtifact(d)


def sparse_artifact(candidate):
    if not isinstance(candidate, dict) or set(candidate) != {'format', 'layers'} or candidate['format'] != 'wspl.sparse_weights.v0' or not isinstance(candidate['layers'], list) or len(candidate['layers']) != 5:
        raise ValueError('five explicit sparse weight/bias layers required')
    layers = []
    for index, (a, b, layer) in enumerate(zip(WIDTHS, WIDTHS[1:], candidate['layers'])):
        if set(layer) != {'rows', 'bias'} or len(layer['rows']) != b or len(layer['bias']) != b:
            raise ValueError('fixed layer shape differs')
        rows = []
        for sparse in layer['rows']:
            if not isinstance(sparse, dict) or any(str(int(k)) != k or not 0 <= int(k) < a for k in sparse):
                raise ValueError('invalid sparse input index')
            rows.append([str(rational(sparse.get(str(i), '0'))) for i in range(a)])
        layers.append(dict(activation='linear' if index == 4 else 'relu', weights=rows,
                           bias=[str(rational(x)) for x in layer['bias']]))
    return ReLUArtifact(dict(format='wspl.relu_mlp.v0', semantics='rational_real_graph',
                             compiler='authoring.sparse_weights.v0', source_sha256=digest(candidate),
                             inputs=[dict(name=k, type=dict(TYPE)) for k in INPUTS],
                             outputs=[dict(name='next_open', type=dict(TYPE))], layers=layers,
                             allocation=dict(mode='fresh_mlp', widths=list(WIDTHS), parameters=PARAMETERS), source_map={}))


def candidate_program(representation, candidate):
    """Lower submitted source; study requirements are never candidate-owned."""
    if representation == 'code':
        return parse_code(candidate)
    if representation == 'wspl':
        if not isinstance(candidate, str):
            raise ValueError('WSPL arm requires module text, not JSON IR')
        parsed = parse_module(candidate)
        if parsed.contracts:
            raise ValueError('study requirements are fixed externally; omit ensures/preserves declarations')
        return parsed.program
    raise ValueError('representation has no scalar source')


def build_candidate(representation, candidate):
    if len(canonical_bytes(candidate)) > 24000:
        raise ValueError('candidate exceeds 24000 byte limit')
    if representation == 'weights':
        artifact = sparse_artifact(candidate)
    elif representation in ('code', 'wspl'):
        p = candidate_program(representation, candidate)
        if not isinstance(p, dict) or p.get('inputs') != {k: dict(TYPE) for k in INPUTS} or tuple(p['inputs']) != INPUTS or len(p.get('outputs', [])) != 1 or p['outputs'][0]['name'] != 'next_open' or p['outputs'][0]['type'] != TYPE:
            raise ValueError('candidate cannot alter task ports, units or declared bounds')
        artifact = fixed_artifact(compile_scalar(p))
    else:
        raise ValueError('unknown representation')
    for layer in artifact.to_dict()['layers']:
        for x in layer['bias'] + [x for row in layer['weights'] for x in row]:
            v = rational(x)
            if abs(v) > 64 or v.denominator > 1024:
                raise ValueError('common stored coefficient limits exceeded')
    return artifact


def grade(task, artifact, rows):
    if not rows:
        raise ValueError('empty evaluation domain is not a proof')
    failures = []
    for x in rows:
        actual = artifact.evaluate(x)['next_open']
        expected = target(task, x)
        if actual != expected:
            failures.append(dict(inputs=x, expected=str(expected), actual=str(actual)))
    return dict(status='PROVED' if not failures else 'REFUTED',
                scope='exact enumeration of supplied finite Boolean domain only',
                nonvacuous=bool(rows), evaluated=len(rows), correct=len(rows)-len(failures),
                first_counterexample=failures[0] if failures else None)


class AuthoringTrial:
    def __init__(self, task, arm, *, trial_id, model_id, max_tokens=65536, max_candidates=8):
        if arm not in ARMS or not trial_id or not model_id:
            raise ValueError('declared arm, trial and model identities required')
        if type(max_tokens) is not int or type(max_candidates) is not int or max_tokens < 1 or not 1 <= max_candidates <= 8:
            raise ValueError('positive token budget and one to eight candidates required')
        self.task = copy.deepcopy(task)
        self.arm, self.trial_id, self.model_id = arm, trial_id, model_id
        self.max_tokens, self.max_candidates = max_tokens, max_candidates
        self.ledger, self.artifacts = [], {}
        self.tokens = 0
        self.closed = False
        # Four shared development states; the other twelve are withheld in feedback-off.
        self.development = [cases()[i] for i in (0, 5, 10, 15)]

    def request(self):
        representation = self.arm.split('_')[0]
        syntax = {
            'weights': 'JSON {format: wspl.sparse_weights.v0, layers: [five layers]}; each layer has rows (one dict per output neuron mapping zero-based input-index strings to exact rational coefficient strings) and bias (one rational string per output neuron). Omitted matrix entries are zero. Four ReLU hidden layers, linear output. No code/scripts execute.',
            'code': 'One Python-style expression in the four named inputs. Use integer constants or constant fractions, +, -, multiplication/division by constants, min(a,b), max(a,b), clip(value,lower,upper). No imports, attributes, indexing, loops, conditionals or variable products. Expression is parsed, not executed.',
            'wspl': 'One WSPL Core PWA module as a text string, not JSON IR. Declare inputs in the supplied order as Real[0,1] normalized_value; declare output next_open: Real[0,1] normalized_value; then assign next_open. Typed lets are allowed. Omit ensures/preserves: study requirements are fixed externally. See PARSER.md and SCALAR_CORE.md. Text is parsed, never executed.',
        }[representation]
        return copy.deepcopy(dict(format='wspl.authoring_request.v1', trial_id=self.trial_id,
                                  model_id=self.model_id, arm=self.arm, task=self.task,
                                  widths=list(WIDTHS), parameters=PARAMETERS,
                                  development_inputs=self.development,
                                  max_candidates=self.max_candidates, max_total_tokens=self.max_tokens,
                                  syntax=syntax, candidate_max_bytes=24000,
                                  coefficient_limits=dict(max_absolute=64, max_denominator=1024),
                                  input_type=TYPE, output_type=TYPE,
                                  rule_semantics='OR of clauses; each clause is AND of [input_index, positive_bool] literals. False polarity means NOT input.',
                                  output='next_open must exactly equal the supplied Boolean rule',
                                  usage='record all input/output tokens, elapsed seconds and money, including failures'))

    def submit(self, candidate, usage):
        if self.closed or len(self.ledger) >= self.max_candidates:
            raise ValueError('trial closed or candidate budget exhausted')
        if not isinstance(usage, dict) or set(usage) != {'input_tokens', 'output_tokens', 'elapsed_seconds', 'cost'}:
            raise ValueError('complete externally reported usage required')
        if any(type(usage[k]) is not int or usage[k] < 0 for k in ('input_tokens', 'output_tokens')):
            raise ValueError('nonnegative token counts required')
        if any(type(usage[k]) not in (float, int) or not 0 <= usage[k] < 1e9 for k in ('elapsed_seconds', 'cost')):
            raise ValueError('finite nonnegative costs required')
        charge = usage['input_tokens'] + usage['output_tokens']
        self.tokens += charge
        index = len(self.ledger)
        entry = dict(candidate_id=index, candidate=copy.deepcopy(candidate), usage=copy.deepcopy(usage),
                     usage_provenance='external report; not independently billing verified', candidate_sha256=digest(candidate))
        started = time.monotonic()
        if self.tokens > self.max_tokens:
            entry['feedback'] = dict(status='BUDGET_EXCEEDED')
            self.closed = True
        else:
            try:
                artifact = build_candidate(self.arm.split('_')[0], candidate)
                self.artifacts[index] = artifact
                feedback = dict(status='VALID', development=grade(self.task, artifact, self.development))
                if self.arm.endswith('_on'):
                    feedback['verification'] = grade(self.task, artifact, cases())
                entry.update(artifact_sha256=artifact.sha256, feedback=feedback)
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, SyntaxError, ZeroDivisionError, RecursionError) as exc:
                entry['feedback'] = dict(status='INVALID', reason=f'{type(exc).__name__}: {exc}')
        entry['evaluator_seconds'] = time.monotonic()-started
        self.ledger.append(entry)
        return copy.deepcopy(dict(candidate_id=index, **entry['feedback']))

    def finalize(self, candidate_id=None):
        # One final evaluator call. Explicit invalid selection never falls back.
        if getattr(self, '_finalized', False):
            raise ValueError('final evaluation is one-shot')
        if candidate_id is not None and (type(candidate_id) is not int or candidate_id < 0):
            raise ValueError('final candidate must be a nonnegative integer ID')
        self.closed = True
        self._finalized = True
        if candidate_id is None:
            candidate_id = max(self.artifacts, default=None)
        artifact = self.artifacts.get(candidate_id)
        result = dict(format='wspl.authoring_result.v0', trial_id=self.trial_id, arm=self.arm,
                      model_id=self.model_id, request_sha256=digest(self.request()),
                      selected_candidate=candidate_id, total_tokens=self.tokens,
                      costs={k: sum(e['usage'][k] for e in self.ledger) for k in ('elapsed_seconds', 'cost')},
                      candidates=len(self.ledger), qualification=False)
        if artifact is None:
            result['verification'] = dict(status='INVALID', reason='no valid selected candidate')
            result['source_equivalence'] = dict(status='NOT_APPLICABLE', reason='no selected valid artifact')
        else:
            result['verification'] = grade(self.task, artifact, cases())
            result['non_development_inputs'] = grade(self.task, artifact, [x for x in cases() if x not in self.development])
            result['qualification'] = result['verification']['status'] == 'PROVED' and self.tokens <= self.max_tokens
            result['artifact_sha256'] = artifact.sha256
            if self.arm.startswith('weights_'):
                result['source_equivalence'] = dict(status='NOT_APPLICABLE', reason='direct weights have no submitted scalar source')
            else:
                # Final-only diagnostic: never expose this through search feedback.
                # Graph task correctness remains the common primary endpoint.
                from .wspl_equiv import check_equivalence
                candidate = self.ledger[candidate_id]['candidate']
                source = candidate_program(self.arm.split('_')[0], candidate)
                result['source_equivalence'] = check_equivalence(source, artifact, timeout_ms=5000)
        result['evaluation_seconds_before_final'] = sum(e['evaluator_seconds'] for e in self.ledger)
        return result
