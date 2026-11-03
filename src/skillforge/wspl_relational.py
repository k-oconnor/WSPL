"""Scoped QF_LRA relational contracts between a parent and an edited ReLU artifact (G3a).

Both stored graphs are encoded over one shared input point. Contracts are Boolean
combinations of affine comparisons over `input.*`, `parent.*` and `edited.*`
ports, e.g. preservation on an unchanged region or a relative repair bound on a
target region. Trusted solver/encoders; counterexamples are replayed through both
artifacts' exact rational evaluators. Not an IEEE execution or physical proof.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from .wspl_artifact import ReLUArtifact, canonical_bytes
from .wspl_verify import _evaluate, _parse_formula

CONTRACT_FORMAT = 'wspl.relational_contract.v0'
SUITE_FORMAT = 'wspl.relational_suite.v0'
STATUS_EXIT = {'PROVED': 0, 'REFUTED': 2, 'UNKNOWN': 3, 'INVALID': 1}


def _encode(z3, document, inputs, prefix, constraints):
    """Encode stored hidden-ReLU/final-linear layers with namespaced variables."""
    def real(q):
        return z3.RealVal(str(F(q)))
    previous = list(inputs)
    for layer_id, layer in enumerate(document['layers']):
        current = []
        for i, (weights, bias) in enumerate(zip(layer['weights'], layer['bias'])):
            s = sum((real(w) * x for w, x in zip(weights, previous)), real(bias))
            v = z3.Real(f'{prefix}_{layer_id}_{i}')
            constraints.append(v == (z3.If(s >= 0, s, real(0)) if layer['activation'] == 'relu' else s))
            current.append(v)
        previous = current
    return previous


def preservation_contract(parent, edited, *, name='preserve', region=None, outputs=None, tolerance='0'):
    """Contract: on `region` (input formula, default everywhere), each named output of
    `edited` equals the parent's output, or stays within +/- `tolerance`."""
    parent, edited = (a if isinstance(a, ReLUArtifact) else ReLUArtifact(a) for a in (parent, edited))
    units = {p['name']: p['type']['unit'] for p in parent.to_dict()['outputs']}
    keys = list(outputs) if outputs is not None else list(parent.output_names)
    tol = F(tolerance)
    if tol < 0:
        raise ValueError('tolerance must be nonnegative')
    clauses = []
    for key in keys:
        if key not in units:
            raise ValueError(f'unknown parent output {key!r}')
        diff = {'terms': {f'edited.{key}': '1', f'parent.{key}': '-1'}, 'bias': '0', 'unit': units[key]}
        if tol == 0:
            clauses.append({'op': 'eq', 'left': diff, 'right': {'terms': {}, 'bias': '0', 'unit': units[key]}})
        else:
            clauses.append({'op': 'le', 'left': diff, 'right': {'terms': {}, 'bias': str(tol), 'unit': units[key]}})
            clauses.append({'op': 'ge', 'left': diff, 'right': {'terms': {}, 'bias': str(-tol), 'unit': units[key]}})
    if not clauses:
        raise ValueError('at least one output required')
    return {'format': CONTRACT_FORMAT, 'name': name, 'parent_sha256': parent.sha256, 'edited_sha256': edited.sha256,
            'semantics': 'rational_real_graph', 'assumptions': region or {'op': 'true'},
            'ensures': clauses[0] if len(clauses) == 1 else {'op': 'and', 'args': clauses}}


def check_relation(parent, edited, contract, *, timeout_ms=5000, max_parameters=20000):
    started = time.monotonic()
    here = Path(__file__)
    result = dict(status='INVALID', scope='stored_weight_rational_real_graph_pair',
                  quantifier='forall shared inputs in the common artifact domain: assumptions imply ensures',
                  numerical_execution='not_established', physical_trajectory='not_established',
                  independently_checked_proof_trace=False,
                  trust=['contract parser', 'paired graph encoder', 'Z3', 'rational witness replay through both artifacts'],
                  timeout_ms=timeout_ms, logic='QF_LRA',
                  checker_source_sha256=hashlib.sha256(here.read_bytes()).hexdigest(),
                  evaluator_source_sha256=hashlib.sha256(here.with_name('wspl_artifact.py').read_bytes()).hexdigest())

    def finish(status, **fields):
        result.update(status=status, **fields)
        result['wall_seconds'] = time.monotonic() - started
        return result
    try:
        parent, edited = (a if isinstance(a, ReLUArtifact) else ReLUArtifact(a) for a in (parent, edited))
        p, e = parent.to_dict(), edited.to_dict()
        result.update(parent_sha256=parent.sha256, edited_sha256=edited.sha256)
        if (not isinstance(contract, dict) or
                set(contract) != {'format', 'name', 'parent_sha256', 'edited_sha256', 'semantics', 'assumptions', 'ensures'} or
                contract['format'] != CONTRACT_FORMAT or not isinstance(contract['name'], str) or not contract['name']):
            raise ValueError('invalid relational contract schema')
        result['name'] = contract['name']
        result['contract_sha256'] = hashlib.sha256(canonical_bytes(contract)).hexdigest()
        if contract['parent_sha256'] != parent.sha256 or contract['edited_sha256'] != edited.sha256:
            raise ValueError('contract artifact identities differ')
        if p['inputs'] != e['inputs']:
            raise ValueError('parent and edited input ports (names, order, types) differ')
        if contract['semantics'] != 'rational_real_graph':
            return finish('UNKNOWN', reason='unsupported numerical semantics')
        units = {'input.' + x['name']: x['type']['unit'] for x in p['inputs']}
        units.update({'parent.' + x['name']: x['type']['unit'] for x in p['outputs']})
        units.update({'edited.' + x['name']: x['type']['unit'] for x in e['outputs']})
        assumption = _parse_formula(contract['assumptions'], units)
        conclusion = _parse_formula(contract['ensures'], units)
        if type(timeout_ms) is not int or timeout_ms < 0 or type(max_parameters) is not int or max_parameters < 1:
            raise ValueError('invalid verification budget')
        if timeout_ms == 0:
            return finish('UNKNOWN', reason='budget_exhausted_before_query')
        if p['allocation']['parameters'] + e['allocation']['parameters'] > max_parameters:
            return finish('UNKNOWN', reason='verification capacity limit')
        try:
            import z3
        except ImportError:
            return finish('UNKNOWN', reason='optional Z3 backend unavailable')
        result.update(backend='Z3', backend_version=z3.get_version_string())

        def real(q):
            return z3.RealVal(str(F(q)))
        xs, base, ports = [], [], {}
        for i, port in enumerate(p['inputs']):
            x = z3.Real(f'input_{i}')
            xs.append(x)
            ports['input.' + port['name']] = x
            base += [x >= real(port['type']['lower']), x <= real(port['type']['upper'])]
        for prefix, d in (('parent', p), ('edited', e)):
            for port, value in zip(d['outputs'], _encode(z3, d, xs, prefix, base)):
                ports[f'{prefix}.{port["name"]}'] = value

        def encode(formula):
            op = formula[0]
            if op in ('true', 'false'):
                return z3.BoolVal(op == 'true')
            if op in ('le', 'lt', 'ge', 'gt', 'eq'):
                a, b = (sum((real(c) * ports[k] for k, c in side[0]), real(side[1])) for side in formula[1:])
                return {'le': a <= b, 'lt': a < b, 'ge': a >= b, 'gt': a > b, 'eq': a == b}[op]
            args = [encode(x) for x in formula[1:]]
            return {'and': z3.And, 'or': z3.Or}[op](*args) if op in ('and', 'or') else (
                z3.Not(args[0]) if op == 'not' else z3.Implies(*args))

        def query(extra):
            remaining = timeout_ms - int(1000 * (time.monotonic() - started))
            if remaining <= 0:
                return None, 'unknown', 'total query budget exhausted'
            solver = z3.SolverFor('QF_LRA')
            solver.set(timeout=remaining)
            solver.add(*base, *extra)
            status = solver.check()
            return solver, str(status), solver.reason_unknown() if status == z3.unknown else None

        def witness(solver):
            model = solver.model()
            values = {}
            for port, x in zip(p['inputs'], xs):
                v = model.eval(x, model_completion=True)
                if not z3.is_rational_value(v):
                    raise ValueError('nonrational witness unsupported')
                values[port['name']] = F(v.numerator_as_long(), v.denominator_as_long())
            parent_out, edited_out = parent.evaluate(values), edited.evaluate(values)
            replay = {**{'input.' + k: v for k, v in values.items()},
                      **{'parent.' + k: v for k, v in parent_out.items()},
                      **{'edited.' + k: v for k, v in edited_out.items()}}
            return values, parent_out, edited_out, replay

        solver, domain, reason = query([encode(assumption)])
        result['domain_query'] = domain
        if solver is not None:
            result['domain_query_sha256'] = hashlib.sha256(solver.sexpr().encode()).hexdigest()
        if domain == 'unknown':
            return finish('UNKNOWN', reason=reason)
        if domain == 'unsat':
            return finish('INVALID', reason='vacuous assumption domain', nonvacuous=False)
        values, _, _, replay = witness(solver)
        if not _evaluate(assumption, replay):
            return finish('UNKNOWN', reason='nonvacuity witness replay failed')
        result.update(nonvacuous=True, domain_witness={k: str(v) for k, v in values.items()})
        solver, violation, reason = query([encode(assumption), z3.Not(encode(conclusion))])
        result['violation_query'] = violation
        if solver is not None:
            result['violation_query_sha256'] = hashlib.sha256(solver.sexpr().encode()).hexdigest()
        if violation == 'unknown':
            return finish('UNKNOWN', reason=reason)
        if violation == 'unsat':
            return finish('PROVED', reason='violation query UNSAT under declared paired real-graph semantics')
        values, parent_out, edited_out, replay = witness(solver)
        if not _evaluate(assumption, replay) or _evaluate(conclusion, replay):
            return finish('UNKNOWN', reason='violating witness replay failed')
        return finish('REFUTED', reason='SAT violation independently replayed', witness_validated=True,
                      counterexample={'inputs': {k: str(v) for k, v in values.items()},
                                      'parent_outputs': {k: str(v) for k, v in parent_out.items()},
                                      'edited_outputs': {k: str(v) for k, v in edited_out.items()}})
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
        return finish('INVALID', reason=str(exc))
    except Exception as exc:
        return finish('UNKNOWN', reason=f'backend/encoding failure: {type(exc).__name__}: {exc}')


def check_relational_suite(parent, edited, suite, **kwargs):
    if (not isinstance(suite, dict) or set(suite) != {'format', 'parent_sha256', 'edited_sha256', 'contracts'} or
            suite['format'] != SUITE_FORMAT or not isinstance(suite['contracts'], list) or not suite['contracts']):
        return dict(status='INVALID', reason='invalid relational suite', results=[])
    results = []
    for c in suite['contracts']:
        if not isinstance(c, dict) or set(c) != {'name', 'assumptions', 'ensures'}:
            return dict(status='INVALID', reason='invalid suite member', results=results)
        results.append(check_relation(parent, edited, dict(
            format=CONTRACT_FORMAT, semantics='rational_real_graph', parent_sha256=suite['parent_sha256'],
            edited_sha256=suite['edited_sha256'], **c), **kwargs))
    status = next((s for s in ['INVALID', 'REFUTED', 'UNKNOWN'] if any(r['status'] == s for r in results)), 'PROVED')
    return dict(status=status, scope='stored_weight_rational_real_graph_pair', results=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('parent', type=Path)
    p.add_argument('edited', type=Path)
    p.add_argument('suite', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout-ms', type=int, default=5000)
    args = p.parse_args()
    try:
        result = check_relational_suite(ReLUArtifact.load(args.parent), ReLUArtifact.load(args.edited),
                                        json.loads(args.suite.read_text(encoding='utf-8')), timeout_ms=args.timeout_ms)
        with args.output.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write('\n')
        print(json.dumps({'status': result['status'], 'output': str(args.output)}))
        raise SystemExit(STATUS_EXIT[result['status']])
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
