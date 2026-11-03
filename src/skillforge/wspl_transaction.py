"""Transactional output-patch edits with commit or rollback (G3b).

Compose a patch onto a pinned parent, then check three kinds of contract:
generated interface obligations on the parent (parent outputs stay inside the
patch inputs' declared bounds), user I/O contracts on the edited network and
user relational parent/edited contracts. The decision is

- COMMITTED: every result PROVED; the edited artifact is returned.
- REJECTED: any result REFUTED or INVALID, or the composition itself is invalid.
- UNRESOLVED: nothing refuted or invalid, but some result UNKNOWN.

Both non-commit outcomes return the unchanged parent (hash re-checked) and keep
the failed candidate's hash and every result in the manifest. Whether a commit
requires any behavioural contract is policy left to the caller; the manifest
records how many contracts of each kind ran. Rational real-graph claims only.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from .wspl_artifact import ReLUArtifact, canonical_bytes
from .wspl_compose import compose_output, composition_record, interface_obligations, verify_placement
from .wspl_relational import CONTRACT_FORMAT as RELATIONAL_FORMAT, check_relation
from .wspl_verify import verify_contract

MANIFEST_FORMAT = 'wspl.patch_transaction.v0'
SPEC_FORMAT = 'wspl.patch_spec.v0'
DECISION_RULE = 'COMMITTED iff every result PROVED; REJECTED if any REFUTED or INVALID; otherwise UNRESOLVED'
EXIT = {'COMMITTED': 0, 'REJECTED': 2, 'UNRESOLVED': 3}


@dataclass
class TransactionResult:
    decision: str
    artifact: ReLUArtifact   # edited if COMMITTED, otherwise the unchanged parent
    candidate: object        # composed ReLUArtifact, or None if composition failed
    manifest: dict


def _templates(contracts, kind):
    out = []
    for c in contracts:
        if not isinstance(c, dict) or set(c) != {'name', 'assumptions', 'ensures'}:
            raise ValueError(f'{kind} contracts need exactly name, assumptions and ensures')
        out.append({'name': c['name'], 'assumptions': c['assumptions'], 'ensures': c['ensures']})
    return out


def _bind_io(template, artifact):
    return {'format': 'wspl.io_contract.v0', 'semantics': 'rational_real_graph', 'artifact_sha256': artifact.sha256, **template}


def _bind_relational(template, parent, edited):
    return {'format': RELATIONAL_FORMAT, 'semantics': 'rational_real_graph',
            'parent_sha256': parent.sha256, 'edited_sha256': edited.sha256, **template}


def _sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _results_digest(results):
    """Hash of each result's kind, name, status and contract hash (timings excluded)."""
    return _sha([[r['kind'], r.get('name'), r['status'], r.get('contract_sha256')] for r in results])


def decide(statuses):
    if any(s in ('REFUTED', 'INVALID') for s in statuses):
        return 'REJECTED'
    if any(s != 'PROVED' for s in statuses):
        return 'UNRESOLVED'
    return 'COMMITTED'


def patch_transaction(parent, patch, *, bindings, replace, io_contracts=(), relational_contracts=(), timeout_ms=5000):
    parent = parent if isinstance(parent, ReLUArtifact) else ReLUArtifact(parent)
    patch = patch if isinstance(patch, ReLUArtifact) else ReLUArtifact(patch)
    parent_sha = parent.sha256
    io_templates = _templates(io_contracts, 'I/O')
    relational_templates = _templates(relational_contracts, 'relational')
    manifest = {'format': MANIFEST_FORMAT, 'operation': 'compose_output', 'label': 'output-patch augmentation',
                'parent_sha256': parent_sha, 'patch_sha256': patch.sha256,
                'bindings': dict(sorted(bindings.items())) if isinstance(bindings, dict) else bindings,
                'replace': dict(sorted(replace.items())) if isinstance(replace, dict) else replace,
                'contracts': {'io': io_templates, 'relational': relational_templates},
                'decision_rule': DECISION_RULE, 'timeout_ms': timeout_ms,
                'numerical_execution': 'not_established', 'physical_trajectory': 'not_established'}

    def finish(decision, candidate, results, **fields):
        if parent.sha256 != parent_sha:
            raise RuntimeError('parent artifact changed during the transaction')
        returned = candidate if decision == 'COMMITTED' else parent
        manifest.update(decision=decision, results=results, parent_unchanged=True,
                        returned_sha256=returned.sha256, results_digest_sha256=_results_digest(results), **fields)
        manifest['manifest_body_sha256'] = _sha({k: v for k, v in manifest.items() if k != 'results'})
        return TransactionResult(decision, returned, candidate, manifest)

    try:
        record = composition_record(parent, patch, bindings, replace)
        candidate = compose_output(parent, patch, bindings=bindings, replace=replace)
        rows_checked = verify_placement(candidate, parent, patch)
    except ValueError as exc:
        return finish('REJECTED', None, [], candidate_sha256=None, reason=f'composition invalid: {exc}',
                      contract_counts={'interface': 0, 'io': len(io_templates), 'relational': len(relational_templates)})
    obligations = interface_obligations(parent, patch, bindings)
    manifest['contracts']['interface'] = obligations
    results = []
    for template in obligations:
        results.append({'kind': 'interface', **verify_contract(parent, _bind_io(template, parent), timeout_ms=timeout_ms)})
    for template in io_templates:
        results.append({'kind': 'io', **verify_contract(candidate, _bind_io(template, candidate), timeout_ms=timeout_ms)})
    for template in relational_templates:
        results.append({'kind': 'relational', **check_relation(
            parent, candidate, _bind_relational(template, parent, candidate), timeout_ms=timeout_ms)})
    composition = candidate.to_dict()['source_map']['composition']
    return finish(decide([r['status'] for r in results]), candidate, results,
                  candidate_sha256=candidate.sha256, composition_sha256=_sha(record),
                  candidate_allocation=candidate.to_dict()['allocation'],
                  added_depth=composition['added_depth'], added_parameters=composition['added_parameters'],
                  placement_rows_checked=rows_checked,
                  contract_counts={'interface': len(obligations), 'io': len(io_templates),
                                   'relational': len(relational_templates)})


def replay_transaction(manifest, parent, patch, *, rerun=False):
    """Recompose from the manifest and check it still describes these artifacts.

    Raises ValueError if the manifest is stale or altered. With rerun=True the
    contracts are checked again and their statuses must match the record.
    """
    if not isinstance(manifest, dict) or manifest.get('format') != MANIFEST_FORMAT:
        raise ValueError('patch transaction manifest required')
    body = {k: v for k, v in manifest.items() if k not in ('results', 'manifest_body_sha256')}
    if _sha(body) != manifest.get('manifest_body_sha256'):
        raise ValueError('manifest body hash differs; the record was altered')
    if _results_digest(manifest.get('results', [])) != manifest.get('results_digest_sha256'):
        raise ValueError('recorded results differ from their digest; the record was altered')
    parent = parent if isinstance(parent, ReLUArtifact) else ReLUArtifact(parent)
    patch = patch if isinstance(patch, ReLUArtifact) else ReLUArtifact(patch)
    if parent.sha256 != manifest['parent_sha256'] or patch.sha256 != manifest['patch_sha256']:
        raise ValueError('parent or patch artifact differs from the manifest')
    again = patch_transaction(parent, patch, bindings=manifest['bindings'], replace=manifest['replace'],
                              io_contracts=manifest['contracts']['io'],
                              relational_contracts=manifest['contracts']['relational'],
                              timeout_ms=manifest['timeout_ms'] if rerun else 0)
    if again.manifest['candidate_sha256'] != manifest['candidate_sha256']:
        raise ValueError('recomposed candidate differs from the manifest')
    recorded = [(r['kind'], r.get('name'), r.get('contract_sha256')) for r in manifest['results']]
    replayed = [(r['kind'], r.get('name'), r.get('contract_sha256')) for r in again.manifest['results']]
    if recorded != replayed:
        raise ValueError('bound contracts differ from the manifest')
    if rerun and [r['status'] for r in again.manifest['results']] != [r['status'] for r in manifest['results']]:
        raise ValueError('re-run contract statuses differ from the manifest')
    return {'consistent': True, 'candidate_sha256': manifest['candidate_sha256'], 'rerun': rerun,
            'decision': again.manifest['decision'] if rerun else manifest['decision']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('parent', type=Path)
    p.add_argument('patch', type=Path)
    p.add_argument('spec', type=Path, help=f'{SPEC_FORMAT} JSON with bindings, replace and contracts')
    p.add_argument('--manifest-output', type=Path, required=True)
    p.add_argument('--artifact-output', type=Path, help='written only if COMMITTED')
    p.add_argument('--timeout-ms', type=int, default=5000)
    args = p.parse_args()
    try:
        spec = json.loads(args.spec.read_text(encoding='utf-8'))
        if not isinstance(spec, dict) or spec.get('format') != SPEC_FORMAT or set(spec) - {
                'format', 'bindings', 'replace', 'io_contracts', 'relational_contracts'}:
            raise ValueError(f'{SPEC_FORMAT} spec with bindings, replace, io_contracts, relational_contracts required')
        result = patch_transaction(ReLUArtifact.load(args.parent), ReLUArtifact.load(args.patch),
                                   bindings=spec['bindings'], replace=spec['replace'],
                                   io_contracts=spec.get('io_contracts', []),
                                   relational_contracts=spec.get('relational_contracts', []), timeout_ms=args.timeout_ms)
        with args.manifest_output.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(result.manifest, f, indent=2, allow_nan=False)
            f.write('\n')
        if args.artifact_output and result.decision == 'COMMITTED':
            result.artifact.save(args.artifact_output)
        print(json.dumps({'decision': result.decision, 'manifest': str(args.manifest_output),
                          'returned_sha256': result.manifest['returned_sha256']}))
        raise SystemExit(EXIT[result.decision])
    except (OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
