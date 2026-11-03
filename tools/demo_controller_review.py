"""Known-answer controller change review. No training, model calls or robot use."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from skillforge.wspl_artifact import ReLUArtifact
from skillforge.wspl_compile import compile_scalar
from skillforge.wspl_verify import verify_suite


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run_demo(output, *, timeout_ms=5000, include_opaque=False):
    """Create a new review directory; leave partial evidence intact on failure."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads((ROOT / 'examples/wspl/brake_patch.core.json').read_text())
    suite = json.loads((ROOT / 'examples/wspl/brake.contracts.json').read_text())
    original = compile_scalar(source)
    # A declared fault injection, not an LLM proposal or a learned repair.
    broken_source = json.loads(json.dumps(source))
    broken_source['outputs'][0]['value']['value']['terms'][1]['coefficient'] = '-1'
    broken = compile_scalar(broken_source)
    # The operator supplies the correction explicitly: restore gain -2.
    repaired = compile_scalar(source)
    stages = []
    for name, artifact, program in (
        ('original', original, source),
        ('reduced_braking_gain', broken, broken_source),
        ('operator_repaired', repaired, source),
    ):
        bound_suite = {**suite, 'artifact_sha256': artifact.sha256}
        result = verify_suite(artifact, bound_suite, timeout_ms=timeout_ms)
        artifact.save(output / f'{name}.weights.json')
        write_json(output / f'{name}.source.json', program)
        write_json(output / f'{name}.contracts.json', bound_suite)
        write_json(output / f'{name}.verification.json', result)
        stages.append(dict(name=name, artifact_sha256=artifact.sha256,
                           status=result['status'],
                           counterexamples=[r['counterexample'] for r in result['results']
                                            if r.get('witness_validated')]))
    opaque = dict(status='NOT_REQUESTED')
    if include_opaque:
        import torch
        from skillforge.interventions import (
            Intervention, audit_intervention, append_evidence,
            replay_intervention, verify_log,
        )
        # fork_rng protects the caller from random constructor initialization.
        with torch.random.fork_rng(devices=[]):
            model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False),
                                        torch.nn.Tanh(), torch.nn.Linear(2, 1, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.eye(2))
            model[2].weight.copy_(torch.tensor([[1., 0.]]))
        inputs = torch.tensor([[1., 2.]])
        evidence = []
        head = None
        for feature in (0, 1):
            record = audit_intervention(model, inputs, Intervention('1', features=(feature,)),
                                        runtime_id='controller-review-known-answer-v1',
                                        declared_stateless=True)
            head = append_evidence(output / 'opaque_interventions.jsonl', record)
            evidence.append(record)
        replay = replay_intervention(model, inputs, evidence[0])
        verify_log(output / 'opaque_interventions.jsonl', expected_head=head)
        opaque = dict(status='ENGINEERING_FIXTURE',
                      model='Linear identity -> Tanh -> Linear readout [1,0]',
                      input_values=[[1., 2.]],
                      outcomes=[r['status'] for r in evidence],
                      head_sha256=head, exact_replay=replay['exact_replay'],
                      scope='empirical computational intervention; no semantic or physical claim')
    expected = [s['status'] for s in stages] == ['PROVED', 'REFUTED', 'PROVED']
    expected = expected and bool(stages[1]['counterexamples'])
    if include_opaque:
        expected = expected and opaque['outcomes'] == ['OBSERVED_EFFECT', 'NO_OBSERVED_EFFECT'] and opaque['exact_replay']
    report = dict(format='wspl.controller_review_demo.v0',
                  status='EXPECTED_FIXTURE_OUTCOMES' if expected else 'INCOMPLETE_OR_UNEXPECTED',
                  scope='inspected known-answer engineering fixture',
                  repair_method='operator-supplied original program; no automatic diagnosis or repair',
                  proof_scope='stored-weight rational-real I/O graph only',
                  source_equivalence='not_proved', numerical_execution='not_proved',
                  physical_safety='not_established', scientific_run=False,
                  stages=stages, opaque=opaque)
    write_json(output / 'review.json', report)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / 'manifest.json', dict(
        format='wspl.demo_bundle_manifest.v0', files=files,
        trust='unsigned integrity inventory; not authenticated proof or research source approval'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-ms', type=int, default=5000)
    parser.add_argument('--include-opaque', action='store_true')
    args = parser.parse_args()
    try:
        report = run_demo(args.output, timeout_ms=args.timeout_ms, include_opaque=args.include_opaque)
        print(json.dumps(dict(status=report['status'], output=str(args.output),
                              stages=[s['status'] for s in report['stages']],
                              opaque=report['opaque']['status'])))
        return 0 if report['status'] == 'EXPECTED_FIXTURE_OUTCOMES' else 2
    except (OSError, ValueError, ImportError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
