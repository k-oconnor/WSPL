"""Known-answer graph surgery fixture; no training, authoring or scored episodes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from skillforge.wsp_artifact import TensorArtifact, diff
from skillforge.wsp_editor import plan, apply
from skillforge.wsp_evidence import describe, check, probe
from skillforge.wsp_graph import GraphBudget, region


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    examples = ROOT / 'examples/wsp_graph'
    parent = TensorArtifact.load(examples / 'parent.json')
    patch = TensorArtifact.load(examples / 'patch.json')
    original = parent.to_dict()
    budget = GraphBudget(**json.loads((examples / 'budget.json').read_text(encoding='utf-8')))
    results = {}
    for name, formula in [('replace', '5*x[0]'), ('residual', '11/2*x[0]+1'), ('untie', '7*x[0]+1')]:
        edit = plan((examples / f'{name}.wspl').read_text(encoding='utf-8'), parent=parent,
                    artifacts={'patch': patch}, budgets={'repair': budget})
        candidate = apply(edit, parent=parent, expected_parent_id=parent.sha256)
        description = describe(candidate, '''describe candidate {
          claim intended_change semantics=real_graph {
            assume x[0] in [-4,4]; ensure y[0]==''' + formula + ''';
          }
          claim preserved_branch semantics=real_graph {
            assume x[0] in [-4,4]; ensure keep[0]==2*x[0];
          }
        }''')
        checked = check(candidate, description)
        if checked['status'] != 'PROVED':
            raise AssertionError(f'{name}: known-answer claims did not prove')
        delta = diff(parent, candidate)
        if 'right' not in delta['graph']['unaffected_dependency_nodes']:
            raise AssertionError('untouched branch dependency changed')
        for tensor in ('W', 'rb'):
            if original['tensors'][tensor] != candidate.to_dict()['tensors'][tensor]:
                raise AssertionError('untouched branch storage changed')
        false_description = describe(candidate, '''describe candidate {
          claim deliberately_wrong semantics=real_graph {
            assume x[0] in [-4,4]; ensure y[0]==0;
          }
        }''')
        refuted = check(candidate, false_description)
        if refuted['status'] != 'REFUTED' or not refuted['results'][0]['witness_replayed']:
            raise AssertionError('false fixture must have an independently replayed witness')
        edit.save(output / f'{name}.plan.json')
        candidate.save(output / f'{name}.candidate.json')
        description.save(output / f'{name}.description.json')
        results[name] = dict(candidate_id=candidate.sha256, expected_formula=formula,
                             claims=checked, false_claim=refuted, diff=delta,
                             probes=probe(candidate, [{'x': [v]} for v in (-4, -1, 0, 1, 4)]))
    if parent.to_dict() != original:
        raise AssertionError('parent changed')
    summary = dict(kind='known-answer engineering fixture', parent_id=parent.sha256,
                   patch_id=patch.sha256, parent_unchanged=True,
                   original_boundary=region(parent, ['left']), edits=results,
                   scientific_runs=0, model_calls=0, training_updates=0,
                   admission='not_performed')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps(dict(parent_unchanged=True, edits={name: dict(
        true_claims=result['claims']['status'], false_claim=result['false_claim']['status'],
        witness_replayed=result['false_claim']['results'][0]['witness_replayed'])
        for name, result in summary['edits'].items()})))


if __name__ == '__main__':
    main()
