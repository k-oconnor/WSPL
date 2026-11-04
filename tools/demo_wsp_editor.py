"""Known-answer editor engineering fixture; no training, LLM or simulator calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from skillforge.wsp_artifact import TensorArtifact, diff
from skillforge.wsp_editor import plan, apply
from skillforge.wsp_evidence import describe, check, probe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    examples=ROOT/'examples/wsp_editor'
    placed=plan((examples/'deadband.wspl').read_text(encoding='utf-8'))
    raw=plan((examples/'deadband_raw.wspl').read_text(encoding='utf-8'))
    parent=TensorArtifact(placed.to_dict()['parent'])
    candidate=apply(placed,parent=parent,expected_parent_id=parent.sha256)
    direct=apply(raw,parent=parent,expected_parent_id=parent.sha256)
    if candidate.sha256!=direct.sha256:raise AssertionError('placed/raw tensors differ')
    description=describe(candidate,(examples/'response.description.wspl').read_text(encoding='utf-8'))
    true_result=check(candidate,description)
    false_description=describe(candidate,'describe candidate { claim false_claim semantics=real_graph { assume x[0] in [-1,1] and x[1] in [-1,1]; ensure y[0]==0; } }')
    false_result=check(candidate,false_description)
    if true_result['status']!='PROVED' or false_result['status']!='REFUTED':
        raise AssertionError('expected independent proved/refuted engineering fixtures')
    observations=probe(candidate,json.loads((examples/'inputs.json').read_text(encoding='utf-8')))
    parent.save(args.output/'parent.json');candidate.save(args.output/'candidate.json')
    placed.save(args.output/'plan.json');description.save(args.output/'description.json')
    result=dict(kind='known-answer engineering fixture',candidate_id=candidate.sha256,
                raw_and_placed_identical=True,parent_unchanged=True,
                diff=diff(parent,candidate),true_claim=true_result,false_claim=false_result,
                finite_probe=observations,admission='not_performed')
    with (args.output/'summary.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(candidate_id=candidate.sha256,raw_and_placed_identical=True,
                         true_claim=true_result['status'],false_claim=false_result['status'],
                         false_witness_replayed=false_result['results'][0]['witness_replayed'])))


if __name__=='__main__':main()
