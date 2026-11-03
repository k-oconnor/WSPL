"""Export public requests only; no candidate generation, grading or API calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments')]
from skillforge.authoring_study import AuthoringTrial
from run_authoring_screen import trial_specs


def prepare(study, output):
    if not study.get('model_id'):
        raise ValueError('select the proposer model before exporting scientific requests')
    specs = trial_specs(study)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for trial_id, task, arm in specs:
        trial = AuthoringTrial(task, arm, trial_id=trial_id, model_id=study['model_id'])
        (output/f'{trial_id}.json').write_text(json.dumps(trial.request(),indent=2)+'\n',encoding='utf-8',newline='\n')
    # Same portable operator documentation available to every arm. It contains
    # grammar and inspected examples, not these generated tasks' reference answers.
    for name in ('SCALAR_CORE.md','COMPILER.md'):
        (output/name).write_bytes((ROOT/'docs/wspl'/name).read_bytes())
    inventory = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}
    (output/'inventory.json').write_text(json.dumps(dict(files=inventory,
        note='public requests; task setup is inspected, not a sealed task store'),indent=2)+'\n',encoding='utf-8',newline='\n')
    return inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=prepare(json.loads(args.study.read_text(encoding='utf-8')),args.output)
        print(json.dumps(dict(public_files=len(result),output=str(args.output),api_calls=0)))
    except (OSError,ValueError) as exc:
        print(str(exc),file=sys.stderr)
        raise SystemExit(1)


if __name__=='__main__':main()
