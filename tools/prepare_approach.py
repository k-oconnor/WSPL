"""Prepare an UNAPPROVED source/environment identity. Never trains or launches."""
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments')]
from run_approach import external_inventory


def main():
    checkout = Path(json.loads((ROOT/'artifacts/local_paths.json').read_text())['libero_checkout'])
    clean = not subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain'], text=True).strip()
    if not clean: raise RuntimeError('audited LIBERO checkout must be clean')
    versions = {d.metadata['Name']: d.version for d in metadata.distributions()}
    environment = dict(versions=versions, python=sys.version, platform=sys.platform,
                       files=external_inventory(checkout),
                       libero_commit=subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip())
    config = dict(source_seeds=[11, 12, 13, 14], development_seeds=[101, 102, 103, 104],
                  evaluation_seeds=list(range(1001, 1009)), max_steps=120, settling_steps=20,
                  standoff_m=.12, lr=.003, updates=2000, wall_guard_seconds=1200,
                  environment=environment)
    spec = dict(model_id='approach-v1', source_roots=['src', 'experiments', 'tests', 'tools'],
                constructor=dict(module='skillforge.approach', symbol='ApproachStudy', source='src/skillforge/approach.py'),
                constructor_kwargs=dict(seed=7, gain=20.),
                runner=dict(module='run_approach', symbol='run', source='experiments/run_approach.py'), runner_config=config)
    path = ROOT/'experiments/source_pins/approach_v1.spec.json'
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(spec, f, indent=2); f.write('\n')
    print('UNAPPROVED spec:', path)
    print('External files:', sum(len(files) for files in environment['files'].values()))


if __name__ == '__main__': main()
