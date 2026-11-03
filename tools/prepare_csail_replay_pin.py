"""Prepare an UNAPPROVED source pin for fixed-artifact replay; never launch it."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.csail_breakout import dependency_identity,file_hash
from source_consistency import inventory,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    args=p.parse_args()
    if args.spec.exists() or args.manifest.exists():raise ValueError('choose new pin paths; never overwrite')
    bundle=Path('data/csail_breakout_reference_v1')
    resource=json.loads((bundle/'bundle.json').read_text())
    weight=next(x for x in resource['files'] if x['path'].endswith('hand.npz'))
    identity=dependency_identity()
    write_json(Path('results/csail_breakout_preparation/dependencies.json'),identity)
    config=dict(bundle=bundle.as_posix(),bundle_sha256=file_hash(bundle/'bundle.json'),
                dependencies_sha256=identity['sha256'],
                episodes_per_arm=1000,max_steps=10000,env_seed_base=310000,
                action_seed_base=410000,wall_guard_seconds=7200,
                auxiliary_files={name:file_hash(name) for name in (
                    'docs/wspl/BREAKOUT_REPLAY_V1.md','requirements/csail-breakout-replay.txt',
                    'tools/prepare_csail_breakout.ps1')})
    spec=dict(schema=1,model_id='csail-breakout-replay-v1',source_roots=['src','experiments','tests','tools'],
              constructor=dict(module='skillforge.csail_breakout',symbol='BreakoutPolicy',source='src/skillforge/csail_breakout.py'),
              runner=dict(module='run_csail_breakout',symbol='run',source='experiments/run_csail_breakout.py'),
              constructor_kwargs=dict(weights_path=(bundle/weight['path']).as_posix(),expected_sha256=weight['sha256']),
              runner_config=config)
    write_json(args.spec,spec)
    write_json(args.manifest,{**spec,'files':inventory(ROOT,spec['source_roots'])})
    print(json.dumps(dict(status='UNAPPROVED',manifest_sha256=file_hash(args.manifest),
                          files=len(inventory(ROOT,spec['source_roots'])),api_calls=0)))


if __name__=='__main__':main()
