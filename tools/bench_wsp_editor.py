"""Time WSP editor plan/apply on archived Go programs; no model calls or games.

Uses the oracle's seeded 1.48M-parameter synthetic parent, so timings are
reproducible on any machine:

  python tools/bench_wsp_editor.py                      # default three programs
  python tools/bench_wsp_editor.py artifacts/go_authoring_v1/2_wspl_turn_00.wspl
  python tools/bench_wsp_editor.py --profile artifacts/go_authoring_v1/2_wspl_turn_00.wspl
"""
import argparse
import cProfile
import json
from pathlib import Path
import platform
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'tools'))
import numpy as np

from skillforge.go_authoring import SHAPES, GRAPH
from skillforge.wsp_artifact import TensorArtifact
from skillforge.wsp_editor import plan, apply
from skillforge.wsp_values import Limits
from wsp_editor_oracle import synthetic, PIN

DEFAULT = ['artifacts/go_authoring_v1/0_wspl_turn_00.wspl', 'artifacts/go_authoring_v1/2_wspl_turn_00.wspl',
           'artifacts/go_authoring_v1/2_wspl_turn_06.wspl']
LIMITS = Limits(max_elements=1480274, max_operations=100_000_000, max_statements=100000, max_loop_iterations=100000)


def time_program(path, parent):
    source = PIN.sub(f'sha256="{parent.sha256}"', (ROOT/path).read_text(encoding='utf-8'), count=1)
    t0 = time.perf_counter(); staged = plan(source, parent=parent, limits=LIMITS)
    t1 = time.perf_counter(); apply(staged, parent=parent, expected_parent_id=parent.sha256)
    t2 = time.perf_counter()
    return dict(program=path, plan_seconds=round(t1-t0, 3), apply_seconds=round(t2-t1, 3),
                writes=len(staged.to_dict()['writes']), plan_megabytes=round(len(staged._text)/1e6, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('programs', nargs='*', default=DEFAULT)
    ap.add_argument('--profile', action='store_true')
    a = ap.parse_args()
    t = time.perf_counter()
    parent = TensorArtifact.from_arrays(synthetic(SHAPES, 100), graph=GRAPH, max_elements=1480274)
    print(json.dumps(dict(python=platform.python_version(), numpy=np.__version__, machine=platform.platform(),
                          parent_seconds=round(time.perf_counter()-t, 3))))
    for path in a.programs:
        if a.profile:
            profiler = cProfile.Profile(); profiler.enable(); result = time_program(path, parent); profiler.disable()
            pstats.Stats(profiler).sort_stats('cumulative').print_stats(25)
        else:
            result = time_program(path, parent)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
