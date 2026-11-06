"""Byte-identity oracle for WSP editor changes; no model calls or games.

Replays every archived WSPL 0.2 program (Go, Breakout, Kinetix, diffusion and
the graph-mode repair screen) through plan/apply. Each case records the
candidate identity and a digest of the complete plan body with only
`implementation_id` removed. That field hashes the editor source and must
change with any code edit; everything else (writes, flat-index order,
rounding examples, charged operations, touched indices, diff) must not.

Archived programs whose parents are not in Git run against seeded synthetic
FP32 parents of the original shapes. Breakout (declared zero target) and the
repair screen (supplied parents) also reproduce Kevin's archived candidate IDs.

  python tools/wsp_editor_oracle.py --write tests/fixtures/wsp_editor_oracle_v1.json
  python tools/wsp_editor_oracle.py --check tests/fixtures/wsp_editor_oracle_v1.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np

from skillforge.wsp_artifact import TensorArtifact, canonical
from skillforge.wsp_editor import plan, apply
from skillforge.wsp_values import Limits

ART = ROOT/'artifacts'
PIN = re.compile(r'sha256="(?:[0-9a-f]{64}|\{\{PARENT_SHA256\}\}|\{PARENT_SHA256\})"')


def synthetic(shapes, seed):
    """Deterministic FP32 weights from a 64-bit LCG (no NumPy RNG stream).

    Magnitudes 2**-20..2**1, random signs, plus exact zeros, negative zeros and
    subnormals so rounding/sign paths are exercised.
    """
    out = {}
    state = np.uint64(seed*0x9E3779B97F4A7C15 % 2**64)
    for k, shape in shapes.items():
        n = int(np.prod(shape))
        i = np.arange(1, n+1, dtype=np.uint64)
        with np.errstate(over='ignore'):
            x = (i*np.uint64(6364136223846793005)+state)*np.uint64(1442695040888963407)
            x ^= x >> np.uint64(29)
        x = (x >> np.uint64(16)).astype(np.uint64)
        mant = (x & np.uint64(0x7FFFFF)).astype(np.uint32)
        expo = (((x >> np.uint64(23)) % np.uint64(22)) + np.uint64(107)).astype(np.uint32)
        sign = ((x >> np.uint64(40)) & np.uint64(1)).astype(np.uint32) << np.uint32(31)
        bits = sign | (expo << np.uint32(23)) | mant
        kind = (x >> np.uint64(41)) % np.uint64(97)
        bits[kind == 0] = 0
        bits[kind == 1] = 0x80000000
        bits[kind == 2] = (mant[kind == 2] | sign[kind == 2])  # subnormal
        out[k] = bits.reshape(shape).view(np.float32)
        state = np.uint64((int(state)+n*7919+1) % 2**64)
    return out


def summary(p, candidate):
    body = p.to_dict()
    body.pop('implementation_id')
    return dict(candidate_id=candidate.sha256, plan_body_sha256=hashlib.sha256(canonical(body).encode()).hexdigest(),
                status=body['status'], resources=body['resources'], rounded=body['rounding']['changed'],
                writes=len(body['writes']))


def run(source, parent, limits):
    if parent is not None:
        source = PIN.sub(f'sha256="{parent.sha256}"', source, count=1)
    try:
        p = plan(source, parent=parent, limits=limits)
        if parent is None:  # Declared dense zero target.
            parent = TensorArtifact(p.to_dict()['parent'], max_elements=limits.max_elements)
        c = apply(p, parent=parent, expected_parent_id=parent.sha256)
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError) as e:
        return dict(error=f'{type(e).__name__}: {e}'), None
    return summary(p, c), c


def go_cases():
    from skillforge.go_authoring import SHAPES, GRAPH, zero_arrays
    limits = Limits(max_elements=1480274, max_operations=100_000_000, max_statements=100000, max_loop_iterations=100000)
    synthetic_parent = TensorArtifact.from_arrays(synthetic(SHAPES, 100), graph=GRAPH, max_elements=1480274)
    for session in range(3):
        # Mirror GoAuthoringStudy: zero start, each accepted candidate is the next parent.
        parent = TensorArtifact.from_arrays(zero_arrays(), graph=GRAPH, max_elements=1480274)
        for path in sorted(ART.glob(f'go_authoring_v1/{session}_wspl_turn_*.wspl')):
            source = path.read_text(encoding='utf-8')
            rec, cand = run(source, parent, limits)
            yield f'go/chain/{path.name}', rec
            if cand is not None:
                parent = cand
            # Each turn also runs alone on one nonzero parent, so +=/-=/*= and
            # parent reads meet random values, signed zeros and subnormals.
            yield f'go/synthetic/{path.name}', run(source, synthetic_parent, limits)[0]


def breakout_cases():
    from skillforge.wsp_breakout import LIMITS
    for path in sorted(ART.glob('wsp_breakout_v1/*.source.wspl')):
        yield f'breakout/{path.name}', run(path.read_text(encoding='utf-8'), None, LIMITS)[0]


def kinetix_cases():
    from skillforge.kinetix_replay import SHAPES, imported_artifact
    from skillforge.wsp_kinetix import LIMITS
    parent = imported_artifact(synthetic(SHAPES, 7))
    for path in sorted(ART.glob('wsp_kinetix_v1/*.source.wspl')):
        yield f'kinetix/{path.name}', run(path.read_text(encoding='utf-8'), parent, LIMITS)[0]


def diffusion_cases():
    from skillforge.diffusion_replay import SHAPES
    from skillforge.diffusion_edit import GRAPH
    limits = Limits(max_elements=23824, max_operations=100_000_000, max_statements=4096, max_loop_iterations=4096)
    for seed in (11, 12, 13):
        parent = TensorArtifact.from_arrays(synthetic(SHAPES, seed), graph=GRAPH)
        for path in sorted(ART.glob('diffusion_edit_v1/*_wspl.source.wspl')):
            yield f'diffusion/{seed}/{path.name}', run(path.read_text(encoding='utf-8'), parent, limits)[0]


def repair_cases():
    from skillforge.wsp_repair import RepairStudy
    index = json.loads((ART/'wsp_repair_v1/index.json').read_text(encoding='utf-8'))
    for t in index['trials']:
        try:
            _, report = RepairStudy().construct(t['task'], t['arm'], (ART/'wsp_repair_v1'/t['answer']).read_text(encoding='utf-8'))
        except (ValueError, TypeError, KeyError, IndexError, ArithmeticError) as e:
            yield f'repair/{t["answer"]}', dict(error=f'{type(e).__name__}: {e}'); continue
        rec = dict(candidate_id=report['candidate_id'], archived_candidate_id=t['candidate_id'])
        for key in ('tensor_plan', 'wspl_plan'):
            if key in report:
                body = dict(report[key]); body.pop('implementation_id')
                rec[key+'_body_sha256'] = hashlib.sha256(canonical(body).encode()).hexdigest()
        yield f'repair/{t["answer"]}', rec


def archived_breakout_ids():
    index = json.loads((ART/'wsp_breakout_v1/index.json').read_text(encoding='utf-8'))
    return {f'breakout/draw_{e["draw"]}_{e["arm"]}.source.wspl': e['candidate_id'] for e in index['entries']}


SUITES = dict(go=go_cases, breakout=breakout_cases, kinetix=kinetix_cases, diffusion=diffusion_cases, repair=repair_cases)


def collect(suites=tuple(SUITES)):
    results, seconds = {}, {}
    for name in suites:
        t = time.perf_counter()
        for key, rec in SUITES[name]():
            results[key] = rec
        seconds[name] = round(time.perf_counter()-t, 3)
    return results, seconds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--write', type=Path); g.add_argument('--check', type=Path)
    ap.add_argument('--suite', action='append', choices=list(SUITES))
    a = ap.parse_args()
    suites = tuple(a.suite or SUITES)
    results, seconds = collect(suites)
    print(json.dumps(dict(seconds=seconds, cases=len(results)), indent=1))
    archived = archived_breakout_ids()
    mismatched = [k for k, v in archived.items() if k in results and results[k].get('candidate_id') != v]
    mismatched += [k for k, v in results.items() if 'archived_candidate_id' in v and v['candidate_id'] != v['archived_candidate_id']]
    if mismatched:
        print('archived candidate IDs not reproduced:', mismatched); return 1
    if a.write:
        with open(a.write, 'x', encoding='utf-8', newline='\n') as f:
            json.dump(dict(format='wsp.editor_oracle.v1', cases=results), f, indent=1, sort_keys=True); f.write('\n')
        return 0
    expected = json.loads(a.check.read_text(encoding='utf-8'))['cases']
    bad = [k for k in results if results[k] != expected.get(k)]
    for k in bad:
        print('DIFFERS', k, '\n  expected', expected.get(k), '\n  actual  ', results[k])
    print('OK' if not bad else f'{len(bad)} differing cases')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
