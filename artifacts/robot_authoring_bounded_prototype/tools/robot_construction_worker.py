"""Trusted worker interprets numeric/WSPL data only. Host controls every path."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'src')]
from source_consistency import verify, CheckedImports


def main():
    p = argparse.ArgumentParser()
    for key in ('manifest', 'sha256', 'parent', 'features', 'request', 'output', 'arm'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    manifest = verify(ROOT, Path(a.manifest), a.sha256)
    sys.meta_path.insert(0, CheckedImports(ROOT, manifest))
    import numpy as np
    from skillforge.robot_construction import construct
    from skillforge.reliable_json import save
    out = Path(a.output)
    request = json.loads(Path(a.request).read_text(encoding='utf-8'))
    try:
        with np.load(a.parent, allow_pickle=False) as file:
            parent = {k: file[k] for k in file.files}
        with np.load(a.features, allow_pickle=False) as file:
            features = {k: file[k] for k in file.files}
        candidate, receipt, exported = construct(parent, features, request, a.arm)
        np.savez(out/'candidate.npz', **candidate)
        np.savez(out/'construction_bindings.npz', **exported)
        save(out/'receipt.json', receipt)
        save(out/'result.json', dict(status='accepted', candidate_id=receipt['candidate_id'],
             changed_elements=receipt['changed_elements'],
             effective_unscaled_delta_norm=receipt['effective_unscaled_delta_norm']))
    except (ValueError, TypeError, KeyError, SyntaxError, IndexError, FloatingPointError,
            OverflowError, RecursionError, np.linalg.LinAlgError) as error:
        save(out/'result.json', dict(status='rejected', error=str(error)[:2000]))


if __name__ == '__main__':
    main()
