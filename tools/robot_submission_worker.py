"""Source-checked inert submission validation; never executes author Python."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import verify, CheckedImports


def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','sha256','parent','submission','arm','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    manifest=verify(ROOT,Path(a.manifest),a.sha256)
    sys.meta_path.insert(0,CheckedImports(ROOT,manifest))
    import numpy as np
    from skillforge.robot_submissions import load_arrays,reconstruct
    from skillforge.reliable_json import save
    out=Path(a.output)
    try:
        candidate,receipt=reconstruct(load_arrays(a.parent),a.submission,a.arm)
        np.savez(out/'candidate.npz',**candidate)
        save(out/'construction.json',receipt)
        save(out/'result.json',dict(status='accepted',candidate_id=receipt['candidate_id']))
    except (ValueError,TypeError,KeyError,IndexError,OverflowError,ArithmeticError,EOFError,zipfile.BadZipFile) as e:
        save(out/'result.json',dict(status='rejected',error=str(e)[:2000]))


if __name__=='__main__':main()
