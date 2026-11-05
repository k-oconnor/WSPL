"""Bounded WSPL construction subprocess; no policy outputs or games."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from skillforge.go_authoring import read_arrays,GRAPH
from skillforge.wsp_artifact import TensorArtifact
from skillforge.wsp_editor import plan,apply
from skillforge.wsp_values import Limits


def main():
    p=argparse.ArgumentParser()
    for name in ('source','parent','output','receipt'):p.add_argument('--'+name,required=True)
    a=p.parse_args();started=time.monotonic()
    try:
        arrays=read_arrays(a.parent);parent=TensorArtifact.from_arrays(arrays,graph=GRAPH,max_elements=1480274)
        source=Path(a.source).read_text(encoding='utf-8')
        if '{{PARENT_SHA256}}' not in source:raise ValueError('parent placeholder required')
        source=source.replace('{{PARENT_SHA256}}',parent.sha256)
        staged=plan(source,parent=parent,limits=Limits(max_elements=1480274,max_operations=100_000_000,max_statements=100000,max_loop_iterations=100000))
        candidate=apply(staged,parent=parent,expected_parent_id=parent.sha256)
        if candidate.to_dict()['graph']!=GRAPH:raise ValueError('fixed graph required')
        Path(a.output).mkdir(exist_ok=False);np.savez_compressed(Path(a.output)/'candidate.npz',**candidate.arrays())
        d=staged.to_dict()
        result=dict(exit_code=0,seconds=time.monotonic()-started,stdout='',stderr='',plan_sha256=staged.sha256,
            parent_id=parent.sha256,candidate_id=candidate.sha256,resources=d['resources'],placements=len(d.get('placements',[])),
            numeric_profile=d['numeric_profile'])
    except (ValueError,TypeError,KeyError,IndexError,ArithmeticError,SyntaxError,RecursionError) as e:
        result=dict(exit_code=1,seconds=time.monotonic()-started,stdout='',stderr=f'{type(e).__name__}: {e}')
    Path(a.receipt).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
