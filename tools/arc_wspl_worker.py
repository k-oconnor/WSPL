"""Separate bounded WSPL constructor; no model or benchmark access."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from skillforge.arc_authoring import read_bank
from skillforge.smollm_adapter import initial_bank,edit_wspl


def main():
    p=argparse.ArgumentParser()
    for name in ('source','parent','base','mode','output','receipt'):p.add_argument('--'+name,required=True)
    a=p.parse_args();started=time.monotonic()
    try:
        with np.load(a.base,allow_pickle=False) as z:base={k:z[k] for k in z.files}
        bank=read_bank(a.parent,initial_bank(base,a.mode))
        updated,effective,receipt=edit_wspl(base,bank,a.mode,Path(a.source).read_text(encoding='utf-8'))
        Path(a.output).mkdir(exist_ok=False);np.savez_compressed(Path(a.output)/'candidate.npz',**updated)
        result=dict(exit_code=0,seconds=time.monotonic()-started,stdout='',stderr='',transaction=receipt)
    except (ValueError,TypeError,KeyError,IndexError,ArithmeticError,SyntaxError,RecursionError) as e:
        result=dict(exit_code=1,seconds=time.monotonic()-started,stdout='',stderr=f'{type(e).__name__}: {e}')
    Path(a.receipt).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
