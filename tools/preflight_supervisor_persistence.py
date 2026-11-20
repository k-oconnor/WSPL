"""Scripted two-process export/reload through production worker, no model calls."""
import json
from pathlib import Path
import re
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from skillforge.supervisor_persistence import run_gate
from skillforge.supervisor_transport import ToolBridge
from skillforge.supervisor_workspace import Budget,save


def main():
    parent=ROOT/'data/supervisor_engineering';parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as tmp:
        root=Path(tmp);work=root/'work';work.mkdir()
        limits=dict(wall_seconds=180,gpu_seconds=60,calls=2)
        budget=Budget(root/'budget.json',**limits)
        cfg=dict(runtime='/home/kevin/venvs/wspl-supervisor',
            python_home='/home/kevin/.local/share/uv/python/cpython-3.11.17-linux-x86_64-gnu',
            readonly={},job_worker=str(ROOT/'tools/supervisor_job_worker.py'))
        bridge=ToolBridge(cfg,work,root/'budget.json',limits)
        def scripted(instruction,history,folder):
            nonce=re.search(r'\b[0-9a-f]{32}\b',instruction).group()
            code="import json, numpy as np, hashlib\nfrom pathlib import Path\np=Path('/work/persistence_gate')\n"
            if not history:
                code+="p.mkdir()\nnp.savez(p/'weights.npz',weights=np.array([[1,2],[3,5]],dtype=np.float32))\n"
                code+=f"(p/'receipt.json').write_text(json.dumps(dict(nonce={nonce!r})))\n"
            else:
                code+="with np.load(p/'weights.npz',allow_pickle=False) as a: y=(a['weights']@np.array([2,3],dtype=np.float32)).tolist()\n"
                code+=f"(p/'reload.json').write_text(json.dumps(dict(nonce={nonce!r},output=y,weights_sha256=hashlib.sha256((p/'weights.npz').read_bytes()).hexdigest())))\n"
            return json.dumps(dict(tool='python',code=code,gpu=False,seconds=60))
        bridge.collect=scripted
        run_gate(root/'gate',work,budget,bridge)
        receipt=json.loads((root/'gate/receipt.json').read_text())
        receipt.update(model_calls=0,scripted_requests=2)
        save(parent/'persistence_preflight.json',receipt)
        print(json.dumps(receipt))


if __name__=='__main__':main()
