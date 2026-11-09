"""Production bank sizes and native sandbox smoke test on synthetic tensors."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import numpy as np
from run_attention_probe import construct
from skillforge.arc_authoring import read_bank
from skillforge.attention_probe import validate


def main():
    rng=np.random.default_rng(42)
    reference={name:rng.normal(size=shape).astype(np.float32) for name,shape in dict(
        Q=(6,9,64),native_K=(6,3,128,64),native_V=(6,3,128,64),attention=(6,9,128),
        O=(6,576,576),answer_embeddings=(4,576),final_norm=(576,)).items()}
    reference.update(K=np.zeros((6,3,64),np.float32),V=np.zeros((6,3,64),np.float32),alpha=np.zeros((6,3),np.float32),
                     K_scale=np.ones((6,3),np.float32),V_scale=np.ones((6,3),np.float32))
    folder=ROOT/'work'/('attention_sandbox_'+str(time.time_ns()));folder.mkdir(parents=True)
    parent=folder/'parent.npz';np.savez_compressed(parent,**reference)
    source="v=load('V'); v[0,0,0]=1; save('V',v)\na=load('alpha'); a[0,0]=0.5; save('alpha',a)\nprint('INSPECT_OK',load('O').shape)"
    config=dict(python_worker=str(ROOT/'tools/arc_python_worker.py'),python_runtime='/home/kevin/venvs/skillforge-author')
    path,feedback=construct(source,parent,folder,config)
    assert feedback['exit_code']==0 and 'INSPECT_OK' in feedback['stdout'],feedback
    bank=read_bank(path,reference);validate(bank,reference,'forced')
    assert bank['V'][0,0,0]==1 and bank['alpha'][0,0]==.5
    result=dict(status='passed',synthetic_only=True,uncompressed_array_bytes=sum(a.nbytes for a in reference.values()),
                archive_bytes=parent.stat().st_size,seconds=feedback['seconds'],output=str(folder))
    (ROOT/'docs/wspl/attention_probe_sandbox_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
