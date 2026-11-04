"""Read-only released ReLU artifact/WSPL compatibility checks, no sampled tasks."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from skillforge.wsp_artifact import TensorArtifact
from skillforge.wsp_editor import plan,apply
from skillforge.wsp_values import Limits


def main():
    root=ROOT/'data/csail_relu_reference_v1';bundle=json.loads((root/'bundle.json').read_text())
    for f in bundle['files']:
        b=(root/f['path']).read_bytes()
        assert len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256']
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==f['git_blob_sha1']
    records=[]
    for path in sorted((root/'exp2_diffusion/networks').glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        graph=dict(kind='dense',widths=[160,64,64,144],activations=['relu','relu','linear'],layout='in_out')
        assert sum(a.size for a in arrays.values())==23824
        assert all(a.dtype==np.float32 and np.isfinite(a).all() for a in arrays.values())
        artifact=TensorArtifact.from_arrays(arrays,graph=graph)
        records.append(dict(id=path.stem,parameters=23824,dtype='float32',artifact_id=artifact.sha256))
    folder=root/'exp6_go_from_scratch/networks/budget_0'
    arrays={}
    for i in range(1,8):
        name='L'+str(i) if i<7 else 'out'
        for suffix in ('W','b'):arrays[suffix+str(i)]=np.load(folder/(name+'.'+suffix+'.npy'),allow_pickle=False)
    assert sum(a.size for a in arrays.values())==1480274
    assert all(a.dtype==np.float32 and np.isfinite(a).all() for a in arrays.values())
    graph=dict(kind='dense',widths=[243]+[512]*6+[82],activations=['relu']*6+['linear'],layout='in_out')
    started=time.perf_counter()
    artifact=TensorArtifact.from_arrays(arrays,graph=graph,max_elements=1480274)
    limits=Limits(max_elements=1480274,max_operations=100_000_000)
    source='wspl 0.2;target base=artifact("go",sha256="'+artifact.sha256+'");edit base as candidate mode=fixed numeric=exact_to_storage {b7[0]=parent.b7[0];}'
    staged=plan(source,parent=artifact,limits=limits)
    restored=apply(staged,parent=artifact,expected_parent_id=artifact.sha256)
    assert restored.sha256==artifact.sha256
    records.append(dict(id='go_budget_0',parameters=1480274,dtype='float32',artifact_id=artifact.sha256,full_size_noop_plan_apply='bitwise_identity',seconds=time.perf_counter()-started,resources=staged.to_dict()['resources']))
    report=dict(status='passed',reference_commit=bundle['reference_commit'],files=len(bundle['files']),bytes=sum(f['bytes'] for f in bundle['files']),artifacts=records,scope='Hash, shape, finite dtype, WSPL import and full-size no-op edit checks only; no diffusion sampling, Go games or model calls.')
    (ROOT/'docs/wspl/csail_relu_input_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
