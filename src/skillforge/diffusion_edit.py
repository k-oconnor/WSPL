"""Coordinate-conjugation editing screen; structural certificates are sufficient only."""
import json
from pathlib import Path
import re
import subprocess
import time
import zipfile
import numpy as np
from .diffusion_replay import Denoiser, PosteriorSampler, MODEL_IDS, SHAPES, sha
from .wsp_artifact import TensorArtifact
from .wsp_editor import plan, apply
from .wsp_values import Limits

GRAPH=dict(kind='dense',widths=[160,64,64,144],activations=['relu','relu','linear'],layout='in_out')
ARMS=('python','wspl')
DEST=np.array([r*12+(c+2)%12 for r in range(12) for c in range(12)])


def shift(x):
    out=np.empty_like(x);out[...,DEST]=x;return out


def reference_edit(parent):
    result={k:v.copy() for k,v in parent.items()}
    result['W1'][DEST,:]=parent['W1'][:144,:]
    result['W3'][:,DEST]=parent['W3']
    result['b3'][DEST]=parent['b3']
    return result


def extract(answer,arm):
    if arm not in ARMS or not isinstance(answer,str) or len(answer.encode())>200_000:
        raise ValueError('known arm and bounded reply required')
    blocks=re.findall(r'```([^\r\n`]*)\r?\n(.*?)```',answer,re.S)
    if len(blocks)!=1 or blocks[0][0].strip()!=arm:raise ValueError('exactly one '+arm+' block required')
    return blocks[0][1]


def linux_path(path):
    p=Path(path).resolve().as_posix()
    if not re.match(r'^[A-Za-z]:/',p):raise ValueError('Windows absolute path required')
    return '/mnt/'+p[0].lower()+p[2:]


def load_candidate(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>2<<20:raise ValueError('bounded regular NPZ required')
    try:
        with zipfile.ZipFile(path) as z:
            infos=z.infolist()
            if len(infos)!=6 or {i.filename for i in infos}!={k+'.npy' for k in SHAPES} or sum(i.file_size for i in infos)>200_000:
                raise ValueError('unexpected or oversized NPZ members')
            for member in infos:
                name=member.filename[:-4]
                with z.open(member) as f:
                    version=np.lib.format.read_magic(f)
                    reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:raise ValueError('unsupported NPY version')
                    shape,fortran,dtype=reader(f,max_header_size=4096)
                    if shape!=SHAPES[name] or dtype!=np.dtype(np.float32) or dtype.hasobject:
                        raise ValueError('invalid NPY shape/dtype before allocation')
                    if member.file_size-f.tell()!=int(np.prod(shape))*4:raise ValueError('NPY payload size differs')
        with np.load(path,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
    except (zipfile.BadZipFile,OSError,EOFError) as exc:
        raise ValueError('invalid NPZ') from exc
    return Denoiser(arrays).weights


class DiffusionEditStudy:
    def __init__(self,reference_root,model_hashes):
        if set(model_hashes)!=set(MODEL_IDS):raise ValueError('all parents required')
        self.parents={};self.paths={}
        for name in MODEL_IDS:
            path=Path(reference_root)/(name+'.npz')
            if sha(path)!=model_hashes[name]:raise ValueError('parent differs')
            self.parents[name]=load_candidate(path);self.paths[name]=path

    def construct(self,arm,answer,folder,worker,runtime):
        folder=Path(folder);source=extract(answer,arm)
        source_path=folder/('proposal.'+('py' if arm=='python' else 'wspl'))
        source_path.write_text(source,encoding='utf-8')
        candidates={};receipts={}
        for name,parent in self.parents.items():
            if arm=='python':
                receipt=folder/(name+'_sandbox.json');output=folder/(name+'_sandbox')
                cmd=['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(worker),
                    '--source',linux_path(source_path),'--parent',linux_path(self.paths[name]),
                    '--output',linux_path(output),'--runtime',runtime,'--receipt',linux_path(receipt)]
                completed=subprocess.run(cmd,capture_output=True,text=True,timeout=35)
                if not receipt.exists():raise RuntimeError('sandbox infrastructure failed: '+completed.stderr[-500:])
                record=json.loads(receipt.read_text())
                if record['status']!='constructed':raise ValueError('Python construction failed: '+name+' '+str(record))
                candidate=load_candidate(output/'candidate.npz');receipts[name]=record
            else:
                artifact=TensorArtifact.from_arrays(parent,graph=GRAPH)
                if '{{PARENT_SHA256}}' not in source:raise ValueError('explicit parent hash placeholder required')
                rendered=source.replace('{{PARENT_SHA256}}',artifact.sha256)
                staged=plan(rendered,parent=artifact,limits=Limits(max_elements=23824,max_operations=100_000_000,max_statements=4096,max_loop_iterations=4096))
                result=apply(staged,parent=artifact,expected_parent_id=artifact.sha256)
                if result.to_dict()['graph']!=artifact.to_dict()['graph']:raise ValueError('fixed architecture required')
                candidate=Denoiser(result.arrays()).weights;receipts[name]=staged.to_dict()
            np.savez_compressed(folder/(name+'_candidate.npz'),**candidate)
            candidates[name]=candidate
        return candidates,receipts


def evaluate_arrays(parent,candidate,*,sample_count=64,seed=9400000):
    """Independent function panel and coupled sampler. No class classifier."""
    f=Denoiser(parent);g=Denoiser(candidate);expected=reference_edit(parent)
    structural=all(candidate[k].tobytes()==expected[k].tobytes() for k in SHAPES)
    rng=np.random.Generator(np.random.PCG64(9300000))
    pixels=np.concatenate((np.zeros((1,144),np.float32),np.ones((1,144),np.float32),
        -np.ones((1,144),np.float32),np.eye(144,dtype=np.float32),
        rng.uniform(-4,4,(128,144)).astype(np.float32)))
    max_error=0.;scale=0.
    for t in (0,1,249,499,749,999):
        reference=shift(f.predict(pixels,t));actual=g.predict(shift(pixels),t)
        max_error=max(max_error,float(np.abs(actual-reference).max()));scale=max(scale,float(np.abs(reference).max()))
    sampler=PosteriorSampler();rng=np.random.Generator(np.random.PCG64(seed))
    x=rng.standard_normal((sample_count,144)).astype(np.float32);y=shift(x)
    for t in range(999,-1,-1):
        noise=rng.standard_normal(x.shape).astype(np.float32) if t else None
        x=sampler.step(x,f.predict(x,t),t,noise)
        y=sampler.step(y,g.predict(y,t),t,shift(noise) if t else None)
    sample_error=float(np.abs(y-shift(x)).max())
    evaluation=dict(structural_certificate='PROVED_REAL_GRAPH_PERMUTATION' if structural else 'NOT_ESTABLISHED',
        certificate_scope='Sufficient stored-tensor permutation witness, not a complete equivalence checker or IEEE proof.',
        probe_max_abs_error=max_error,probe_reference_scale=scale,probe_count=len(pixels)*6,
        sample_max_abs_error=sample_error,samples=sample_count,
        numeric_pass=max_error<=1e-3 and sample_error<=1e-3,
        target='candidate(P*x,t)=P*parent(x,t); P wraps columns right by two')
    return evaluation,dict(parent=x,candidate=y,target=shift(x))
