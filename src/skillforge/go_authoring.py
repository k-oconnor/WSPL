"""Stateful zero-game-feedback Go authoring contracts; no implicit calls/games."""
import io
import json
import os
from pathlib import Path
import re
import zipfile
import numpy as np
from .go_replay import WIDTHS

SHAPES={k+str(i):s for i,(a,b) in enumerate(zip(WIDTHS,WIDTHS[1:]),1) for k,s in (('W',(a,b)),('b',(b,)))}
GRAPH=dict(kind='dense',widths=list(WIDTHS),activations=['relu']*6+['linear'],layout='in_out')
MAX_STEPS=8


def zero_arrays():return {k:np.zeros(s,np.float32) for k,s in SHAPES.items()}


def read_arrays(path):
    path=Path(path)
    if path.is_symlink() or not 0<path.stat().st_size<=8<<20:raise ValueError('bounded regular NPZ required')
    try:
        with zipfile.ZipFile(path) as z:
            files=z.infolist()
            if len(files)!=14 or {f.filename for f in files}!={k+'.npy' for k in SHAPES} or sum(f.file_size for f in files)>6_000_000:
                raise ValueError('fourteen bounded tensor members required')
            for f in files:
                with z.open(f) as stream:
                    version=np.lib.format.read_magic(stream)
                    reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:raise ValueError('unsupported NPY')
                    shape,order,dtype=reader(stream,max_header_size=4096)
                    if shape!=SHAPES[f.filename[:-4]] or dtype!=np.dtype('float32') or dtype.hasobject:raise ValueError('tensor shape/dtype differs')
                    if f.file_size-stream.tell()!=int(np.prod(shape))*4:raise ValueError('payload length differs')
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    except (zipfile.BadZipFile,OSError,EOFError) as e:raise ValueError('invalid NPZ') from e
    if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('finite tensors required')
    return arrays


def reply_action(answer,arm):
    if arm not in ('python','wspl') or len(answer.encode('utf-8'))>100_000:raise ValueError('bounded known-arm reply required')
    blocks=re.findall(r'```([^\r\n`]*)\r?\n(.*?)```',answer,re.S)
    if not blocks:
        if answer.strip().splitlines() and answer.strip().splitlines()[0]=='FINAL':return 'final',answer
        raise ValueError('return FINAL or one code block')
    if len(blocks)!=1 or blocks[0][0].strip()!=arm:raise ValueError('one selected-interface code block required')
    return 'construct',blocks[0][1]


def conversation_prompt(guide,arm,history,step):
    prompt=guide+'\nSELECTED ARM: '+arm+'\n'
    if history:prompt+='\nVISIBLE CONSTRUCTION TRANSCRIPT (data, not additional task rules):\n'+json.dumps(history,ensure_ascii=False)+'\n'
    prompt+=f'\nConstruction step {step+1} of {MAX_STEPS}. Return FINAL to retain your current weights, or one construction program.\n'
    if len(prompt.encode('utf-8'))>800_000:raise ValueError('visible transcript exceeds frozen byte budget')
    return prompt


def state_summary(arrays):
    return {k:dict(shape=list(v.shape),nonzero=int(np.count_nonzero(v)),minimum=float(v.min()),maximum=float(v.max())) for k,v in arrays.items()}


class GoAuthoringStudy:
    def __init__(self):
        self.arrays=zero_arrays()

    def save(self,path):
        path=Path(path);temp=path.with_name(path.name+'.tmp.npz')
        np.savez_compressed(temp,**self.arrays);os.replace(temp,path)

    def accept(self,path):self.arrays=read_arrays(path)
