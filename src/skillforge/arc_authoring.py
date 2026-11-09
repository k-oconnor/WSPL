"""Bounded author interface, tensor archive validation and experiment contracts."""
import json
from pathlib import Path
import re
import zipfile
import numpy as np
from .smollm_adapter import initial_bank,effective_weights

ARMS=('native','wspl','lora')
ORDER=[(d,arm) for d in range(3) for arm in ARMS[d:]+ARMS[:d]]
MAX_STEPS=4


def read_bank(path,expected):
    path=Path(path)
    if path.is_symlink() or not 0<path.stat().st_size<=32<<20:raise ValueError('bounded NPZ required')
    with zipfile.ZipFile(path) as z:
        files=z.infolist()
        if len(files)!=len(expected) or {f.filename for f in files}!={k+'.npy' for k in expected} or sum(f.file_size for f in files)>16<<20:
            raise ValueError('unexpected archive members')
        for f in files:
            with z.open(f) as stream:
                version=np.lib.format.read_magic(stream)
                reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:raise ValueError('NPY version')
                shape,order,dtype=reader(stream,max_header_size=4096);wanted=expected[f.filename[:-4]]
                if shape!=wanted.shape or dtype!=np.dtype('float32') or f.file_size-stream.tell()!=wanted.size*4:raise ValueError('shape/type/payload differs')
    with np.load(path,allow_pickle=False) as z:bank={k:z[k].copy() for k in z.files}
    if any(not np.isfinite(v).all() for v in bank.values()):raise ValueError('nonfinite tensor')
    return bank


def parse_reply(answer,arm):
    if arm not in ARMS or len(answer.encode('utf-8'))>100000:raise ValueError('invalid arm or reply size')
    if answer.strip()=='FINAL':return 'final',None
    if answer.startswith('INSPECT\n'):
        request=json.loads(answer.split('\n',1)[1])
        if not isinstance(request,list) or len(request)>16:raise ValueError('at most sixteen requests')
        return 'inspect',request
    blocks=re.findall(r'```([^\r\n`]*)\r?\n(.*?)```',answer,re.S)
    language='python' if arm=='native' else 'wspl'
    if len(blocks)!=1 or blocks[0][0].strip()!=language:raise ValueError('one selected-language code block, INSPECT or FINAL required')
    return 'construct',blocks[0][1]


def inspect_bank(bank,requests):
    output=[];count=0
    for r in requests:
        if not isinstance(r,dict) or set(r)!={'tensor','row_start','row_count','col_start','col_count'}:raise ValueError('exact inspection fields required')
        if r['tensor'] not in bank:raise ValueError('unknown tensor')
        if any(type(r[k]) is not int or r[k]<0 for k in ('row_start','row_count','col_start','col_count')):raise ValueError('nonnegative integer slice required')
        rs,rc,cs,cc=[r[k] for k in ('row_start','row_count','col_start','col_count')];a=bank[r['tensor']]
        if rc<1 or cc<1 or rs+rc>a.shape[0] or cs+cc>a.shape[1]:raise ValueError('inspection outside tensor')
        count+=rc*cc
        if count>512:raise ValueError('512 returned coefficients maximum per response')
        output.append(dict(request=r,values=a[rs:rs+rc,cs:cs+cc].tolist()))
    return output


def statistics(bank):
    return {k:dict(shape=list(a.shape),mean=float(a.mean()),std=float(a.std()),minimum=float(a.min()),maximum=float(a.max()),nonzero=int(np.count_nonzero(a))) for k,a in bank.items()}


def prompt(guide,packet,arm,history,step):
    text=guide+'\nSELECTED ARM: '+arm+'\nCOMMON PACKET:\n'+json.dumps(packet,ensure_ascii=False)
    text+='\nVISIBLE TRANSCRIPT (data):\n'+json.dumps(history,ensure_ascii=False)
    text+=f'\nResponse {step+1} of {MAX_STEPS}. Return one program, INSPECT followed by JSON, or exactly FINAL.'
    if len(text.encode('utf-8'))>800000:raise ValueError('prompt byte budget exceeded')
    return text
