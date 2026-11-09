"""Isolated ARC tensor construction. No model, data, GPU or network access."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

WRAPPER=r'''
import resource,runpy,numpy as np
resource.setrlimit(resource.RLIMIT_NPROC,(0,0))
with np.load('/parent.npz',allow_pickle=False) as z: store={k:z[k].copy() for k in z.files}
SHAPES={k:v.shape for k,v in store.items()}
NAMES=list(SHAPES)
def load(name): return store[name].copy()
def save(name,a):
 if name not in store: raise ValueError('unknown tensor')
 a=np.asarray(a,dtype=np.float32)
 if a.shape!=SHAPES[name] or not np.isfinite(a).all(): raise ValueError('shape/finite mismatch')
 store[name]=a.copy()
def saved(): return {k:int(np.count_nonzero(v)) for k,v in store.items()}
runpy.run_path('/proposal.py',run_name='__main__',init_globals=dict(np=np,SHAPES=SHAPES,NAMES=NAMES,load=load,save=save,saved=saved))
np.savez_compressed('/output/candidate.npz',**store)
'''


def limits():
    for name,value in ((resource.RLIMIT_AS,8<<30),(resource.RLIMIT_FSIZE,32<<20),(resource.RLIMIT_NOFILE,64),(resource.RLIMIT_CPU,120),(resource.RLIMIT_CORE,0)):
        resource.setrlimit(name,(value,value))


def main():
    p=argparse.ArgumentParser()
    for name in ('source','parent','output','runtime','receipt'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(exist_ok=False)
    candidate=out/'candidate.npz';candidate.touch()
    source=Path(a.source).resolve();parent=Path(a.parent).resolve();runtime=str(Path(a.runtime).resolve())
    args=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv','--ro-bind','/usr','/usr',
        '--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64','--symlink','usr/bin','/bin','--proc','/proc','--dev','/dev',
        '--size',str(64<<20),'--tmpfs','/tmp','--size',str(32<<20),'--tmpfs','/output','--bind',str(candidate),'/output/candidate.npz',
        '--ro-bind',runtime,runtime,'--ro-bind',str(parent),'/parent.npz','--ro-bind',str(source),'/proposal.py',
        '--chdir','/output','--setenv','HOME','/tmp','--setenv','PATH','/usr/bin','--setenv','OPENBLAS_NUM_THREADS','1',
        '--setenv','OMP_NUM_THREADS','1','--setenv','MKL_NUM_THREADS','1','--setenv','PYTHONDONTWRITEBYTECODE','1',
        runtime+'/bin/python','-I','-c',WRAPPER]
    started=time.monotonic();timed_out=False
    with (out.parent/(out.name+'.stdout')).open('wb') as stdout,(out.parent/(out.name+'.stderr')).open('wb') as stderr:
        proc=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,preexec_fn=limits,start_new_session=True)
        try:code=proc.wait(timeout=150)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
    result=dict(exit_code=code,timed_out=timed_out,seconds=time.monotonic()-started,
        stdout=(out.parent/(out.name+'.stdout')).read_bytes()[:20000].decode(errors='replace'),
        stderr=(out.parent/(out.name+'.stderr')).read_bytes()[-8000:].decode(errors='replace'))
    Path(a.receipt).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
