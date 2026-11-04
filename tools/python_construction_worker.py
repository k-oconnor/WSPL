"""Linux bubblewrap worker for a single untrusted NumPy construction program.

Only the input NPZ and output directory enter the sandbox; no repository, home,
network or devices from the host. No agent credentials or evaluator results.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 << 20, 2 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CPU, (15, 16))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run(source, parent, output, runtime):
    source=Path(source).resolve(); parent=Path(parent).resolve(); output=Path(output).resolve()
    if len(source.read_bytes())>200_000: raise ValueError('source exceeds 200KB')
    output.mkdir(exist_ok=False)
    candidate=output/'candidate.npz'
    candidate.touch(exist_ok=False)
    runtime=str(Path(runtime).resolve())
    args=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv',
          '--ro-bind','/usr','/usr','--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64',
          '--symlink','usr/bin','/bin','--proc','/proc','--dev','/dev','--size','2097152','--tmpfs','/tmp',
          '--ro-bind',runtime,runtime,'--ro-bind',str(parent),'/parent.npz',
          '--ro-bind',str(source),'/proposal.py','--size','2097152','--tmpfs','/output',
          '--bind',str(candidate),'/output/candidate.npz',
          '--chdir','/output','--setenv','HOME','/tmp','--setenv','PATH','/usr/bin',
          '--setenv','OPENBLAS_NUM_THREADS','1','--setenv','OMP_NUM_THREADS','1',
          '--setenv','MKL_NUM_THREADS','1','--setenv','PYTHONDONTWRITEBYTECODE','1',
          runtime+'/bin/python','-I','-c',
          "import resource,runpy; resource.setrlimit(resource.RLIMIT_NPROC,(0,0)); runpy.run_path('/proposal.py',run_name='__main__')"]
    started=time.monotonic()
    # /dev/null prevents unbounded stdout/stderr from filling RAM or disk.
    error_path=output.with_name(output.name+'.stderr')
    error_file=error_path.open('wb')
    process=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
        stderr=error_file,preexec_fn=limits,start_new_session=True)
    timed_out=False
    try: code=process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(process.pid,signal.SIGKILL);code=process.wait()
    error_file.close()
    files=list(output.iterdir())
    valid=(not timed_out and code==0 and len(files)==1 and files[0].name=='candidate.npz'
           and files[0].is_file() and not files[0].is_symlink() and files[0].stat().st_size<2<<20)
    return dict(status='constructed' if valid else 'invalid',exit_code=code,timed_out=timed_out,
                seconds=time.monotonic()-started,output_names=[p.name for p in files],stderr_tail=error_path.read_bytes()[-4000:].decode(errors='replace'),
                isolation='bubblewrap user/pid/net/ipc/uts namespaces; explicit mounts; rlimits')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--parent',required=True)
    p.add_argument('--output',required=True);p.add_argument('--runtime',required=True);p.add_argument('--receipt',required=True)
    a=p.parse_args(); result=run(a.source,a.parent,a.output,a.runtime)
    Path(a.receipt).write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='constructed': raise SystemExit(2)


if __name__=='__main__': main()
