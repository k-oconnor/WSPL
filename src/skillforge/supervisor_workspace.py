"""Artifact and process boundaries for an engineering supervisor.

This module never makes model calls. Candidate Python is executable code, not
trusted data: run it only through the isolated Linux worker, never on the host.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    from .reliable_json import save as atomic_save
    atomic_save(Path(path),value)


def inventory(root):
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise ValueError('Real directory required')
    result={}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('Symlinks cannot enter a submission')
        if p.is_file():
            if p.stat().st_nlink!=1:raise ValueError('Hard links cannot enter a submission')
            result[p.relative_to(root).as_posix()]=digest(p)
        elif not p.is_dir():raise ValueError('Special file in submission')
    return result


def freeze(source,store,metadata):
    """Host-owned snapshot; caller must first stop all candidate writers."""
    source=Path(source).resolve();store=Path(store).resolve()
    if store==source or store.is_relative_to(source):raise ValueError('Store must be separate')
    json.dumps(metadata,allow_nan=False)
    before=inventory(source)
    if 'bundle_manifest.json' in before:raise ValueError('Reserved manifest name')
    if 'submission.json' not in before:raise ValueError('Missing deployment declaration')
    store.mkdir(parents=True,exist_ok=True)
    temporary=store/('.pending-'+uuid.uuid4().hex)
    try:
        shutil.copytree(source,temporary)
        copied=inventory(temporary)
        if copied!=before or inventory(source)!=before:raise ValueError('Candidate changed during freeze')
        manifest=dict(schema=1,files=copied,metadata=metadata)
        identity=hashlib.sha256(json.dumps(manifest,sort_keys=True,allow_nan=False).encode()).hexdigest()
        target=store/identity
        if target.exists():
            verify_bundle(target);shutil.rmtree(temporary)
        else:
            save(temporary/'bundle_manifest.json',manifest)
            temporary.rename(target)
        return target
    except BaseException:
        if temporary.exists():shutil.rmtree(temporary)
        raise


def verify_bundle(root):
    root=Path(root);manifest=json.loads((root/'bundle_manifest.json').read_text())
    expected=hashlib.sha256(json.dumps(manifest,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if root.name!=expected:raise ValueError('Bundle identity mismatch')
    actual=inventory(root);actual.pop('bundle_manifest.json',None)
    if actual!=manifest['files']:raise ValueError('Bundle contents changed')
    return manifest


class Budget:
    """Single host-owned serialized job ledger. GPU time means allocated job time.

    GPU jobs run exclusively, so all descendants share one metered device lease.
    CPU jobs receive no GPU mount. Wall time includes downtime on resume.
    """
    def __init__(self,path,wall_seconds,gpu_seconds,calls,clock=time.time):
        self.path=Path(path);self.clock=clock
        if any(not math.isfinite(x) or x<=0 for x in (wall_seconds,gpu_seconds,calls)):
            raise ValueError('Positive finite limits required')
        limits=dict(wall_seconds=wall_seconds,gpu_seconds=gpu_seconds,calls=calls)
        if self.path.exists():
            self.state=json.loads(self.path.read_text())
            if self.state['limits']!=limits:raise ValueError('Cannot change limits on resume')
        else:
            self.state=dict(limits=limits,started=clock(),gpu_used=0.,calls_used=0,active=None,jobs=[])
            self._save()
    def _save(self):save(self.path,self.state)
    def remaining(self,gpu=False):
        left=self.state['limits']['wall_seconds']-(self.clock()-self.state['started'])
        if gpu:left=min(left,self.state['limits']['gpu_seconds']-self.state['gpu_used'])
        return max(0.,left)
    def reserve_call(self):
        self.state=json.loads(self.path.read_text())
        if self.remaining()<=0 or self.state['calls_used']>=self.state['limits']['calls']:
            raise RuntimeError('Supervisor budget exhausted')
        self.state['calls_used']+=1;self._save()
    def begin(self,label,gpu,seconds):
        self.state=json.loads(self.path.read_text())
        if self.state['active'] is not None:raise RuntimeError('Active or interrupted job must be resolved first')
        allowed=min(float(seconds),self.remaining(gpu))
        if not math.isfinite(allowed) or allowed<=0:raise RuntimeError('Job budget exhausted')
        self.state['active']=dict(label=label,gpu=bool(gpu),started=self.clock(),limit=allowed)
        self._save();return allowed
    def finish(self,outcome):
        job=self.state['active']
        if job is None:raise RuntimeError('No job')
        elapsed=max(0.,self.clock()-job['started'])
        if job['gpu']:self.state['gpu_used']+=elapsed
        self.state['jobs'].append(dict(job,elapsed=elapsed,outcome=outcome))
        self.state['active']=None;self._save()


def sandbox_command(workspace,runtime,command,readonly=None,gpu=False,python_home=None):
    """Explicit Linux mount allowlist; no host home, /mnt, or network exposure.

    Runtime must be a self-contained Linux virtualenv rooted at /runtime.
    GPU runtime compatibility is a separate preflight, never inferred here.
    """
    if os.name!='posix':raise RuntimeError('Linux worker required')
    readonly=readonly or {}
    if not command or any(not isinstance(x,str) for x in command):raise ValueError('Command argv required')
    args=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv',
          '--ro-bind','/usr','/usr','--symlink','usr/bin','/bin',
          '--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64',
          '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/home',
          '--bind',str(Path(workspace).resolve()),'/work',
          '--ro-bind',str(Path(runtime).resolve()),'/runtime',
          '--setenv','HOME','/tmp','--setenv','PATH','/runtime/bin:/usr/bin',
          '--setenv','PYTHONNOUSERSITE','1','--setenv','PYTHONDONTWRITEBYTECODE','1',
          '--setenv','MUJOCO_GL','osmesa','--setenv','PYOPENGL_PLATFORM','osmesa',
          '--setenv','NUMBA_CACHE_DIR','/tmp/numba','--setenv','OMP_NUM_THREADS','4',
          '--setenv','HF_HUB_OFFLINE','1','--chdir','/work']
    for name,host in readonly.items():
        if not name or not name.replace('_','').isalnum():raise ValueError('Invalid mount name')
        args+=['--ro-bind',str(Path(host).resolve()),'/inputs/'+name]
    if python_home:
        base=str(Path(python_home).resolve())
        args+=['--ro-bind',base,base]
    if gpu:
        if not Path('/dev/dxg').exists():raise RuntimeError('WSL GPU device unavailable')
        args+=['--dev-bind','/dev/dxg','/dev/dxg','--ro-bind','/usr/lib/wsl','/usr/lib/wsl',
               '--setenv','LD_LIBRARY_PATH','/usr/lib/wsl/lib']
    return args+['--']+list(command)


def run_job(budget,label,argv,logs,gpu=False,seconds=1200):
    """Run a sandbox process; namespace lifetime and process group contain children."""
    if os.name!='posix':raise RuntimeError('Linux worker required')
    logs=Path(logs);logs.mkdir(parents=True,exist_ok=False)
    allowed=budget.begin(label,gpu,seconds);proc=None;outcome='launch_failed'
    start=time.monotonic()
    try:
        save(logs/'command.json',dict(argv=argv,gpu=gpu,limit=allowed))
        with (logs/'stdout.log').open('wb') as out,(logs/'stderr.log').open('wb') as err:
            proc=subprocess.Popen(argv,stdout=out,stderr=err,start_new_session=True)
            try:
                code=proc.wait(timeout=allowed);outcome='complete' if code==0 else 'failed'
            except subprocess.TimeoutExpired:
                outcome='timeout';os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=proc.returncode
        result=dict(outcome=outcome,returncode=code,elapsed=time.monotonic()-start)
        save(logs/'result.json',result);return result
    finally:
        if proc is not None:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
        budget.finish(outcome)
