"""Isolated design construction; no model, GPU, test data or network access."""
import argparse,json,os,resource,signal,subprocess,time
from pathlib import Path

WRAPPER=r'''
import sys,json,runpy,resource,numpy as np
sys.path.insert(0,'/library')
from skillforge.capability_bridge import lowrank,validate_plan
from skillforge.wsp_artifact import TensorArtifact
with open('/request.json') as f: request=json.load(f)
store=request['plan'];events=[]
def names():return sorted(store['branches'])
def load(name):
 a=TensorArtifact(store['branches'][name],max_elements=250000)
 return a.to_dict()['graph'],a.arrays()
def save(name,graph,arrays):
 store['branches'][name]=TensorArtifact.from_arrays({k:np.asarray(v,dtype=np.float32).copy() for k,v in arrays.items()},graph=graph,max_elements=250000).to_dict()
 events.append(dict(kind='native_save',branch=name))
def add_lora(name,rank=16,seed=701):
 store['branches'][name]=lowrank(name,rank,seed).to_dict();events.append(dict(kind='add_lora',branch=name,rank=rank))
def remove(name):del store['branches'][name];events.append(dict(kind='remove',branch=name))
def recipe(**kwargs):store['recipe'].update(kwargs)
def wspl(name,source,patches=None,bindings=None):
 if request['interface']!='wspl':raise ValueError('WSPL unavailable in native arm')
 from skillforge.wsp_editor import plan,apply
 from skillforge.wsp_graph import GraphBudget
 from skillforge.wsp_values import Limits
 a=TensorArtifact(store['branches'][name],max_elements=250000)
 # $PARENT is a host convenience shared with the provided example, never an inferred identity.
 text=source.replace('$PARENT',a.sha256)
 artifacts={k:TensorArtifact(v,max_elements=250000) for k,v in (patches or {}).items()}
 p=plan(text,parent=a,bindings=bindings,artifacts=artifacts,limits=Limits(max_elements=250000),budgets={'module':GraphBudget(max_nodes=16,max_depth=8,max_parameters=250000)})
 b=apply(p,parent=a,expected_parent_id=a.sha256);store['branches'][name]=b.to_dict()
 events.append(dict(kind='wspl',branch=name,source=text,plan=p.to_dict()))
resource.setrlimit(resource.RLIMIT_NPROC,(0,0))
runpy.run_path('/proposal.py',run_name='__main__',init_globals=dict(np=np,names=names,load=load,save=save,add_lora=add_lora,remove=remove,recipe=recipe,wspl=wspl,packet=request['packet']))
meta=validate_plan(store)
with open('/output/candidate.json','w') as f:json.dump(dict(plan=store,events=events,metadata=meta),f,allow_nan=False)
'''


def limits():
    for key,value in ((resource.RLIMIT_AS,8<<30),(resource.RLIMIT_FSIZE,256<<20),(resource.RLIMIT_NOFILE,64),(resource.RLIMIT_CPU,240),(resource.RLIMIT_CORE,0)):
        resource.setrlimit(key,(value,value))


def main():
    p=argparse.ArgumentParser()
    for key in ('source','request','library','output','runtime','receipt'):p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(exist_ok=False);candidate=out/'candidate.json';candidate.touch()
    args=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv','--ro-bind','/usr','/usr','--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64','--symlink','usr/bin','/bin','--proc','/proc','--dev','/dev',
        '--size',str(256<<20),'--tmpfs','/tmp','--size',str(256<<20),'--tmpfs','/output','--bind',str(candidate),'/output/candidate.json',
        '--ro-bind',a.runtime,a.runtime,'--ro-bind',str(Path(a.library).resolve()),'/library','--ro-bind',str(Path(a.request).resolve()),'/request.json','--ro-bind',str(Path(a.source).resolve()),'/proposal.py',
        '--chdir','/output','--setenv','HOME','/tmp','--setenv','OPENBLAS_NUM_THREADS','1','--setenv','OMP_NUM_THREADS','1','--setenv','PYTHONDONTWRITEBYTECODE','1',a.runtime+'/bin/python','-I','-c',WRAPPER]
    start=time.monotonic();timed=False
    with (out.parent/'construction.stdout').open('wb') as stdout,(out.parent/'construction.stderr').open('wb') as stderr:
        proc=subprocess.Popen(args,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,preexec_fn=limits,start_new_session=True)
        try:code=proc.wait(timeout=270)
        except subprocess.TimeoutExpired:timed=True;os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
    report=dict(exit_code=code,timed_out=timed,seconds=time.monotonic()-start,stdout=(out.parent/'construction.stdout').read_bytes()[:12000].decode(errors='replace'),stderr=(out.parent/'construction.stderr').read_bytes()[-4000:].decode(errors='replace'))
    Path(a.receipt).write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
