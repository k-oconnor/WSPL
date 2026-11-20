"""Host-only deployment evaluation. No scorer/simulator is mounted into policy.

Each policy process receives only reset/observation JSON, via an isolated pipe.
This module is an evaluator component, not a supervisor development tool.
"""
import gzip
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from .supervisor_workspace import sandbox_command,verify_bundle,save
from .reliable_json import status


class PolicyProcess:
    def __init__(self,argv,folder,timeout=120):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.err=(self.folder/'stderr.log').open('wb');self.trace=(self.folder/'rpc.jsonl').open('w')
        self.process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,
            text=True,encoding='utf8',start_new_session=True,bufsize=1)
        self.q=queue.Queue();self.timeout=timeout;self.audit={}
        def read():
            try:
                for line in self.process.stdout:self.q.put(line)
            finally:self.q.put(None)
        threading.Thread(target=read,daemon=True).start()
    def call(self,request):
        self.trace.write(json.dumps(dict(request=request))+'\n');self.trace.flush()
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        try:line=self.q.get(timeout=self.timeout)
        except queue.Empty:raise RuntimeError('Policy response timeout')
        if line is None:raise RuntimeError('Policy process exited')
        reply=json.loads(line)
        self.trace.write(json.dumps(dict(reply=reply))+'\n');self.trace.flush()
        if reply.get('ok') is not True:raise RuntimeError('Policy response failed')
        self.audit=reply.get('audit',{})
        return reply['result']
    def close(self):
        try:os.killpg(self.process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        self.process.wait(timeout=10);self.err.close();self.trace.close()


def evaluate_bundle(bundle,missions,config,output):
    from .executive_teaching import TeachingMission
    bundle=Path(bundle);verify_bundle(bundle)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    work=output/'private_policy_scratch';work.mkdir()
    inputs=dict(config['deployment_inputs'],bundle=str(bundle),worker=config['deployment_worker'])
    argv=sandbox_command(work,config['runtime'],['/runtime/bin/python','/inputs/worker','/inputs/bundle'],
                         readonly=inputs,gpu=True,python_home=config['python_home'])
    policy=PolicyProcess(argv,output/'policy');rows=[];started=time.monotonic();policy_failed=None
    try:
        for index,item in enumerate(missions):
            status(output.parent/'progress.json',dict(arm=output.name,mission=index,total=len(missions)))
            if time.monotonic()-started>config['evaluation_seconds']:raise TimeoutError('Evaluation wall guard')
            m=TeachingMission(config['motor_artifact'],output/f'scene_{index}',item['seed'],item['count'],item['condition'])
            try:
                initial=m.public();start_hash=__import__('hashlib').sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()
                prior_calls=policy.audit.get('qwen_forward_calls',0)
                # Regime, seed and private scorer are not in this reset message.
                if policy_failed is None:
                    try:
                        policy.call(dict(op='reset',public=initial))
                        while not m.episode.finished:
                            if time.monotonic()-started>config['evaluation_seconds']:raise TimeoutError('Evaluation wall guard')
                            public=m.public()
                            action=policy.call(dict(op='act',observation=public))
                            m.execute(action if isinstance(action,str) else json.dumps(action))
                    except (RuntimeError,ValueError,BrokenPipeError) as error:
                        policy_failed=repr(error)
                        policy.close()
                result=m.result()
                result['observed_qwen_forward_calls']=policy.audit.get('qwen_forward_calls',0)-prior_calls
                if policy_failed:result.update(success=False,deployment_error=policy_failed)
                private={k:result.pop(k) for k in ('trace','decision_trace','sensor_trace')}
                (output/f'mission_{index}.json.gz').write_bytes(gzip.compress(json.dumps(private).encode(),mtime=0))
                rows.append(dict(mission=index,condition=item['condition'],start_hash=start_hash,**result))
                save(output/'scores.json',rows)
            finally:m.close()
        verify_bundle(bundle);return rows
    finally:policy.close()
