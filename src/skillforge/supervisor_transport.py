"""Windows Astra collection / Linux isolated Python tool bridge.

Visible history and work files persist. Each author invocation is a new model
request with the full transcript, not an assertion of hidden-state continuity.
"""
import json
from pathlib import Path
import subprocess
import time
from .supervisor_session import TOOL_INSTRUCTIONS
from .supervisor_workspace import save,inventory
from .codex_one_shot import collect
from .reliable_json import status


def linux_path(path):
    path=Path(path).resolve().as_posix()
    if len(path)<3 or path[1:3]!=':/':raise ValueError('Windows drive path required')
    return '/mnt/'+path[0].lower()+path[2:]


class ToolBridge:
    def __init__(self,config,workspace,ledger,limits):
        self.config=config;self.workspace=Path(workspace);self.ledger=Path(ledger);self.limits=limits
    def collect(self,brief,transcript,folder):
        prompt=brief+'\n'+TOOL_INSTRUCTIONS+'\nComplete visible history:\n'+json.dumps(transcript)
        # Each collect is separately receipted. No attempt to silently reuse a
        # failed server-side turn or switch models when capacity is unavailable.
        frames=[]
        if transcript and transcript[-1]['role']=='tool':
            frames=[Path(p) for p in transcript[-1]['content'].get('archived_images',[])]
        text,audit=collect(self.config['cli'],self.config['catalog'],self.config['instructions'],
            folder/'collection',prompt,1200,self.expired,images=frames,profile='supervisor')
        save(folder/'author_usage.json',audit)
        return text
    def expired(self):
        state=json.loads(self.ledger.read_text())
        status(self.ledger.parent/'status.json',dict(status='astra_call',calls_used=state['calls_used'],
            elapsed_seconds=time.time()-state['started']))
        return time.time()-state['started']>=state['limits']['wall_seconds']
    def execute(self,request,folder):
        save(folder/'workspace_before.json',inventory(self.workspace))
        image_request=self.workspace/'requested_images.json'
        if image_request.exists():image_request.unlink()
        script=self.workspace/'_supervisor_request.py'
        script.write_text(request['code'],encoding='utf8')
        (folder/'executed.py').write_bytes(script.read_bytes())
        readonly={k:linux_path(v) for k,v in self.config['readonly'].items()}
        cfg=dict(ledger=linux_path(self.ledger),limits=self.limits,workspace=linux_path(self.workspace),
            runtime=self.config['runtime'],python_home=self.config['python_home'],readonly=readonly,
            command=['/runtime/bin/python','/work/_supervisor_request.py'],
            gpu=request['gpu'],seconds=request['seconds'],logs=linux_path(folder/'job'),label=folder.name)
        save(folder/'job_config.json',cfg)
        args=['wsl.exe','-d','Ubuntu','--',self.config['runtime']+'/bin/python',
              linux_path(self.config['job_worker']),linux_path(folder/'job_config.json')]
        with (folder/'worker.stdout.log').open('wb') as out,(folder/'worker.stderr.log').open('wb') as err:
            process=subprocess.Popen(args,stdout=out,stderr=err)
            while process.poll() is None:
                state=json.loads(self.ledger.read_text())
                status(self.ledger.parent/'status.json',dict(status='supervisor_job',turn=folder.name,
                    gpu=request['gpu'],elapsed_seconds=time.time()-state['started'],
                    job_logs=str(folder/'job')))
                time.sleep(2)
        if process.returncode:raise RuntimeError('Isolated worker failed; inspect archived stderr')
        response=json.loads((folder/'worker.stdout.log').read_text())
        save(folder/'workspace_after.json',inventory(self.workspace))
        code_archive=folder/'candidate_source';code_archive.mkdir()
        for p in self.workspace.rglob('*.py'):
            destination=code_archive/p.relative_to(self.workspace)
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(p.read_bytes())
        for name in ('stdout','stderr'):
            text=(folder/'job'/f'{name}.log').read_text(encoding='utf8',errors='replace')
            response[name]=text[-12000:];response[name+'_truncated']=len(text)>12000
        if image_request.exists():
            frames=json.loads(image_request.read_text())
            if not isinstance(frames,list) or len(frames)>12:raise ValueError('At most 12 requested frames per response')
            response['archived_images']=[]
            for i,name in enumerate(frames):
                path=(self.workspace/name).resolve()
                if not path.is_relative_to(self.workspace.resolve()):raise ValueError('Frame escapes workspace')
                archived=folder/f'frame_{i:02d}.png';archived.write_bytes(path.read_bytes())
                response['archived_images'].append(str(archived.resolve()))
        return response
