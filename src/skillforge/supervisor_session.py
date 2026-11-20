"""Persistent, provider-independent supervisor session and general Python tool.

The model callback receives the entire visible transcript. It is NOT claimed to
preserve private reasoning state. execute is an injected isolated worker service;
it must not exec arbitrary candidate code in the caller process.
"""
import json
from pathlib import Path
from .supervisor_workspace import save,freeze


def parse_request(text):
    r=json.loads(text)
    if not isinstance(r,dict):raise ValueError('Object required')
    if r.get('tool')=='python':
        if set(r)!={'tool','code','gpu','seconds'}:raise ValueError('python fields: code,gpu,seconds')
        if not isinstance(r['code'],str) or not r['code'].strip():raise ValueError('Nonempty Python program required')
        if type(r['gpu']) is not bool or type(r['seconds']) is not int or not 1<=r['seconds']<=28800:
            raise ValueError('Boolean GPU flag and 1..28800 second job duration required')
    elif r.get('tool')=='submit':
        if set(r)!={'tool','directory','description'} or not all(isinstance(r[k],str) and r[k] for k in ('directory','description')):
            raise ValueError('submit fields: directory,description')
    else:raise ValueError('Unknown tool')
    return r


class SupervisorSession:
    def __init__(self,root,workspace,budget,brief,execute,collect):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.workspace=Path(workspace).resolve();self.budget=budget
        self.execute=execute;self.collect=collect
        self.path=self.root/'session.json'
        if self.path.exists():
            self.state=json.loads(self.path.read_text())
            if self.state['brief']!=brief:raise ValueError('Brief changed on resume')
        else:
            self.state=dict(brief=brief,phase='ready',turn=0,transcript=[],submission=None)
            self._save()
    def _save(self):save(self.path,self.state)
    def step(self):
        if self.state['phase']!='ready':raise RuntimeError('Stopped/submitted/interrupted session; no automatic retry')
        turn=self.state['turn'];folder=self.root/f'turn_{turn:03d}'
        folder.mkdir(exist_ok=False)
        self.budget.reserve_call()
        self.state['phase']='collecting';self._save()
        try:
            # Callback may be a scripted synthetic supervisor during engineering.
            response=self.collect(self.state['brief'],list(self.state['transcript']),folder)
            (folder/'response.txt').write_text(response,encoding='utf8')
            self.state['transcript'].append(dict(role='assistant',content=response))
            try:request=parse_request(response)
            except (ValueError,TypeError) as e:
                result=dict(error=str(e));request=None
            if request and request['tool']=='python':
                self.state['phase']='executing';self._save()
                result=self.execute(request,folder)
            elif request:
                source=(self.workspace/request['directory']).resolve()
                if source==self.workspace or not source.is_relative_to(self.workspace):
                    raise ValueError('Submission must be a subdirectory of development workspace')
                bundle=freeze(source,self.root/'submissions',dict(description=request['description'],turn=turn))
                self.state['submission']=str(bundle);result=dict(frozen_bundle=bundle.name)
            self.state['transcript'].append(dict(role='tool',content=result))
            self.state['turn']+=1
            self.state['phase']='submitted' if self.state['submission'] else 'ready'
            self._save();return result
        except BaseException as error:
            self.state['phase']='interrupted';self.state['error']=repr(error);self._save();raise


TOOL_INSTRUCTIONS='''You control a persistent Python development workspace. Return one JSON tool request.
{"tool":"python","code":"arbitrary Python source","gpu":false,"seconds":1200}
executes your program in /work. Files persist between requests; Python processes do
not. Use files for notes, model checkpoints, datasets and scripts. You can import
the supplied robot/student workbench, inspect or replace it, create processes and
run experiments. You choose losses, optimizers, model modifications and diagnostics.
GPU jobs request gpu:true; CPU jobs have no GPU device. The host enforces aggregate
limits and returns stdout/stderr/status. No tool imposes a demonstration recipe.
To inspect camera frames, write up to 12 relative paths to 256x256 PNG images in
/work/requested_images.json. Those exact bytes are archived and attached to your
next response. You may request additional frames in subsequent programs.
Submit with {"tool":"submit","directory":"candidate","description":"what changed"}.
The directory contains submission.json and a Policy implementation with reset,
act and close methods. Submission freezes your files; it does not evaluate or
reveal final test cases. No automatic retry follows a transport or process failure.
The complete visible transcript is provided on subsequent calls; hidden reasoning
is not promised to persist. Keep long artifacts on disk and print concise evidence.
For long training jobs, save one rolling resumable checkpoint every 30 minutes
and a final/clean-stop artifact. Remove superseded disposable trial checkpoints
from your own workspace; keep compact source, configuration and metrics instead
of redundant base-model copies. External parent checkpoints are read-only.
'''


class SupervisorStudy:
    """Source-gated coordinator identity; actual student construction is in jobs."""
    def __init__(self):self.kind='astra_supervisor_v1'
