"""Local, read-only experiment dashboard. Does not import the experiment runtime."""
import argparse
import ctypes
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import time

ORDER = ('native0','wspl0','wspl1','native1','native2','wspl2')
FINALS = ('parent','gradient',*ORDER)
CONDITIONS = ('ordinary','changed','switch','return')


def process_alive(pid):
    if not pid:return None
    if os.name != 'nt':
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:return False
    try:
        code=ctypes.c_uint32()
        return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
    finally:kernel.CloseHandle(handle)


class Reader:
    def __init__(self):self.cache={};self.warnings=[]
    def read(self,path,default=None):
        path=Path(path)
        try:
            stat=path.stat();stamp=(stat.st_mtime_ns,stat.st_size)
            prior=self.cache.get(path)
            if prior and prior[0]==stamp:return prior[1]
            data=json.loads(path.read_text(encoding='utf-8-sig'))
            self.cache[path]=(stamp,data);return data
        except FileNotFoundError:return default
        except (OSError,ValueError):
            self.warnings.append(path.name+' is updating; showing last readable value')
            return self.cache.get(path,(None,default))[1]


def snapshot(root,reader=None):
    root=Path(root);reader=reader or Reader();reader.warnings=[];read=reader.read
    now=time.time();status=read(root/'status.json',{})
    launch=read(root.with_suffix('.launch.json'),{})
    exit_record=read(root.with_suffix('.exit.json'))
    receipt=read(root/'source_receipt.json',{})
    result=read(root/'results.json')
    alive=process_alive(launch.get('pid'))
    if result and result.get('status')=='complete' and receipt.get('status')=='complete' and exit_record and exit_record.get('returncode')==0:
        state='Complete'
    elif status.get('status')=='failed' or (exit_record and exit_record.get('returncode')!=0):state='Failed'
    elif result and result.get('status')=='complete':state='Finishing'
    elif alive is False:state='Stopped'
    else:state='Running' if launch else 'Starting'
    elapsed=max(0,(exit_record.get('finished_unix',now) if exit_record else now)-launch.get('started_unix',now))
    authors=[];activity=[];tokens={k:0 for k in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')}
    for name in ORDER:
        folder=root/name;frozen=read(folder/'frozen.json');local=read(folder/'status.json',{})
        turns=sorted(folder.glob('turn_*')) if folder.exists() else []
        completed=0;queries=[];evaluations=0;last_dev=None;last_event=None;python_errors=0
        for turn in turns:
            usage=read(turn/'usage.json')
            if usage:
                completed+=1
                for key in tokens:tokens[key]+=usage.get('usage',{}).get(key,0)
            diag=read(turn/'inspection/diagnostics.json')
            if diag:
                queries.append(dict(projections=list(diag.get('projections',{}).values()),examples=diag.get('examples'),
                    correct=diag.get('action_correct'),nll=diag.get('mean_target_nll'),seconds=diag.get('seconds'),
                    tokens=diag.get('tokens'),weights=diag.get('weights')))
                activity.append(dict(author=name,turn=turn.name,kind='Acquisition inspection',
                    details=', '.join(queries[-1]['projections']) or 'Full-model scoring',
                    time=(turn/'inspection/diagnostics.json').stat().st_mtime))
            submission=read(turn/'submission/result.json')
            if submission:
                evaluations+=1
                activity.append(dict(author=name,turn=turn.name,kind='Candidate '+submission.get('status','checked'),
                    details=submission.get('error',''),time=(turn/'submission/result.json').stat().st_mtime))
            episodes=read(turn/'development/episodes.json')
            if episodes is not None:
                last_dev=dict(done=len(episodes),successes=sum(bool(x.get('success')) for x in episodes),
                    counts={c:sum(bool(x.get('success')) for x in episodes if x.get('condition')==c) for c in CONDITIONS})
        for p in folder.glob('sandbox/receipts/*.json'):
            py=read(p,{})
            if py.get('exit_code',0)!=0 or py.get('timed_out'):python_errors+=1
        if turns:
            events=turns[-1]/'collection/events.jsonl'
            if events.exists():
                # Bound tail reads: author message streams can be large.
                try:
                    with events.open('rb') as f:
                        f.seek(max(0,events.stat().st_size-4096));lines=f.read().decode('utf8',errors='replace').splitlines()
                    for line in reversed(lines):
                        try:
                            event=json.loads(line);last_event=event.get('type');break
                        except ValueError:continue
                except OSError:pass
        author_state=(frozen or {}).get('status') or ('active' if folder.exists() else 'queued')
        authors.append(dict(name=name,state=author_state,completed=completed,started=len(turns),
            inspection_attempts=local.get('inspection_attempts',len(queries)),inspections=len(queries),
            evaluations=local.get('proposal_attempts',evaluations),latest_inspection=queries[-1] if queries else None,
            sites=sorted({p for q in queries for p in q['projections']}),development=last_dev,
            python_errors=python_errors,last_event=last_event))
    finals=[]
    ledger=read(root/'all_frozen.json',{})
    for name in FINALS:
        episodes=read(root/('final_'+name)/'episodes.json',[])
        skipped=ledger.get(name,{}).get('status')=='authoring_failed'
        finals.append(dict(name=name,done=len(episodes),expected=0 if skipped else 64,skipped=skipped,
            successes=sum(bool(x.get('success')) for x in episodes),
            counts={c:sum(bool(x.get('success')) for x in episodes if x.get('condition')==c) for c in CONDITIONS}))
    current=next((x['name'] for x in authors if x['state']=='active'),None)
    phase=status.get('status','starting')
    if phase=='final_evaluation':current=status.get('final_arm')
    if state in ('Complete','Failed','Stopped'):current=None
    if phase=='author_call' and current:
        author=next(a for a in authors if a['name']==current)
        phase='Executing author request' if author['completed']>=author['started'] else 'Waiting for Astra response'
    elif phase=='final_evaluation':phase='Final mission evaluation'
    error=status.get('error') or receipt.get('error')
    stamp=(root/'status.json').stat().st_mtime if (root/'status.json').exists() else now
    return dict(state=state,phase=phase,current=current,pid=launch.get('pid'),process_alive=alive,
        elapsed_seconds=elapsed,compute_seconds=status.get('development_compute_seconds',0),
        observed_utc=datetime.now(timezone.utc).isoformat(),status_age_seconds=max(0,now-stamp),
        manifest=receipt.get('manifest_sha256',launch.get('manifest_sha256')),
        authors=authors,finished_authors=sum(a['state'] in ('frozen','authoring_failed') for a in authors),
        responses=sum(a['completed'] for a in authors),response_limit=144,
        inspections=sum(a['inspections'] for a in authors),inspection_limit=48,
        finals=finals,final_done=sum(f['done'] for f in finals),final_expected=sum(f['expected'] for f in finals),
        tokens=tokens,error=error,warnings=reader.warnings,activity=sorted(activity,key=lambda x:x['time'],reverse=True)[:10],
        gate_passes=(result or {}).get('failure_adjusted_gate_passes'),run_path=str(root))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--port',type=int,default=0)
    p.add_argument('--record',type=Path,required=True);args=p.parse_args()
    root=args.run.resolve(strict=True);reader=Reader()
    html=Path(__file__).with_name('robot_status_viewer.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get('Host','').split(':')[0] not in ('127.0.0.1','localhost'):
                self.send_error(403);return
            route=self.path.split('?',1)[0]
            if route=='/':body=html;content='text/html; charset=utf-8'
            elif route=='/api/status':
                try:body=json.dumps(snapshot(root,reader)).encode();content='application/json'
                except Exception:
                    self.send_error(503,'Snapshot temporarily unavailable');return
            elif route=='/favicon.ico':self.send_response(204);self.end_headers();return
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',content)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    record=dict(pid=os.getpid(),url=f'http://127.0.0.1:{server.server_port}',run=str(root),read_only=True)
    args.record.parent.mkdir(parents=True,exist_ok=True);args.record.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
