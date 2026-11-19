"""Source-gated active visual/telemetry controller authoring, then frozen deployment."""
import gzip
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from skillforge.factory_observation import PandaBackend,ObservationSession,CAMERAS
from skillforge.factory_pilot import packet,request,rollout
from skillforge.codex_one_shot import collect
from skillforge.reliable_json import save,status


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(constructor,output_dir,config):
    out=Path(output_dir);started=time.monotonic();backends=[];live={}
    def report(**kw):
        live.update(kw,elapsed_seconds=time.monotonic()-started);status(out/'status.json',live)
    def stop():
        control=out/'control.json'
        return time.monotonic()-started>config['wall_guard_seconds'] or (control.exists() and json.loads(control.read_text()).get('stop'))
    def guard():
        if stop():raise InterruptedError('Study stop or guard')
    def cameras(value):
        if not isinstance(value,list) or len(set(value))!=len(value) or any(c not in CAMERAS for c in value):raise ValueError('Invalid camera list')
        return value
    def archive(label,row):
        log=row.pop('log');p=out/(label+'.trace.json.gz')
        p.write_bytes(gzip.compress(json.dumps(log).encode(),mtime=0));row['trace_path']=p.name
        return row
    try:
        report(status='verifying')
        for p,h in config['resources'].items():
            guard()
            if sha(p)!=h:raise ValueError('Resource changed: '+p)
        study=constructor();original=study.original.sha256
        identity=np.eye(3).tolist();mapping=config['changed_mapping']
        benches={}
        for name,mat in [('A',identity),('B',mapping)]:
            backend=PandaBackend(331,mat);backends.append(backend)
            benches[name]=ObservationSession(backend,out/('bench_'+name),441 if name=='A' else 442,max_steps=160,max_frames=48)
        history=[dict(initial={name:s.observe() for name,s in benches.items()})]
        pending_images=[];evaluations=0
        for turn in range(1,13):
            guard();report(status='astra',turn=turn)
            remaining=dict(responses=13-turn,evaluations=2-evaluations,
                benches={name:dict(steps=s.max_steps-s.steps,frames=s.max_frames-len(s.frames)) for name,s in benches.items()})
            def heartbeat():report(status='astra',turn=turn);return stop()
            text,audit=collect(config['cli'],config['catalog'],config['instructions'],out/f'turn_{turn}',
                packet(history,remaining),1200,heartbeat,images=pending_images)
            pending_images=[];entry=dict(assistant=text,usage=audit['usage']);finish=False
            try:
                a=request(text);tool=a['tool']
                if 'cameras' in a:cameras(a['cameras'])
                if tool=='observe':
                    s=benches[a['bench']]
                    if len(s.frames)+len(a['cameras'])>s.max_frames:raise ValueError('Frame budget')
                    frames=[s.capture(c) for c in a['cameras']]
                    pending_images=[s.output/f['path'] for f in frames]
                    feedback=dict(telemetry=s.observe(),frames=frames)
                elif tool=='act':
                    s=benches[a['bench']]
                    feedback=s.act(a['command'],a['ticks'],a['cameras'],a['events'],a['clip_stride'])
                    pending_images=[s.output/f['path'] for f in feedback['frames']]
                elif tool=='install':
                    study.install(a);feedback=dict(installed=True,artifact=study.artifact())
                elif tool=='evaluate':
                    if evaluations>=2:raise ValueError('Evaluation budget')
                    evaluations+=1;feedback=[]
                    for name,mat in [('A',identity),('B',mapping)]:
                        guard();label=f'calibration_{evaluations}_{name}';folder=out/label
                        row=archive(label,rollout(study,folder,551,mat,[.035,-.025,.02],'selected',a['cameras']))
                        save(out/(label+'.json'),row)
                        # Author sees only robot-readable error, not oracle physical labels.
                        feedback.append(dict(bench=name,goal_m=row['goal_m'],steps=row['steps'],
                            measured_error_m=row['final_measured_error_m'],telemetry=row['final_telemetry'],frames=row['frames']))
                        pending_images.extend(folder/f['path'] for f in row['frames'])
                else:feedback=dict(frozen=True);finish=True
                entry['feedback']=dict(accepted=True,result=feedback)
            except (ValueError,KeyError,TypeError) as err:
                entry['feedback']=dict(accepted=False,error=str(err))
            # Explicit order maps attachments to feedback records, including repeated local frame IDs.
            entry['attachments_next_response']=[dict(index=i+1,path=str(p.relative_to(out)),sha256=sha(p)) for i,p in enumerate(pending_images)]
            history.append(entry);save(out/'history.json',history);save(out/'artifact_last.json',study.artifact())
            for name,s in benches.items():save(out/('bench_'+name+'_events.json'),s.log)
            if finish:break
        for b in backends:b.close()
        backends=[]
        save(out/'artifact_final.json',study.artifact());frozen=sha(out/'artifact_final.json')
        save(out/'frozen.json',dict(artifact_sha256=frozen,original_sha256=original))
        rows=[]
        for phase,mat in [('original_before',identity),('changed',mapping),('original_return',identity)]:
            for arm in ('baseline','matched_original','selected','reference'):
                for i,offset in enumerate(config['offsets']):
                    guard();label=f'{phase}_{arm}_{i}';report(status='evaluation',phase=phase,arm=arm,case=i,total=len(config['offsets']))
                    row=archive(label,rollout(study,out/label,771+i,mat,offset,arm))
                    rows.append(dict(phase=phase,arm=arm,index=i,**row));save(out/'scores.json',rows)
        assert sha(out/'artifact_final.json')==frozen and study.original.sha256==original
        summary={};paired={}
        for row in rows:
            v=summary.setdefault(row['phase']+'/'+row['arm'],dict(n=0,success=0,steps=0))
            v['n']+=1;v['success']+=row['success'];v['steps']+=row['steps']
        for phase in ('original_before','changed','original_return'):
            a={r['index']:r['success'] for r in rows if r['phase']==phase and r['arm']=='baseline'}
            b={r['index']:r['success'] for r in rows if r['phase']==phase and r['arm']=='selected'}
            paired[phase]=dict(improvements=sum(b[i] and not a[i] for i in a),regressions=sum(a[i] and not b[i] for i in a))
        retention=all(paired[p]['regressions']==0 for p in ('original_before','original_return'))
        result=dict(status='complete',summary=summary,paired=paired,retention_passed=retention,
            improvement_passed=paired['changed']['improvements']-paired['changed']['regressions']>=2,
            reference_passed=all(r['success'] for r in rows if r['arm']=='reference'),
            baseline_old_feasible=summary['original_before/baseline']['success']>=6,
            author_responses=len(history)-1,elapsed_seconds=time.monotonic()-started)
        result['feasibility_passed']=all(result[k] for k in ('retention_passed','improvement_passed','reference_passed','baseline_old_feasible'))
        save(out/'results.json',result);report(status='complete',results=result)
    except BaseException as err:
        report(status='paused' if isinstance(err,InterruptedError) else 'failed',error=repr(err));raise
    finally:
        for b in backends:b.close()
