"""Astra demonstrations -> matched student continuations -> frozen deployment."""
import gc
import hashlib
import json
import queue
import subprocess
import threading
import time
from pathlib import Path
import torch
from skillforge.authoring_sft import adapter_state,restore_adapter,frozen_hash,encode
from skillforge.executive_teaching import CONDITIONS,ordinary_action,teacher_prompt,parse_teacher,eligible_rows
from skillforge.student_control import prompt
from skillforge.teaching_training import fit,state_hash
from skillforge.codex_one_shot import collect
from skillforge.reliable_json import save,status


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Simulator:
    def __init__(self,config,out,manifest,identity):
        self.log=(out/'simulator.stderr.log').open('wb');self.events=(out/'simulator_rpc.jsonl').open('x')
        self.process=subprocess.Popen([config['sim_python'],config['sim_worker'],'--manifest',str(manifest),'--sha256',identity],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf8',bufsize=1)
        self.queue=queue.Queue()
        def read():
            try:
                for line in self.process.stdout:self.queue.put(line)
            finally:self.queue.put(None)
        self.reader=threading.Thread(target=read,daemon=True);self.reader.start()
    def call(self,request):
        self.events.write(json.dumps(dict(request=request))+'\n');self.events.flush()
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        try:line=self.queue.get(timeout=60)
        except queue.Empty:raise RuntimeError('Physics worker timeout; no retry')
        if line is None:raise RuntimeError('Physics worker exited')
        reply=json.loads(line)
        self.events.write(json.dumps(dict(reply=reply))+'\n');self.events.flush()
        if not reply.get('ok'):raise RuntimeError(reply.get('error'))
        return reply['result']
    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=10)
        self.log.close();self.events.close()


def summarize(rows):
    summary={};paired={}
    for row in rows:
        key=row['arm']+'/'+row['condition'];v=summary.setdefault(key,dict(n=0,success=0,probes=0,ticks=0))
        v['n']+=1;v['success']+=row['success'];v['probes']+=row['probes'];v['ticks']+=row['physical_ticks']
    for arm in ('unchanged','rehearsal'):
        for condition in CONDITIONS:
            a={r['mission']:r['success'] for r in rows if r['arm']==arm and r['condition']==condition}
            b={r['mission']:r['success'] for r in rows if r['arm']=='taught' and r['condition']==condition}
            paired[arm+'/'+condition]=dict(improvements=sum(b[i] and not a[i] for i in a),regressions=sum(a[i] and not b[i] for i in a))
    # Main contrast: require a useful ordinary-task control before attributing adaptation gains.
    ordinary_ok=summary['rehearsal/ordinary']['success']>=6
    retention=all(paired[a+'/ordinary']['regressions']==0 for a in ('unchanged','rehearsal'))
    gains={a:sum(paired[a+'/'+c]['improvements']-paired[a+'/'+c]['regressions'] for c in CONDITIONS if c!='ordinary') for a in ('unchanged','rehearsal')}
    per_condition=all(paired['rehearsal/'+c]['improvements']>=paired['rehearsal/'+c]['regressions'] for c in CONDITIONS if c!='ordinary')
    return dict(summary=summary,paired=paired,adaptation_net_gains=gains,ordinary_control_passed=ordinary_ok,
        retention_passed=retention,learning_signal_passed=ordinary_ok and retention and per_condition and min(gains.values())>=4)


def run(constructor,output_dir,config):
    out=Path(output_dir);started=time.monotonic();live={};sim=None
    identity=json.loads((out/'source_receipt.json').read_text())['manifest_sha256']
    def report(**kw):
        live.update(kw,elapsed_seconds=time.monotonic()-started);status(out/'status.json',live)
    def stop():
        p=out/'control.json'
        return time.monotonic()-started>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    def guard():
        if stop():raise InterruptedError('Stop or study guard')
    try:
        report(status='verifying')
        for p,h in config['resources'].items():
            guard()
            if sha(p)!=h:raise ValueError('Resource changed '+p)
        study=constructor();model,tokenizer=study.model,study.tokenizer
        parent=torch.load(config['parent'],map_location='cpu',weights_only=False)
        if parent['manifest_sha256']!=config['parent_manifest'] or parent['base_hash']!=frozen_hash(model):raise ValueError('Parent identity')
        restore_adapter(model,parent['adapter']);parent_id=state_hash(parent['adapter'])
        if state_hash(adapter_state(model))!=parent_id:raise ValueError('Restored executive differs from parent')
        save(out/'parent_identity.json',dict(adapter_sha256=parent_id,base_hash=parent['base_hash']))
        sim=Simulator(config,out,out/'approved_manifest.json',identity)
        def start(label,seed,count,condition,budget):
            guard()
            return sim.call(dict(op='start',artifact=config['motor_artifact'],output=str(out/label),seed=seed,count=count,condition=condition,max_decisions=budget))
        def finish(label):
            return sim.call(dict(op='finish',trace_path=str(out/(label+'.trace.json.gz'))))
        def generate_action(public,phase,arm,mission):
            guard();report(status=phase,arm=arm,mission=mission)
            text=prompt(public['observation'],public['documentation'])
            rendered=tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True)
            ids=tokenizer(rendered,add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
            if ids.shape[1]+96>4096:raise ValueError('Evaluation context limit')
            with torch.inference_mode():
                generated=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,use_cache=True,max_new_tokens=96,pad_token_id=tokenizer.eos_token_id)
            tokens=generated[0,ids.shape[1]:].tolist();action=tokenizer.decode(tokens,skip_special_tokens=True)
            with (out/'student_replies.jsonl').open('a') as f:f.write(json.dumps(dict(phase=phase,arm=arm,mission=mission,prompt=text,reply=action,tokens=len(tokens),hit_cap=len(tokens)==96))+'\n')
            return action
        model.eval();model.gradient_checkpointing_disable()
        acquisition=[]
        for index,condition in enumerate(CONDITIONS):
            label=f'acquisition_{index}';public=start(label,34000000+2*index,2,condition,16)
            while not public['finished']:
                action=generate_action(public,'student_acquisition','unchanged',index)
                public=sim.call(dict(op='act',action=action))
            acquisition.append(dict(condition=condition,**finish(label)));save(out/'student_acquisition.json',acquisition)
        if all(r['success'] for r in acquisition if r['condition']!='ordinary'):
            result=dict(status='complete',phase='no_acquisition_adaptation_deficit',student_acquisition=acquisition,
                author_calls=0,student_training_started=False,reserved_evaluation_started=False,elapsed_seconds=time.monotonic()-started)
            save(out/'results.json',result);report(status='complete',results=result);return
        episodes=[]
        for index in range(8):
            condition=CONDITIONS[index//2];label=f'teacher_{index}'
            public=start(label,34000000+index,2,condition,16);records=[]
            for decision in range(16):
                guard();report(status='teacher',mission=index,condition=condition,decision=decision)
                student=prompt(public['observation'],public['documentation'])
                def heartbeat():report(status='teacher',mission=index,condition=condition,decision=decision);return stop()
                text,audit=collect(config['cli'],config['catalog'],config['instructions'],out/f'{label}_turn_{decision}',teacher_prompt(public),1200,heartbeat)
                try:prediction,action=parse_teacher(text)
                except (ValueError,TypeError):prediction=None;action='{}'
                response=sim.call(dict(op='act',action=action))
                record=dict(student_prompt=student,teacher_response=text,prediction=prediction,action=action,
                    accepted=prediction is not None and response['outcome']['status'] not in ('invalid_action','error'),
                    feedback=response['outcome'],usage=audit['usage'])
                records.append(record);save(out/(label+'.partial.json'),records);public=response
                if public['finished']:break
            result=finish(label);episode=dict(mission=label,condition=condition,records=records,result=result)
            episodes.append(episode);save(out/'teacher_episodes.json',episodes)
        try:teacher,coverage=eligible_rows(episodes)
        except ValueError as error:
            result=dict(status='complete',phase='teacher_gate_failed',reason=str(error),
                coverage={c:sum(e['condition']==c and e['result']['success'] for e in episodes) for c in CONDITIONS},
                student_training_started=False,reserved_evaluation_started=False,elapsed_seconds=time.monotonic()-started)
            save(out/'results.json',result);report(status='complete',results=result);return
        save(out/'teacher_rows.json',teacher);save(out/'teacher_coverage.json',coverage)
        ordinary=[];mission=0
        while len(ordinary)<512:
            guard();label=f'ordinary_{mission}';public=start(label,35000000+mission,2+mission%4,'ordinary',32);records=[]
            while not public['finished']:
                guard();student=prompt(public['observation'],public['documentation']);action=json.dumps(ordinary_action(public),separators=(',',':'))
                response=sim.call(dict(op='act',action=action))
                records.append(dict(mission=label,messages=[dict(role='user',content=student),dict(role='assistant',content=action)]));public=response
            result=finish(label);save(out/(label+'.json'),result)
            if not result['success']:raise ValueError('Ordinary rehearsal reference failed; do not silently filter')
            ordinary.extend(records);mission+=1
            report(status='ordinary_collection',missions=mission,rows=len(ordinary))
        save(out/'ordinary_rows.json',ordinary)
        for row in ordinary+teacher:encode(tokenizer,row['messages'],4096)
        save(out/'corpus_identity.json',dict(teacher_sha256=sha(out/'teacher_rows.json'),ordinary_sha256=sha(out/'ordinary_rows.json'),
            teacher_records=len(teacher),ordinary_records=len(ordinary),
            teacher_unique=len({json.dumps(r['messages'],sort_keys=True) for r in teacher}),
            ordinary_unique=len({json.dumps(r['messages'],sort_keys=True) for r in ordinary}),
            teacher_unique_observations=len({r['messages'][0]['content'] for r in teacher}),
            teacher_missions=len(episodes),positive_teacher_missions=sum(coverage.values()),ordinary_missions=mission))
        for arm in ('rehearsal','taught'):
            fit(model,tokenizer,parent,ordinary,teacher,arm,out/arm,identity,guard,report)
        frozen={a:sha(out/a/'adapter_final.pt') for a in ('rehearsal','taught')}
        save(out/'frozen.json',dict(parent_sha256=sha(config['parent']),adapters=frozen,motor_sha256=sha(config['motor_artifact'])))
        model.eval();model.gradient_checkpointing_disable();gc.collect();torch.cuda.empty_cache()
        rows=[];starts={}
        for arm in ('unchanged','rehearsal','taught'):
            state=parent if arm=='unchanged' else torch.load(out/arm/'adapter_final.pt',map_location='cpu',weights_only=False)
            restore_adapter(model,state['adapter'])
            for index in range(32):
                condition=CONDITIONS[index//8];label=f'{arm}_mission_{index}'
                public=start(label,36000000+index,3+index%3,condition,4*(3+index%3)+12)
                start_hash=hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()
                if index in starts and starts[index]!=start_hash:raise ValueError('Paired mission starts differ')
                starts[index]=start_hash;save(out/'paired_starts.json',starts)
                while not public['finished']:
                    action=generate_action(public,'evaluation',arm,index)
                    public=sim.call(dict(op='act',action=action))
                result=finish(label);rows.append(dict(arm=arm,condition=condition,mission=index,**result));save(out/'scores.json',rows)
        assert frozen_hash(model)==parent['base_hash']
        assert sha(config['parent'])==config['resources'][config['parent']]
        assert sha(config['motor_artifact'])==config['resources'][config['motor_artifact']]
        assert all(sha(out/a/'adapter_final.pt')==h for a,h in frozen.items())
        result=dict(status='complete',**summarize(rows),elapsed_seconds=time.monotonic()-started)
        save(out/'results.json',result);report(status='complete',results=result)
    except BaseException as error:
        report(status='paused' if isinstance(error,InterruptedError) else 'failed',error=repr(error));raise
    finally:
        if sim is not None:sim.close()
