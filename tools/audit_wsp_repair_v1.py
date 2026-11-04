"""Read-only completion replay against the isolated approved repair source."""
import argparse
from collections import Counter
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import re
import struct
import sys


PIN='807c27b5bc0d769969a6bae14dda809fc68b4b185cccfad704b491658e2a28fe'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def independent(artifact,inputs):
    """Recursive scalar rational execution, independent of production traversal."""
    d=artifact.to_dict();g=d['graph'];nodes={n['id']:n for n in g['nodes']}
    arrays={k:[Q(struct.unpack('!f',bytes.fromhex(s))[0]) for s in v['bits']] for k,v in d['tensors'].items()}
    cache={f'input:{k}':list(map(Q,v)) for k,v in inputs.items()}
    def signal(key):
        if key in cache:return cache[key]
        n=nodes[key.split(':',1)[1]];v={p:signal(s) for p,s in n['inputs'].items()};op=n['op']
        if op=='affine':
            name=n['params']['weight'];rows,columns=d['tensors'][name]['shape'];w=arrays[name];b=arrays[n['params']['bias']]
            if n['attrs']['layout']=='in_out':out=[sum((v['x'][i]*w[i*columns+j] for i in range(rows)),b[j]) for j in range(columns)]
            else:out=[sum((x*w[j*columns+i] for i,x in enumerate(v['x'])),b[j]) for j in range(rows)]
        elif op=='relu':out=[max(Q(0),x) for x in v['x']]
        elif op=='identity':out=list(v['x'])
        elif op=='add':out=[a+b for a,b in zip(v['a'],v['b'])]
        elif op=='scale':out=[x*arrays[n['params']['gain']][0] for x in v['x']]
        elif op=='concat':out=v['a']+v['b']
        elif op=='slice':out=v['x'][slice(n['attrs']['start'],n['attrs']['stop'],n['attrs']['step'])]
        else:raise AssertionError('unexpected operator '+op)
        cache[key]=out;return out
    return {k:signal(v) for k,v in g['outputs'].items()}


def target(task,x):
    a,b=x
    if task=='shared_gain':return {'drive':[3*a],'keep':[2*b]}
    if task=='clipped_drive':return {'drive':[min(Q(1,2),max(Q(-1,2),2*a))],'keep':[b]}
    return {'drive':[max(Q(0),a-Q(1,4))-max(Q(0),-a-Q(1,4))],'keep':[max(Q(0),b)]}


def run(root):
    root=Path(root).resolve();sys.path[:0]=[str(root/'src'),str(root/'experiments')]
    from source_consistency import verify
    from skillforge.wsp_artifact import TensorArtifact,digest
    from skillforge.wsp_repair import RepairStudy,TASKS,public_task,prompt
    from skillforge.codex_one_shot import audit_events,audit_preview,command
    from skillforge.breakout_matched import matched_dependency_identity
    from run_wsp_repair import ORDER,comparisons
    out=root/'results/wsp_repair_v1';manifest=root/'experiments/source_pins/wsp_repair_v1.manifest.json'
    m=verify(root,manifest,PIN);c=m['runner_config'];receipt=json.loads((out/'source_receipt.json').read_text(encoding='utf-8'))
    results=json.loads((out/'results.json').read_text(encoding='utf-8'));status=json.loads((out/'status.json').read_text(encoding='utf-8'))
    assert receipt['status']==results['status']==status['status']=='complete'
    assert receipt['constructor_calls']==results['planned_invocations']==results['attempted_invocations']==12
    assert receipt['manifest_sha256']==results['manifest_sha256']==PIN and sha(out/'approved_manifest.json')==PIN
    assert results==status
    for p,h in c['bound_files'].items():assert sha(root/p)==h,p
    assert matched_dependency_identity()['sha256']==c['dependencies_sha256']
    assert {t:digest(public_task(t)) for t in TASKS}==c['task_ids']
    cli=json.loads((root/c['local_paths']).read_text(encoding='utf-8'))['codex_cli'];assert sha(Path(cli))==c['cli_sha256']
    trials=results['trials'];assert [(t['task'],t['draw'],t['arm']) for t in trials]==list(ORDER)
    assert results['comparisons']==comparisons(trials)
    guide=(root/c['guide']).read_text(encoding='utf-8');threads=set();rows=[];raw_hashes={}
    for t in trials:
        folder=out/f'{t["task"]}_{t["draw"]}_{t["arm"]}';answer=(folder/'answer.md').read_text(encoding='utf-8')
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        a=audit_events(events,answer);saved=json.loads((folder/'collection.json').read_text(encoding='utf-8'))
        for k,v in a.items():assert saved[k]==v
        assert t['collection']==saved and a['thread_id'] not in threads;threads.add(a['thread_id'])
        text=(folder/'prompt.txt').read_text(encoding='utf-8');assert text==prompt(t['task'],t['arm'],guide)
        audit_preview(json.loads((folder/'prompt_preview.json').read_text(encoding='utf-8')),text)
        actual_command=json.loads((folder/'command.json').read_text(encoding='utf-8'))
        assert actual_command==command(cli,root/c['catalog'],root/c['instructions'],(folder/'empty').resolve(),(folder/'answer.md').resolve())
        candidate,construction=RepairStudy().construct(t['task'],t['arm'],answer)
        assert candidate.sha256==t['candidate_id']==TensorArtifact.load(folder/'candidate.json').sha256
        assert construction==json.loads((folder/'construction.json').read_text(encoding='utf-8'))
        evaluated=RepairStudy().evaluate(t['task'],candidate);old=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'));assert old==t['evaluation']
        assert evaluated['success']==old['success'] and evaluated['numerical']==old['numerical'] and evaluated['description']==old['description']
        assert [(r['name'],r['status'],r['claim_id']) for r in evaluated['formal']['results']]==[(r['name'],r['status'],r['claim_id']) for r in old['formal']['results']]
        # Fresh exact diagnostic panel; not substituted for the preregistered score.
        values=[Q(-1),Q(-3,7),Q(-1,4),Q(-1,9),Q(0),Q(1,9),Q(1,4),Q(3,7),Q(1)]
        for x in [(a,b) for a in values for b in values]:assert independent(candidate,{'x':x})==target(t['task'],x)
        reply=json.loads(re.findall(r'```json\s*\n(.*?)```',answer,re.S)[0])
        rows.append(dict(task=t['task'],draw=t['draw'],arm=t['arm'],success=old['success'],candidate_id=candidate.sha256,
            parameters=t['resources']['parameters'],nodes=t['resources']['nodes'],reply_fields=sorted(reply),
            structured_operations=t['structured_operations'],reply_bytes=(folder/'answer.md').stat().st_size,
            usage=a['usage'],collection_seconds=saved['active_seconds'],construction_seconds=t['construction_seconds'],evaluation_seconds=t['evaluation_seconds'],
            numeric_max_absolute_error=old['numerical']['max_absolute_error'],numerical_inputs=old['numerical']['completed_samples'],
            thread_id=a['thread_id']))
        for f in folder.rglob('*'):
            if f.is_file() and not {'offline-home','empty'}&set(f.relative_to(folder).parts):raw_hashes[f.relative_to(out).as_posix()]=sha(f)
    summary={}
    for arm in ('direct','wspl'):
        arm_rows=[r for r in rows if r['arm']==arm]
        summary[arm]=dict(successes=sum(r['success'] for r in arm_rows),proposals=len(arm_rows),
            usage={k:sum(r['usage'].get(k,0) for r in arm_rows) for k in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')},
            collection_seconds=sum(r['collection_seconds'] for r in arm_rows),
            construction_seconds=sum(r['construction_seconds'] for r in arm_rows),evaluation_seconds=sum(r['evaluation_seconds'] for r in arm_rows),
            structured_proposals=sum(bool(r['structured_operations']) for r in arm_rows),
            operations=dict(Counter(op for r in arm_rows for op in r['structured_operations'])))
    for f in ('results.json','status.json','source_receipt.json','approved_manifest.json','dependencies.json'):raw_hashes[f]=sha(out/f)
    verify(root,manifest,PIN)
    return dict(status='passed',kind='read-only completion audit; no new authoring or training',manifest_sha256=PIN,
        source_files=len(m['files']),source_unchanged=True,bound_files=len(c['bound_files']),runtime_and_cli_verified=True,
        unique_threads=len(threads),model_tool_calls=0,reply_to_candidate_replays=12,formal_and_numeric_replays=12,
        independent_rational_diagnostic_inputs_per_candidate=81,independent_diagnostic_is_not_primary_score=True,
        started_utc=receipt['started_utc'],finished_utc=receipt['finished_utc'],study_active_seconds=results['active_seconds'],
        decision=results['comparisons']['decision'],arms=summary,trials=rows,raw_sha256=raw_hashes,
        audit_script_sha256=sha(Path(__file__)),new_model_calls=0,training_updates=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.root)
    with a.output.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('trials','raw_sha256')},indent=2))


if __name__=='__main__':main()
