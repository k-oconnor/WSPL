"""Read-only accounting of archived Go authoring and checkpoint evaluation."""
import argparse
import difflib
import hashlib
import html
import json
from pathlib import Path
import shutil
import numpy as np


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def esc(value):return html.escape(str(value))


def program_metrics(path):
    raw=path.read_bytes();text=raw.decode('utf-8');lines=text.splitlines()
    return dict(bytes=len(raw),characters=len(text),lines=len(lines),
        nonblank_lines=sum(bool(x.strip()) for x in lines),sha256=digest(path))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--authoring',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    author=read(a.authoring/'results.json');evaluation=read(a.evaluation/'results.json')
    assert author['status']==evaluation['status']=='complete'
    assert evaluation['completed_games']==42000 and evaluation['historical_final_batches_matched']==189
    all_steps=[];sessions=[];arms={}
    for trial in author['trials']:
        session=dict(name=trial['name'],arm=trial['arm'],usage={},program_bytes=0,program_lines=0,
            completed_responses=len(trial['turns']),accepted_programs=0,collection_seconds=0.,construction_seconds=0.)
        for turn in trial['turns']:
            folder=a.authoring/trial['name']/f"turn_{turn['step']:02d}"
            row=dict(session=trial['name'],arm=trial['arm'],step=turn['step'],status=turn['status'],
                usage=turn['collection']['usage'],collection_seconds=turn['collection']['active_seconds'])
            for k,v in row['usage'].items():session['usage'][k]=session['usage'].get(k,0)+v
            session['collection_seconds']+=row['collection_seconds']
            source=folder/('proposal.py' if trial['arm']=='python' else 'proposal.wspl')
            if source.exists():
                row['program']=program_metrics(source);row['construction']=read(folder/'construction.json')
                session['program_bytes']+=row['program']['bytes'];session['program_lines']+=row['program']['lines']
                session['construction_seconds']+=row['construction']['seconds']
                session['accepted_programs']+=int(turn['status']=='accepted')
            all_steps.append(row)
        session['mean_win_rate']=author['evaluations'][trial['name']]['mean_win_rate'];sessions.append(session)
    for arm in ('python','wspl'):
        subset=[s for s in sessions if s['arm']==arm]
        arms[arm]={k:sum(s[k] for s in subset) for k in ('program_bytes','program_lines','completed_responses','accepted_programs','collection_seconds','construction_seconds')}
        arms[arm]['usage']={k:sum(s['usage'].get(k,0) for s in subset) for k in subset[0]['usage']}
        arms[arm]['mean_win_rate']=sum(s['mean_win_rate'] for s in subset)/3
    totals=dict(arms=arms,sessions=sessions,steps=all_steps,
        unknown_usage='One interrupted Python draw2 step0 call; excluded from reported usage totals.',
        definitions=dict(program_bytes='UTF-8 bytes of saved proposal files, including comments and line endings.',
            program_lines='splitlines count including comments and blank lines; cumulative sums count repeated code.',
            tokens='Provider-reported response usage; output includes reasoning. No tokenizer estimate for source alone.',
            operations='WSPL interpreter charged operations, not FLOPs or comparable Python operation counts.'))
    write(out/'authoring_metrics.json',totals)
    winner=[r for r in all_steps if r['session']=='2_wspl'];edits=[];previous=None;previous_source=''
    for i in range(7):
        folder=a.authoring/'2_wspl'/f'turn_{i:02d}';dest=out/f'edit_{i}';dest.mkdir()
        for name in ('proposal.wspl','answer.md','collection.json','construction.json'):
            shutil.copyfile(folder/name,dest/name)
        source=(folder/'proposal.wspl').read_text(encoding='utf-8')
        patch=''.join(difflib.unified_diff(previous_source.splitlines(True),source.splitlines(True),
            fromfile='previous_submitted_transaction' if i else 'empty',tofile=f'edit_{i}_submitted_transaction'))
        (dest/'submission.diff').write_text(patch,encoding='utf-8')
        checkpoint=folder/'constructed/candidate.npz'
        with np.load(checkpoint,allow_pickle=False) as z:current={k:z[k].copy() for k in z.files}
        changes=[];tensor_counts={}
        for k,arr in current.items():
            before=previous[k] if previous is not None else np.zeros_like(arr)
            indices=np.argwhere(before!=arr);tensor_counts[k]=len(indices)
            for ix in indices:
                t=tuple(ix);changes.append(dict(tensor=k,index=ix.tolist(),before=float(before[t]),after=float(arr[t])))
        write(dest/'weight_changes.json',changes)
        score=evaluation['evaluations'][f'edit_{i}']
        # Recompute scores from every saved episode, not just the summary JSON.
        for opp in range(3):
            wins=0;ids=[]
            for batch in sorted((a.evaluation/f'edit_{i}').glob(f's{opp}_batch_*.npz')):
                with np.load(batch,allow_pickle=False) as z:
                    wins+=int((z['reward']>0).sum());ids.extend(z['episode_ids'].tolist())
            assert sorted(ids)==list(range(2000)) and wins==score['summaries'][str(opp)]['wins']
        row=dict(winner[i],edit=i,checkpoint_sha256=digest(checkpoint),evaluation=score,
            changed_coefficients=len(changes),changed_by_tensor=tensor_counts,
            nonzero_coefficients=sum(int(np.count_nonzero(x)) for x in current.values()),
            contrast=evaluation.get('contrasts',{}).get(f'edit_{i}_minus_edit_{i-1}'))
        write(dest/'metrics.json',row);edits.append(row);previous=current;previous_source=source
    final=a.authoring/'2_wspl/turn_07';(out/'final_response').mkdir()
    for name in ('answer.md','collection.json'):shutil.copyfile(final/name,out/'final_response'/name)
    write(out/'checkpoint_metrics.json',edits)
    for name,path in [('authoring_results',a.authoring/'results.json'),('evaluation_results',a.evaluation/'results.json')]:
        shutil.copyfile(path,out/(name+'.json'))
    page=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Go neural-program edit review</title><style>
body{font:16px/1.55 system-ui,sans-serif;max-width:1250px;margin:36px auto;padding:0 22px;color:#172536;background:#f5f7fa}
h1,h2,h3{line-height:1.2}table{border-collapse:collapse;width:100%;background:white;margin:18px 0;font-variant-numeric:tabular-nums}
th,td{padding:9px 12px;border-bottom:1px solid #d8dee7;text-align:right}th:first-child,td:first-child{text-align:left}
article{background:white;border:1px solid #d8dee7;border-radius:8px;margin:24px 0;padding:20px}pre{overflow:auto;background:#101925;color:#e8eef7;padding:16px;font:13px/1.5 monospace;max-height:600px}
.add{color:#9aefad}.del{color:#ffabab}.meta{color:#a4c8ff}summary{cursor:pointer;font-weight:650;padding:10px 0}a{color:#125bb5}.scroll{overflow:auto}.note{color:#536277}
</style><h1>Go neural-program edit review</h1>
<p>Seven saved WSPL transactions, 42,000 CPU games, no gameplay feedback during authoring.
All 189 final-checkpoint batches exactly reproduce the original evaluation.</p>
<p class="note">Post-hoc analysis of the winning session on previously seen seeds. Intervals are descriptive,
not evidence over independent author replicates. No new games or model calls were used to build this report.</p>''']
    page.append('<h2>Checkpoint progression</h2><div class="scroll"><table><tr><th>Edit</th><th>Win rate</th><th>Change (pp)</th><th>Changed weights</th><th>Source bytes</th><th>Lines</th><th>Input tokens</th><th>Output tokens</th><th>Reasoning</th></tr>')
    for r in edits:
        delta='—' if r['contrast'] is None else f"{r['contrast']['difference']*100:+.2f}"
        page.append('<tr>'+''.join(f'<td>{v}</td>' for v in [f'<a href="#edit{r["edit"]}">{r["edit"]}</a>',f'{r["evaluation"]["mean_win_rate"]*100:.2f}%',delta,f'{r["changed_coefficients"]:,}',f'{r["program"]["bytes"]:,}',r['program']['lines'],f'{r["usage"]["input_tokens"]:,}',f'{r["usage"]["output_tokens"]:,}',f'{r["usage"].get("reasoning_output_tokens",0):,}'])+'</tr>')
    page.append('</table></div><h2>Whole-screen accounting</h2><p>Includes FINAL responses; excludes the interrupted call with unknown usage. Output tokens include reasoning. Source bytes include comments and original line endings; sums count all submitted transactions, including repeated code.</p><table><tr><th>Arm</th><th>Responses</th><th>Programs</th><th>Program bytes</th><th>Lines</th><th>Input</th><th>Output</th><th>Reasoning</th></tr>')
    for arm,s in arms.items():
        page.append('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in [arm,s['completed_responses'],s['accepted_programs'],f'{s["program_bytes"]:,}',s['program_lines'],f'{s["usage"]["input_tokens"]:,}',f'{s["usage"]["output_tokens"]:,}',f'{s["usage"]["reasoning_output_tokens"]:,}'])+'</tr>')
    page.append('</table><p><a href="authoring_metrics.json">Full per-call and per-session accounting</a> · <a href="checkpoint_metrics.json">Checkpoint metrics</a></p>')
    for r in edits:
        i=r['edit'];dest=out/f'edit_{i}';c=r['construction'];interval=r['contrast']
        page.append(f'<article id="edit{i}"><h2>Edit {i} · {r["evaluation"]["mean_win_rate"]*100:.2f}% win rate</h2>')
        page.append(f'<p>{r["changed_coefficients"]:,} coefficients changed; {r["nonzero_coefficients"]:,} nonzero afterward. Collection: {r["collection_seconds"]:.1f}s; construction: {c["seconds"]:.2f}s. WSPL charged operations: {c.get("resources",{}).get("operations",0):,}; placements: {c.get("placements",0)}.</p>')
        if interval:page.append(f'<p>Paired change: {interval["difference"]*100:+.2f} pp; descriptive 95% interval [{interval["descriptive_normal_ci95"][0]*100:+.2f}, {interval["descriptive_normal_ci95"][1]*100:+.2f}] pp. Won instead: {interval["improved"]:,}; lost instead: {interval["worsened"]:,}; unchanged: {interval["unchanged"]:,}.</p>')
        page.append(f'<p><a href="edit_{i}/proposal.wspl">Exact submitted program</a> · <a href="edit_{i}/answer.md">Raw response</a> · <a href="edit_{i}/weight_changes.json">All weight changes</a> · <a href="edit_{i}/metrics.json">Metrics and hashes</a></p>')
        page.append('<details><summary>Actual stored-weight changes (first 80; full JSON linked above)</summary><pre>')
        for change in read(dest/'weight_changes.json')[:80]:
            label=change['tensor']+'['+','.join(map(str,change['index']))+']'
            page.append(f'<span class="del">- {esc(label)} = {change["before"]}</span>\n<span class="add">+ {esc(label)} = {change["after"]}</span>\n')
        page.append('</pre></details><details><summary>Submitted transaction</summary><pre>'+esc((dest/'proposal.wspl').read_text())+'</pre></details>')
        page.append('<details><summary>Text diff against previous submitted transaction</summary><p class="note">These are separate patch programs. This text diff is NOT a cumulative controller diff; the stored-weight changes above show the actual state transition.</p><pre>')
        for line in (dest/'submission.diff').read_text().splitlines():
            cls='add' if line.startswith('+') else 'del' if line.startswith('-') else 'meta' if line.startswith('@') else ''
            page.append(f'<span class="{cls}">{esc(line)}</span>\n')
        page.append('</pre></details></article>')
    page.append('<p>Raw FINAL response and collection accounting are preserved under final_response/. Checkpoint weights and complete episode archives remain in the original local study directories. Weight comparisons use numeric FP32 inequality; signed-zero-only bit differences are not counted.</p></html>')
    (out/'index.html').write_text(''.join(page),encoding='utf-8')
    manifest={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}
    write(out/'artifact_hashes.json',manifest)
    print(json.dumps(dict(output=str(out),files=len(manifest),arms=arms,edits=[dict(edit=r['edit'],bytes=r['program']['bytes'],lines=r['program']['lines'],usage=r['usage']) for r in edits])))


if __name__=='__main__':main()
