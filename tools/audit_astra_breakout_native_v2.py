"""Read-only completion audit; no model requests, fitting or game steps."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

if len(sys.argv)!=2:raise SystemExit("Usage: python tools/audit_astra_breakout_native_v2.py CHECKOUT")
ROOT=Path(sys.argv[1]).resolve()
OUT=ROOT/"results/astra_breakout_native_v2"
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import verify
from skillforge.csail_breakout import TENSORS,dependency_identity,file_hash
from skillforge.native_weight_edits import NativeEditedPolicy
from skillforge.codex_one_shot import audit_events,audit_preview
import numpy as np

SHA='73bf70a24e92b978831aeca4998c44d049b6b2f39b4dda9544c5409b02b7fc46'
manifest=verify(ROOT,OUT/'approved_manifest.json',SHA)
config=manifest['runner_config']
for p,h in config['bound_files'].items():assert file_hash(ROOT/p)==h,p
assert dependency_identity()['sha256']==config['dependencies_sha256']
assert file_hash(json.loads((ROOT/config['local_paths']).read_text())['codex_cli'])==config['cli_sha256']
receipt=json.loads((OUT/'source_receipt.json').read_text())
result=json.loads((OUT/'results.json').read_text())
assert receipt['status']==result['status']=='complete'
assert receipt['constructor_calls']==3
assert result['planned_draws']==result['attempted_draws']==result['completed_evaluations']==3
assert json.loads((OUT/'status.json').read_text())==result
rows=[json.loads(x) for x in (OUT/'episodes.jsonl').read_text().splitlines()]
assert len(rows)==3000
trials=[];threads=set();totals={k:0 for k in ('input_tokens','cached_input_tokens','cache_write_input_tokens','output_tokens','reasoning_output_tokens')}
prompt=(ROOT/config['prompt']).read_text(encoding='utf-8')
for i,t in enumerate(result['trials']):
    assert t['draw']==i and t['status']=='complete'
    d=OUT/f'draw_{i}'
    events=[json.loads(x) for x in (d/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    answer=(d/'answer.md').read_text(encoding='utf-8')
    audit=audit_events(events,answer)
    assert audit['thread_id'] not in threads;threads.add(audit['thread_id'])
    assert audit['usage']==t['collection']['usage']
    assert audit['tool_calls']==0 and audit['agent_messages']==1
    for k in totals:totals[k]+=audit['usage'][k]
    assert (d/'prompt.txt').read_text(encoding='utf-8')==prompt
    audit_preview(json.loads((d/'prompt_preview.json').read_text(encoding='utf-8')),prompt)
    command=json.loads((d/'command.json').read_text())
    assert command[command.index('--model')+1]=='gpt-6-astra'
    assert 'model_reasoning_effort="xhigh"' in command
    policy=NativeEditedPolicy();parsed=policy.apply_answer(answer)
    assert parsed==t['parse'] and parsed['skipped']==0
    assert file_hash(d/'policy.npz')==t['weights_sha256']
    with np.load(d/'policy.npz',allow_pickle=False) as weights:
        assert set(weights.files)==set(TENSORS)
        for key,shape in TENSORS.items():
            assert weights[key].shape==shape and weights[key].dtype==np.float32
            assert np.array_equal(weights[key],policy.weights[key])
    records=[r for r in rows if r['draw']==i]
    assert len(records)==1000 and [r['index'] for r in records]==list(range(1000))
    for r in records:
        assert r['status']=='complete'
        assert r['env_seed']==510000+r['index'] and r['action_seed']==610000+r['index']
        assert type(r['steps']) is int and 1<=r['steps']<=10000
        assert math.isfinite(r['reward']) and r['reward']>=0
        assert len(r['action_counts'])==6 and all(type(v) is int and v>=0 for v in r['action_counts'])
        assert sum(r['action_counts'])==r['steps']
        assert type(r['terminal']) is bool and type(r['truncated']) is bool
        assert r['terminal']!=r['truncated']
        if r['truncated']:assert r['steps']==10000
    rewards=[r['reward'] for r in records]
    summary=dict(episodes=1000,mean=statistics.mean(rewards),se=statistics.stdev(rewards)/math.sqrt(1000),
                 min=min(rewards),max=max(rewards),mean_length=statistics.mean(r['steps'] for r in records),
                 truncations=sum(r['truncated'] for r in records))
    assert summary.keys()==t['summary'].keys()
    for k,v in summary.items():assert math.isclose(v,t['summary'][k],rel_tol=1e-12,abs_tol=1e-9),(i,k)
    trials.append(dict(draw=i,summary=summary,statements=parsed['applied'],nonzero=parsed['nonzero'],
        collection_seconds=t['collection']['active_seconds'],usage=audit['usage'],
        tensor_sha256=parsed['tensor_sha256'],weights_sha256=t['weights_sha256']))

names=['approved_manifest.json','source_receipt.json','results.json','status.json','episodes.jsonl']
for i in range(3):
    names += [f'draw_{i}/{name}' for name in ('answer.md','prompt.txt','prompt_preview.json','events.jsonl','stderr.log','preflight.stderr','command.json','collection.json','policy.npz')]
raw={name:file_hash(OUT/name) for name in names}
audit=dict(status='passed',manifest_sha256=SHA,source_files=len(manifest['files']),
    episodes_checked=len(rows),trials=trials,policy_mean_return=statistics.mean(t['summary']['mean'] for t in trials),
    policy_sample_sd=statistics.stdev(t['summary']['mean'] for t in trials),usage_totals=totals,
    reasoning_tokens_included_in_output_tokens=True,active_seconds=result['active_seconds'],
    collection_seconds=sum(t['collection_seconds'] for t in trials),
    checks=dict(source=True,bound_resources=True,dependencies=True,executable=True,receipts=True,
                unique_threads=True,one_turn_per_draw=True,tool_calls_zero=True,prompt_identity=True,
                preview_context=True,reply_to_npz_replay=True,all_episode_accounting=True,independent_statistics=True),
    prior_attempt=dict(invocations=1,completed_replies=0,evaluated_policies=0,usage='unknown',
        manifest_sha256='07eeeeda72b5bfc91ecb07ba94c4ee0593fcc3caee50577dde2542f2c87f9f33'),
    total_authoring_history=dict(invocations=4,completed_replies=3),raw_file_sha256=raw,
    audit_script_sha256=file_hash(__file__),
    limits=['three exploratory authoring draws','episode SE is conditional on a fixed policy',
            'prefix caching is not shared conversation state','CLI previews are not server attestations',
            'no WSPL comparison, gradient baseline or formal controller proof'])
(OUT/'completion_audit_reproducible.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
assert all(file_hash(OUT/name)==h for name,h in raw.items())
verify(ROOT,OUT/'approved_manifest.json',SHA)
print(json.dumps({k:v for k,v in audit.items() if k not in ('raw_file_sha256','trials')},indent=2))
