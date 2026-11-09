"""Read-only frozen-result audit. No proposer, optimizer, solver rerun or game steps."""
from collections import Counter
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

if len(sys.argv)!=2:raise SystemExit('Usage: python tools/audit_breakout_matched_v1.py CHECKOUT')
ROOT=Path(sys.argv[1]).resolve();OUT=ROOT/'results/breakout_matched_v1'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import verify
from skillforge.breakout_matched import MatchedBreakoutPolicy,matched_dependency_identity,INPUTS,wspl_body,NODE_LIMIT
from skillforge.codex_one_shot import audit_events,audit_preview
from skillforge.csail_breakout import TENSORS,file_hash
from skillforge.wspl_artifact import ReLUArtifact,canonical_bytes
from skillforge.wspl_compile import compile_scalar
from skillforge.wspl_core import ScalarProgram
import numpy as np

SHA='9771cbf74140ac577d366ccd935139d9efe1e2f6016cf91dc385c78d7d2256b7'
manifest=verify(ROOT,OUT/'approved_manifest.json',SHA);config=manifest['runner_config']
for p,h in config['bound_files'].items():assert file_hash(ROOT/p)==h,p
assert matched_dependency_identity()['sha256']==config['dependencies_sha256']
assert json.loads((OUT/'dependencies.json').read_text())['sha256']==config['dependencies_sha256']
assert file_hash(json.loads((ROOT/config['local_paths']).read_text())['codex_cli'])==config['cli_sha256']
result=json.loads((OUT/'results.json').read_text());receipt=json.loads((OUT/'source_receipt.json').read_text())
assert result['status']==receipt['status']=='complete' and receipt['constructor_calls']==9
assert result==json.loads((OUT/'status.json').read_text())
assert result['planned_invocations']==result['attempted_invocations']==9
order=[('weights',0),('code',0),('wspl',0),('code',1),('wspl',1),('weights',1),('wspl',2),('weights',2),('code',2)]
assert [(t['arm'],t['draw']) for t in result['trials']]==order
rows=[json.loads(x) for x in (OUT/'episodes.jsonl').read_text().splitlines()]
assert len(rows)==8000
threads=set();trials=[];failure={};usage={k:0 for k in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')}
rng=np.random.default_rng(20261104)
probes=[np.zeros(600,dtype=np.float32),np.ones(600,dtype=np.float32)]+[rng.integers(0,2,size=600).astype(np.float32) for _ in range(2)]
names=['approved_manifest.json','source_receipt.json','results.json','status.json','episodes.jsonl','dependencies.json']
for t in result['trials']:
    arm,i=t['arm'],t['draw'];name=f'{arm}_{i}';d=OUT/name
    answer=(d/'answer.md').read_text(encoding='utf-8')
    events=[json.loads(x) for x in (d/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    audit=audit_events(events,answer);assert audit['thread_id'] not in threads;threads.add(audit['thread_id'])
    assert audit['agent_messages']==1 and audit['tool_calls']==0
    assert audit['usage']==t['collection']['usage']
    for k in usage:usage[k]+=audit['usage'][k]
    prompt=(ROOT/config['prompts'][arm]).read_text(encoding='utf-8')
    assert (d/'prompt.txt').read_text(encoding='utf-8')==prompt
    audit_preview(json.loads((d/'prompt_preview.json').read_text(encoding='utf-8')),prompt)
    command=json.loads((d/'command.json').read_text())
    assert command[command.index('--model')+1]=='gpt-6-astra' and 'model_reasoning_effort="xhigh"' in command
    names += [name+'/'+n for n in ('answer.md','prompt.txt','prompt_preview.json','events.jsonl','stderr.log','preflight.stderr','command.json','collection.json')]
    records=[r for r in rows if r['arm']==arm and r['draw']==i]
    policy=MatchedBreakoutPolicy()
    if t['status']=='invalid_proposal':
        assert name=='wspl_1' and not records and not (d/'policy.npz').exists()
        try:policy.load_answer(arm,answer)
        except ValueError as exc:assert str(exc)=='source graph exceeds two 32-wide hidden layers'
        else:raise AssertionError('rejected proposal unexpectedly accepted')
        body=re.findall(r'```wspl[^\S\r\n]*\r?\n(.*?)```',answer,re.S)[-1]
        program,expanded=wspl_body(body)
        compiled=compile_scalar(program,max_layers=2,max_units=128,max_parameters=100000,max_nodes=NODE_LIMIT).to_dict()
        allocation=compiled['source_map']['allocation'][0]
        zero_columns=[j for j,u in enumerate(allocation) if u['kind']=='input_negative']
        for j in zero_columns:
            row=compiled['layers'][0]['weights'][j];bias=compiled['layers'][0]['bias'][j]
            source_index=int(allocation[j]['source'][1:])
            assert all(Q(v)==(-1 if k==source_index else 0) for k,v in enumerate(row)) and Q(bias)==0
            assert Q(program['inputs'][INPUTS[source_index]]['lower'])>=0
        failure=dict(stage='fixed-target allocation; parsing and interval validation succeeded',
            compiled_allocation=compiled['allocation'],
            unit_kinds_by_layer=[dict(Counter(u['kind'] for u in layer)) for layer in compiled['source_map']['allocation']],
            negative_input_relays_algebraically_zero_on_declared_domain=len(zero_columns),
            first_layer_width_after_removing_only_those_zero_units=len(allocation)-len(zero_columns),
            diagnostic_scope='algebraic capacity observation only; no patched deployment or game evaluation',
            interpretation='current allocator needs46 first-layer units; this is not a proof of mathematical nonrepresentability')
        trials.append(dict(arm=arm,draw=i,status=t['status'],error=t['error'],failure_adjusted_score=0.))
        continue
    assert t['status']=='complete' and len(records)==1000
    construction=policy.load_answer(arm,answer);assert construction==t['construction']
    saved=ReLUArtifact.load(d/'deployment.relu.json',expected_sha256=construction['artifact_sha256'])
    assert saved.sha256==policy.artifact.sha256
    assert file_hash(d/'policy.npz')==t['weights_sha256']
    with np.load(d/'policy.npz',allow_pickle=False) as w:
        assert set(w.files)==set(TENSORS)
        for key,shape in TENSORS.items():
            assert w[key].shape==shape and w[key].dtype==np.float32 and np.array_equal(w[key],policy.weights[key])
    assert [r['index'] for r in records]==list(range(1000))
    for r in records:
        assert r['status']=='complete' and r['env_seed']==710000+r['index'] and r['action_seed']==810000+r['index']
        assert type(r['steps']) is int and 1<=r['steps']<=10000 and math.isfinite(r['reward']) and r['reward']>=0
        assert len(r['action_counts'])==6 and all(type(v) is int and v>=0 for v in r['action_counts'])
        assert sum(r['action_counts'])==r['steps']
        assert type(r['terminal']) is bool and type(r['truncated']) is bool and r['terminal']!=r['truncated']
        if r['truncated']:assert r['steps']==10000
    rewards=[r['reward'] for r in records]
    summary=dict(episodes=1000,mean=statistics.mean(rewards),se=statistics.stdev(rewards)/math.sqrt(1000),
        min=min(rewards),max=max(rewards),mean_length=statistics.mean(r['steps'] for r in records),truncations=sum(r['truncated'] for r in records))
    assert summary.keys()==t['summary'].keys()
    for k,v in summary.items():assert math.isclose(v,t['summary'][k],rel_tol=1e-12,abs_tol=1e-9),(name,k)
    if arm!='weights':
        assert json.loads((d/'source.ir.json').read_text())==policy.program
        if arm=='wspl':assert (d/'expanded.wspl').read_text(encoding='utf-8')==policy.expanded_source
        assert json.loads((d/'source_equivalence.json').read_text())==t['source_equivalence']
        assert t['construction']['frontend']['rounding']['coefficients_changed']==0
        source=ScalarProgram(policy.program,max_nodes=NODE_LIMIT)
    max_abs=0.
    for x in probes:
        inputs={n:str(int(v)) for n,v in zip(INPUTS,x)}
        exact=saved.evaluate(inputs)
        if arm!='weights':assert source.evaluate(inputs)==exact
        floats=policy.logits(x)
        max_abs=max(max_abs,max(abs(float(exact[f'a{j}'])-float(floats[j])) for j in range(6)))
    trials.append(dict(arm=arm,draw=i,status='complete',summary=summary,construction=construction,
        source_equivalence_status=t['source_equivalence']['status'],synthetic_probe_count=4,
        synthetic_fp32_vs_rational_max_absolute_difference=max_abs))
    names += [name+'/'+n for n in ('policy.npz','deployment.relu.json')]
    if arm!='weights':names += [name+'/'+n for n in ('source.ir.json','source_equivalence.json')]
    if arm=='wspl':names.append(name+'/expanded.wspl')

arms={}
for arm in ('weights','code','wspl'):
    selected=[t for t in result['trials'] if t['arm']==arm]
    valid=[t['summary']['mean'] for t in selected if t['status']=='complete']
    scores=[t['summary']['mean'] if t['status']=='complete' else 0. for t in selected]
    measured=dict(evaluated=len(valid),invalid=3-len(valid),failure_adjusted_mean=statistics.mean(scores),
                  valid_only_mean=statistics.mean(valid),valid_policy_sample_sd=statistics.stdev(valid),policy_scores=scores)
    recorded=result['comparisons']['arms'][arm]
    for k,v in measured.items():
        if isinstance(v,list):assert v==recorded[k]
        else:assert math.isclose(v,recorded[k],rel_tol=1e-12,abs_tol=1e-9)
    measured['reported_usage']={k:sum(t['collection']['usage'][k] for t in selected) for k in usage}
    assert measured['reported_usage']==recorded['reported_usage']
    measured['collection_seconds']=sum(t['collection']['active_seconds'] for t in selected)
    arms[arm]=measured
assert result['comparisons']['internal_feasibility_decision']=='no_go_for_this_configuration'
for baseline in ('weights','code'):
    assert arms[baseline]['failure_adjusted_mean']>=100
    ratio=arms['wspl']['failure_adjusted_mean']/arms[baseline]['failure_adjusted_mean']
    assert math.isclose(ratio,result['comparisons']['wspl_to_baseline_ratios'][baseline],rel_tol=1e-12)
assert arms['wspl']['evaluated']<3 and arms['wspl']['failure_adjusted_mean']<100
raw={name:file_hash(OUT/name) for name in names}
audit=dict(status='passed',manifest_sha256=SHA,source_files=len(manifest['files']),episodes_checked=8000,
    planned_proposals=9,completed_replies=9,evaluated_policies=8,invalid_policies=1,
    arms=arms,trials=trials,allocation_failure=failure,usage_totals=usage,
    reasoning_tokens_included_in_output_tokens=True,active_seconds=result['active_seconds'],
    collection_seconds=sum(t['collection']['active_seconds'] for t in result['trials']),
    internal_feasibility_decision=result['comparisons']['internal_feasibility_decision'],
    checks=dict(source=True,bound_resources=True,core_and_solver_dependencies=True,executable=True,
        receipts=True,nine_unique_threads=True,one_turn_per_reply=True,no_tool_calls=True,prompts_and_previews=True,
        reply_to_saved_weights=True,episode_accounting=True,independent_statistics=True,arm_aggregations=True,
        original_failure_reproduced=True,four_synthetic_rational_probes_per_policy=True),
    synthetic_probe_scope='no game states or new scored episodes; finite checks do not upgrade UNKNOWN proofs',
    raw_file_sha256=raw,audit_script_sha256=file_hash(__file__))
(OUT/'completion_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
assert all(file_hash(OUT/n)==h for n,h in raw.items())
verify(ROOT,OUT/'approved_manifest.json',SHA)
print(json.dumps({k:v for k,v in audit.items() if k not in ('raw_file_sha256','trials')},indent=2))
