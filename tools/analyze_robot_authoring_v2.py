"""Read-only saved-artifact analysis. No model construction, fitting or inference."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

MAIN = Path(__file__).resolve().parents[1]
STUDY = MAIN.parent / 'WSPL-robot-authoring-v2'
ROOT = STUDY / 'results/robot_authoring_v2'
sys.path[:0] = [str(STUDY / 'src'), str(STUDY / 'experiments')]
from source_consistency import verify
from skillforge.robot_submissions import verify_frozen
from skillforge.robot_trials import compare

def read(p):
    return json.loads(p.read_text())

def main():
    result = read(ROOT / 'results.json')
    pin = result['manifest_sha256']
    source = verify(STUDY, STUDY / 'experiments/source_pins/robot_authoring_v2.manifest.json', pin)
    assert read(ROOT / 'source_receipt.json')['status'] == 'complete'
    assert read(ROOT.with_suffix('.exit.json'))['returncode'] == 0
    ledger = read(ROOT / 'all_frozen.json')
    parent = verify_frozen(ledger['parent'])
    episodes = {a: read(ROOT / ('final_' + a) / 'episodes.json') for a in ledger}
    report = dict(manifest_sha256=pin, source_files_verified=len(source['files']),
                  scope='Saved episode/decision records, frozen arrays and final WSPL binding audit; no new model forwards. Complete semantic audit of every author script is not claimed.',
                  elapsed_seconds=result['elapsed_seconds'], arms={})
    for arm, entry in ledger.items():
        bank = verify_frozen(entry)
        rows = episodes[arm]
        assert len(rows) == 64
        paired = compare(episodes['parent'], rows)
        if arm != 'parent': assert paired == result['comparisons'][arm]
        decisions = []
        for i, row in enumerate(rows):
            saved = read(ROOT / ('final_' + arm) / f'mission_{i:03d}' / 'result.json')
            ds = read(ROOT / ('final_' + arm) / f'mission_{i:03d}' / 'decisions.json')
            assert all(row[k] == v for k, v in saved.items())
            assert set(row)-set(saved) == {'mission','model_calls','invalid_tokens'}
            assert row['mission'] == i
            assert len(ds) == row['model_calls']
            assert sum(bool(d['invalid']) for d in ds) == row['invalid_tokens']
            decisions.extend(ds)
        changed = [k for k in bank if not np.array_equal(bank[k].view(np.uint32), parent[k].view(np.uint32))]
        counts = {c: v['candidate'] for c,v in paired['conditions'].items()}
        g = episodes['gradient']
        data = dict(counts=counts, ordinary=counts['ordinary'], adaptation=sum(counts[c] for c in ('changed','switch','return')),
                    total=sum(counts.values()), gate=paired['exploratory_gate'], changed_tensors=changed,
                    changed_parameters=sum(bank[k].size for k in changed), invalid_tokens=sum(d['invalid'] for d in decisions),
                    vs_gradient=dict(author_only=sum(a['success'] and not b['success'] for a,b in zip(rows,g)),
                                     gradient_only=sum(b['success'] and not a['success'] for a,b in zip(rows,g))))
        if arm not in ('parent','gradient'):
            transcript=read(ROOT/arm/'transcript.json')
            dev=[x['content']['development']['summary'] for x in transcript if x['role']=='tool' and 'development' in x['content']]
            data['development']={c:x['successes'] for c,x in dev[-1].items()}
            usages=[read(p) for p in sorted((ROOT/arm).glob('turn_*/usage.json'))]
            data['calls']=len(usages)
            data['tokens']={k:sum(u['usage'].get(k,0) for u in usages) for k in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')}
            data['author_seconds']=sum(u['active_seconds'] for u in usages)
            replies=[x['content'] for x in transcript if x['role']=='tool']
            data['python_seconds']=sum(x.get('seconds',0) for x in replies)
            data['python_errors']=sum(x.get('exit_code',0)!=0 for x in replies)
            data['evaluation_attempts']=read(ROOT/arm/'frozen.json')['proposal_attempts']
            requests=sorted((ROOT/arm/'sandbox/requests').glob('[0-9][0-9][0-9].py'))
            data['python_sources']=[dict(file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in requests]
            data['syntax_errors']=[]; calls=set()
            for p in requests:
                try: tree=ast.parse(p.read_text())
                except SyntaxError: data['syntax_errors'].append(p.name);continue
                calls.update(ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call))
            data['method_call_screen']=[c for c in sorted(calls) if any(x in c.lower() for x in ('optim','backward','grad','linalg','exec'))]
            if arm.startswith('wspl'):
                folder=Path(entry['file']).parent
                sub=read(folder/'submission.json');bindings=np.load(folder/sub['bindings'],allow_pickle=False)
                data['wspl_programs']=sub['programs']
                import re
                for projection,body in sub['programs'].items():
                    matches=re.fullmatch(r'A\[:,\s*:\]\s*=\s*(\w+)\s*;\s*B\[:,\s*:\]\s*=\s*(\w+)\s*;',body)
                    assert matches,body
                    for factor,key in zip(('A','B'),matches.groups()):
                        assert np.array_equal(bank[projection+'.lora_'+factor].view(np.uint32),bindings[key].view(np.uint32))
                data['wspl_classification']='Whole-matrix assignment of supplied Python-fitted bindings; exact final binding equality verified.'
        report['arms'][arm]=data
    report['groups']={kind:{k:sum(report['arms'][kind+str(i)][k] for i in range(3))/3 for k in ('ordinary','adaptation','total','author_seconds','python_seconds','calls')} for kind in ('native','wspl')}
    report['gate_passes']=result['failure_adjusted_gate_passes']
    report['author_adaptation_union']=sum(any(episodes[a][i]['success'] for a in ledger if a not in ('parent','gradient')) for i in range(16,64))
    report['author_adaptation_unique_vs_gradient']=[i for i in range(16,64) if not episodes['gradient'][i]['success'] and any(episodes[a][i]['success'] for a in ledger if a not in ('parent','gradient'))]
    target=MAIN/'docs/wspl/robot_authoring_v2_analysis.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='arms'},indent=2))
    for arm,d in report['arms'].items():print(arm,json.dumps({k:v for k,v in d.items() if k not in ('python_sources','method_call_screen')}))

if __name__=='__main__':main()
