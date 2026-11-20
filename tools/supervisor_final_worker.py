"""Host-only final evaluation; start only after the supervisor has submitted."""
import json
import os
from pathlib import Path
import sys
import signal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ['MUJOCO_GL']='osmesa';os.environ['PYOPENGL_PLATFORM']='osmesa'
os.environ['NUMBA_CACHE_DIR']='/tmp/wspl-supervisor-evaluator-numba'
from skillforge.supervisor_evaluation import evaluate_bundle
from skillforge.supervisor_workspace import save


def main():
    config=json.loads(Path(sys.argv[1]).read_text())
    signal.alarm(config.get('final_guard_seconds',14400))
    missions=json.loads(Path(config['missions']).read_text())
    all_rows=[]
    for arm,bundle in config['bundles'].items():
        rows=evaluate_bundle(bundle,missions,config,Path(config['output'])/arm)
        all_rows.extend(dict(arm=arm,**r) for r in rows)
    for i in range(len(missions)):
        assert len({r['start_hash'] for r in all_rows if r['mission']==i})==1
    summary={}
    for r in all_rows:
        s=summary.setdefault(r['arm']+'/'+r['condition'],dict(n=0,success=0,probes=0))
        s['n']+=1;s['success']+=int(r['success']);s['probes']+=r['probes']
    baseline={r['mission']:r for r in all_rows if r['arm']=='ordinary_practice'}
    proposed={r['mission']:r for r in all_rows if r['arm']=='submission'}
    paired={}
    for c in ('ordinary','changed','switch','return'):
        indexes=[i for i,r in baseline.items() if r['condition']==c]
        paired[c]=dict(improvements=sum(proposed[i]['success'] and not baseline[i]['success'] for i in indexes),
                       regressions=sum(baseline[i]['success'] and not proposed[i]['success'] for i in indexes))
    gain=sum(v['improvements']-v['regressions'] for k,v in paired.items() if k!='ordinary')
    gate=(summary['ordinary_practice/ordinary']['success']>=6 and paired['ordinary']['regressions']==0
          and gain>=4 and all(v['improvements']>=v['regressions'] for k,v in paired.items() if k!='ordinary'))
    save(Path(config['output'])/'results.json',dict(status='complete',summary=summary,paired=paired,
        exploratory_gate=gate,rows=all_rows,claim='System performance; student attribution requires component audit/ablations.'))


if __name__=='__main__':main()
