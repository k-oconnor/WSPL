"""Frozen-value native key authoring: supplied query diagnostics and held-out metrics."""
import numpy as np
from .attention_probe import prefix_score,validate


def features(study,rows):
    queries=[];partitions=[];scores=[]
    for row in rows:
        capture={};scores.append(prefix_score(study,row,capture=capture))
        q=capture['Q'][2][:3].astype(np.float64)
        k=capture['native_K'][2][0].astype(np.float64)
        logits=q@k.T/8;peak=logits.max(axis=-1)
        queries.append(q);partitions.append(peak+np.log(np.exp(logits-peak[:,None]).sum(axis=-1)))
    return np.asarray(queries,np.float32),np.asarray(partitions,np.float32),scores


def validate_key(bank,reference):
    validate(bank,reference,'retrieval',reference)
    # Only the active slot is writable, not unused key vectors.
    off=reference['alpha']==0
    if bank['K'][off].tobytes()!=reference['K'][off].tobytes():raise ValueError('inactive keys are frozen')


def metrics(rows,records,baseline):
    if len(rows)!=len(records) or len(rows)!=len(baseline):raise ValueError('unaligned records')
    if any(r['id']!=s['id'] or r['id']!=b['id'] for r,s,b in zip(rows,records,baseline)):raise ValueError('IDs differ')
    positive=[r['answerKey']==r['choices']['label'][r['choices']['text'].index('conduction')] for r in rows]
    mass=[float(np.mean(s['virtual_mass'][0]['mass'][0])) if s['virtual_mass'] else 0. for s in records]
    pos=[m for m,p in zip(mass,positive) if p];neg=[m for m,p in zip(mass,positive) if not p]
    if not pos or not neg:raise ValueError('both applicability classes required')
    auc=float(np.mean([float(a>b)+.5*float(a==b) for a in pos for b in neg]))
    retention=[(r,s,b) for r,s,b,p in zip(rows,records,baseline,positive) if not p and b['correct']]
    corrected=[r['id'] for r,s,b in zip(rows,records,baseline) if s['correct'] and not b['correct']]
    broken=[r['id'] for r,s,b in zip(rows,records,baseline) if not s['correct'] and b['correct']]
    by_category={c:dict(n=sum(r['category']==c for r in rows),correct=sum(s['correct'] for r,s in zip(rows,records) if r['category']==c)) for c in sorted({r['category'] for r in rows})}
    settings={c:dict(n=sum(r.get('setting')==c for r in rows),correct=sum(s['correct'] for r,s in zip(rows,records) if r.get('setting')==c)) for c in sorted({r['setting'] for r in rows if 'setting' in r})}
    coverage={c:sum(r['category']==c for r,_,_ in retention) for c in by_category}
    return dict(categories=by_category,settings=settings,initially_correct_negative_coverage=coverage,corrected=corrected,broken=broken,positive_correct=sum(s['correct'] for s,p in zip(records,positive) if p),
                positive_mean_mass=float(np.mean(pos)),negative_mean_mass=float(np.mean(neg)),mass_gap=float(np.mean(pos)-np.mean(neg)),routing_auc=auc,
                positive_mass_at_least_half=sum(m>=.5 for m in pos),positive_n=len(pos),negative_mass_at_most_tenth=sum(m<=.1 for m in neg),negative_n=len(neg),
                initially_correct_negative_n=len(retention),initially_correct_negative_retained=sum(s['correct'] for _,s,_ in retention))


def evaluate(study,rows,bank,mode,baseline,check=lambda:None):
    records=[]
    for row in rows:check();records.append(prefix_score(study,row,bank,mode))
    return dict(records=records,metrics=metrics(rows,records,baseline))


def descriptive_gate(report,base_report):
    m=report['metrics'];b=base_report['metrics']
    enough=m['initially_correct_negative_n']>=4
    return dict(retention_assessable=enough,
        selective=m['positive_mean_mass']>=.5 and m['negative_mean_mass']<=.1 and m['routing_auc']>=.9,
        useful=m['positive_correct']>b['positive_correct'] and len(m['corrected'])>len(m['broken']),
        retains_initially_correct_negatives=enough and m['initially_correct_negative_retained']==m['initially_correct_negative_n'],
        note='Descriptive exploratory criteria, not statistical significance or proof of a concept.')
