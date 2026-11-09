"""Bounded recipe-driven learning of one attention key on cached queries."""
import json
import numpy as np

DEFAULT=dict(positive_target=.75,negative_target=.02,negative_weight=3.,retention_weight=3.,ridge=.001,worst_negative=1.,head_variance=0.,anchor='initial')
LIMITS=dict(positive_target=(.5,.95),negative_target=(.001,.1),negative_weight=(1.,100.),retention_weight=(0.,100.),ridge=(0.,.1),worst_negative=(0.,100.),head_variance=(0.,10.))


def validate_recipe(recipe):
    if not isinstance(recipe,dict) or set(recipe)!=set(DEFAULT):raise ValueError('exact recipe fields required')
    for name,(low,high) in LIMITS.items():
        x=recipe[name]
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or not low<=x<=high:raise ValueError('invalid '+name)
    if recipe['anchor'] not in ('initial','zero'):raise ValueError('anchor')
    return dict(recipe)


def parse_recipe(answer):
    text=answer.strip()
    if text.startswith('```json\n') and text.endswith('```'):text=text[8:-3]
    if len(text.encode())>4096:raise ValueError('recipe too large')
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('duplicate field')
            d[k]=v
        return d
    return validate_recipe(json.loads(text,object_pairs_hook=unique))


def nested_indices(labels,seed,sizes):
    labels=np.asarray(labels)
    # Equal representation of four mechanisms, nested and independent of outcomes.
    classes=sorted(set(labels.tolist()))
    if len(classes)!=4 or any(n%4 for n in sizes):raise ValueError('four balanced classes required')
    rng=np.random.default_rng(seed);orders={c:rng.permutation(np.flatnonzero(labels==c)) for c in classes}
    if any(len(v)<max(sizes)//4 for v in orders.values()):raise ValueError('insufficient examples')
    return {n:np.asarray([orders[c][i] for i in range(n//4) for c in classes]) for n in sizes}


def fit_key(q,logz,applicable,base_correct,initial,bound,recipe,steps=1000,check=lambda:None):
    import torch
    recipe=validate_recipe(recipe)
    arrays=[np.asarray(a) for a in (q,logz,applicable,base_correct,initial)]
    if not all(np.isfinite(a).all() for a in arrays):raise ValueError('nonfinite data')
    q,z,y,correct,k0=arrays
    if q.ndim!=3 or q.shape[1:]!=(3,64) or z.shape!=q.shape[:2] or y.shape!=(len(q),) or correct.shape!=y.shape or k0.shape!=(64,):raise ValueError('shape')
    if not 0<bound or not np.isfinite(bound) or np.linalg.norm(k0)>bound or steps<1:raise ValueError('bound/steps')
    if not set(y.tolist())<= {0,1,False,True} or not set(correct.tolist())<={0,1,False,True} or not np.any(y) or np.all(y):raise ValueError('binary labels and both classes required')
    Q=torch.tensor(q,dtype=torch.float64)/8;Z=torch.tensor(z,dtype=torch.float64)
    Y=torch.tensor(y,dtype=torch.bool);C=torch.tensor(correct,dtype=torch.bool)
    key=torch.nn.Parameter(torch.tensor(k0,dtype=torch.float64));anchor=key.detach().clone() if recipe['anchor']=='initial' else torch.zeros_like(key)
    target=torch.where(Y,recipe['positive_target'],recipe['negative_target']).double()[:,None].expand(-1,3)
    weights=torch.where(Y,1.,recipe['negative_weight']).double()
    weights=weights+((~Y)&C).double()*recipe['retention_weight']
    opt=torch.optim.Adam([key],lr=.03);trace=[]
    for step in range(steps):
        if step%100==0:check()
        opt.zero_grad();logits=Q@key-Z;mass=torch.sigmoid(logits)
        bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none').mean(1)
        loss=(bce*weights).sum()/weights.sum()
        loss=loss+recipe['ridge']*((key-anchor)**2).mean()
        loss=loss+recipe['worst_negative']*mass[~Y].max(dim=1).values.square().mean()
        loss=loss+recipe['head_variance']*mass.var(dim=1,unbiased=False).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite fit')
        loss.backward();torch.nn.utils.clip_grad_norm_([key],1.);opt.step()
        with torch.no_grad():key.mul_(torch.clamp(torch.tensor(bound,dtype=key.dtype)/torch.linalg.vector_norm(key),max=1.))
        if step%100==0 or step==steps-1:trace.append(dict(step=step+1,pre_update_loss=float(loss.detach()),norm=float(key.detach().norm())))
    result=key.detach().numpy().astype(np.float32)
    if np.linalg.norm(result.astype(np.float64))>bound:result*=np.float32((bound*(1-1e-6))/np.linalg.norm(result.astype(np.float64)))
    return result,trace


def reward(report,baseline):
    m=report['metrics'];b=baseline['metrics']
    gain=(m['positive_correct']-b['positive_correct'])/m['positive_n']
    retained=m['initially_correct_negative_retained'];eligible=m['initially_correct_negative_n']
    damage=(eligible-retained)/max(eligible,1)
    # Reward is fixed by evaluator, never author-controlled.
    score=gain-2*damage-.5*m['negative_mean_mass']
    return dict(value=score,positive_accuracy_gain=gain,negative_damage_rate=damage,negative_mean_mass=m['negative_mean_mass'],retention_n=eligible)


def best_trial(trials):
    valid=[t for t in trials if t['status']=='accepted']
    if not valid:return dict(recipe=dict(DEFAULT),selection='predeclared fallback: no valid proposals')
    best=max(valid,key=lambda t:(t['reward']['value'],-t['step']))
    return dict(recipe=best['recipe'],selection='highest development reward, earliest tie',step=best['step'],reward=best['reward'])
