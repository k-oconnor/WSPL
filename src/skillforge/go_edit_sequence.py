"""Frozen edit-sequence inputs and paired, descriptive outcome accounting."""
from pathlib import Path
import numpy as np
from skillforge.go_authoring import read_arrays
from skillforge.go_replay import sha,load_actor


class GoEditSequence:
    def __init__(self,checkpoints,opponent_root):
        if list(checkpoints)!=[f'edit_{i}' for i in range(7)]:raise ValueError('all seven ordered checkpoints required')
        self.actors={}
        for name,row in checkpoints.items():
            if sha(row['path'])!=row['sha256']:raise ValueError('checkpoint changed: '+name)
            arrays=read_arrays(row['path'])
            self.actors[name]=tuple((arrays[f'W{i}'],arrays[f'b{i}']) for i in range(1,8))
        self.opponents=[load_actor(Path(opponent_root)/f'ppo_same_size_lr3e-5_s{i}') for i in range(3)]


def validate_batch(result,ids):
    n=len(ids)
    if (np.asarray(result['reward']).shape!=(n,) or np.any(result['invalid'])
        or not np.all(result['terminal']) or not np.isin(result['reward'],[-1,1]).all()):
        raise ValueError('invalid or incomplete policy execution')


def paired_difference(before,after):
    a=np.asarray(before);b=np.asarray(after)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[0]!=3 or a.shape[1]<2:
        raise ValueError('three matched opponent strata required')
    if not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all():raise ValueError('binary wins required')
    d=b.astype(float)-a.astype(float)
    mean=float(d.mean());se=float(np.sqrt(np.var(d,axis=1,ddof=1).sum()/d.shape[1]/9))
    return dict(difference=mean,per_opponent=d.mean(axis=1).tolist(),
        improved=int((d>0).sum()),worsened=int((d<0).sum()),unchanged=int((d==0).sum()),
        descriptive_normal_ci95=[mean-1.959963984540054*se,mean+1.959963984540054*se])
