"""Released Kinetix actor import and CPU replay primitives; no implicit run."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

LAYERS=('embed','actor0','actor1','actor2','actor3','actor4','actor_head')
WIDTHS=(679,128,128,128,128,128,128,16)
SHAPES={name+suffix:shape for name,a,b in zip(LAYERS,WIDTHS,WIDTHS[1:])
        for suffix,shape in (('.W',(a,b)),('.b',(b,)))}
ALIASES={name+suffix:kind+str(i) for i,name in enumerate(LAYERS,1) for suffix,kind in (('.W','W'),('.b','b'))}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_parent(text):
    """Read all integer-times-scale rows from the public prompt, never eval text."""
    section=text.split('THE WEIGHTS (complete actor).',1)[1]
    headers=list(re.finditer(r'(?m)^(embed|actor[0-4]|actor_head)\.(W|b) \((\d+)(?: x (\d+))?\), scale (1e-\d+):[^\n]*\n',section))
    arrays={}
    for i,h in enumerate(headers):
        name=h[1]+'.'+h[2];shape=tuple(int(x) for x in (h[3],h[4]) if x)
        if name in arrays or SHAPES.get(name)!=shape:raise ValueError('duplicate/malformed tensor header')
        body=section[h.end():headers[i+1].start() if i+1<len(headers) else len(section)].strip()
        if len(shape)==2:
            rows=body.splitlines()
            if len(rows)!=shape[0]:raise ValueError('matrix row count differs')
            values=[]
            for row,line in enumerate(rows):
                label,items=line.split(':',1)
                if label.strip()!=str(row):raise ValueError('matrix row order differs')
                items=items.split()
                if len(items)!=shape[1]:raise ValueError('matrix column count differs')
                values.extend(items)
        else:values=body.split()
        if len(values)!=int(np.prod(shape)) or any(not re.fullmatch(r'-?\d+',v) for v in values):
            raise ValueError('bounded integer tensor contents required')
        arrays[name]=(np.asarray(values,dtype=np.float64)*float(h[5])).reshape(shape)
    if set(arrays)!=set(SHAPES):raise ValueError('complete 14-tensor actor required')
    return arrays


def tensor_identity(arrays):
    return hashlib.sha256(b''.join(np.asarray(arrays[k],dtype='<f8').tobytes() for k in SHAPES)).hexdigest()


def imported_artifact(arrays):
    from .wsp_artifact import TensorArtifact
    return TensorArtifact.from_arrays({ALIASES[k]:np.asarray(v,dtype=np.float32).copy() for k,v in arrays.items()},
        graph=dict(kind='dense',widths=list(WIDTHS),activations=['relu']+['tanh']*5+['linear'],layout='in_out'))


def reconstruct_edit(parent,answer):
    from .native_weight_edits import statements
    from .wsp_editor import plan,apply
    blocks=re.findall(r'```edits\s*\n(.*?)```',answer,re.S)
    if len(blocks)!=1:raise ValueError('one published edit block required')
    rows=list(statements(blocks[0]));normalized=[]
    for row in rows:
        match=re.match(r'^(embed|actor[0-4]|actor_head)\.(W|b)(\[.*)',row,re.S)
        if not match:raise ValueError('declared actor tensor required')
        normalized.append(ALIASES[match[1]+'.'+match[2]]+match[3]+';')
    original=imported_artifact(parent)
    source=('wspl 0.2; target base=artifact("rounded_actor",sha256="'+original.sha256+'"); '
            'edit base as candidate mode=fixed numeric=native_numpy_v1 {\n'+'\n'.join(normalized)+'\n}')
    p=plan(source,parent=original);candidate=apply(p,parent=original,expected_parent_id=original.sha256)
    arrays=candidate.arrays()
    return {k:arrays[v] for k,v in ALIASES.items()},p


class KinetixPolicy:
    def __init__(self,mode,reference_root):
        if mode not in ('parent','edited'):raise ValueError('unknown reference policy')
        root=Path(reference_root);self.mode=mode
        if mode=='parent':
            self.weights=parse_parent((root/'runs/oneshot_repair/reply/prompt.txt').read_text(encoding='utf-8'))
        else:
            with np.load(root/'networks/edited_actor.npz',allow_pickle=False) as data:
                self.weights={k:data[k].copy() for k in data.files}
        if set(self.weights)!=set(SHAPES):raise ValueError('actor keys differ')
        for k,a in self.weights.items():
            if a.shape!=SHAPES[k] or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('actor shape/dtype/finite check failed')
            a.flags.writeable=False
        self.tensor_sha256=tensor_identity(self.weights)

    def logits(self,observation):
        x=np.asarray(observation,dtype=np.float32)
        if x.shape!=(679,) or not np.isfinite(x).all():raise ValueError('679 finite inputs required')
        for i,name in enumerate(LAYERS):
            x=x@self.weights[name+'.W']+self.weights[name+'.b']
            if i==0:x=np.maximum(x,0)
            elif i<6:x=np.tanh(x)
        return x


def reconstruct_native(parent,answer):
    """Replay data-only literal assignments in the released FP64 storage format."""
    import ast
    from .native_weight_edits import statements,index,value
    blocks=re.findall(r'```edits\s*\n(.*?)```',answer,re.S)
    if len(blocks)!=1:raise ValueError('one edit block required')
    result={k:v.copy() for k,v in parent.items()}
    rows=list(statements(blocks[0]))
    for row in rows:
        tree=ast.parse(row)
        if len(tree.body)!=1:raise ValueError('one statement required')
        n=tree.body[0]
        if isinstance(n,ast.Assign) and len(n.targets)==1:t=n.targets[0];op='set'
        elif isinstance(n,ast.AugAssign) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult)):
            t=n.target;op=type(n.op).__name__
        else:raise ValueError('literal tensor assignment required')
        if not isinstance(t,ast.Subscript) or not isinstance(t.value,ast.Attribute) or not isinstance(t.value.value,ast.Name):raise ValueError('tensor alias required')
        key=t.value.value.id+'.'+t.value.attr
        if key not in SHAPES:raise ValueError('unknown tensor')
        ix=t.slice.elts if isinstance(t.slice,ast.Tuple) else [t.slice]
        if len(ix)!=result[key].ndim:raise ValueError('rank differs')
        ix=tuple(index(v) for v in ix);v=value(n.value)
        if op=='set':result[key][ix]=v
        elif op=='Add':result[key][ix]+=v
        elif op=='Sub':result[key][ix]-=v
        else:result[key][ix]*=v
        if not np.isfinite(result[key]).all():raise ValueError('nonfinite edit')
    return result,len(rows)


def sample_action(logits,rng):
    logits=np.asarray(logits,dtype=np.float64)
    if logits.shape!=(16,) or not np.isfinite(logits).all():raise ValueError('16 finite logits required')
    action=[];offset=0
    for size in (3,3,3,3,2,2):
        z=logits[offset:offset+size];p=np.exp(z-z.max());p/=p.sum()
        action.append(min(int(np.searchsorted(np.cumsum(p),rng.random(),side='right')),size-1))
        offset+=size
    return np.asarray(action,dtype=np.int32)


class ReplayStudy:
    def __init__(self,reference_root):
        self.policies={m:KinetixPolicy(m,reference_root) for m in ('parent','edited')}
        answer=(Path(reference_root)/'runs/oneshot_repair/reply/answer.md').read_text(encoding='utf-8')
        rebuilt,n=reconstruct_native(self.policies['parent'].weights,answer)
        expected=self.policies['edited'].weights
        if n!=51 or any(not np.array_equal(rebuilt[k],expected[k]) for k in SHAPES):
            raise ValueError('published edit does not reconstruct released weights')
        self.reconstruction=dict(statements=n,parameters=sum(a.size for a in rebuilt.values()),
            numerical_mismatches=0,signed_zero_bit_differences=sum(int(np.count_nonzero(rebuilt[k].view('u8')!=expected[k].view('u8'))) for k in SHAPES))


def make_environment(level):
    from kinetix.util.saving import load_from_json_file
    from kinetix.environment.env import make_kinetix_env,ActionType,ObservationType
    state,static,params=load_from_json_file(str(level))
    if (static.num_polygons,static.num_circles,static.num_joints,static.num_thrusters,static.frame_skip)!=(12,4,6,2,2):
        raise ValueError('level dimensions or frame skip differ')
    params=params.replace(dense_reward_scale=.2,max_timesteps=256)
    env=make_kinetix_env(ActionType.MULTI_DISCRETE,ObservationType.SYMBOLIC_FLAT,None,params,static,auto_reset=False)
    return env,state,params
