"""Three data-only front ends for the same fixed FP32 ReLU Breakout target."""
import ast
from fractions import Fraction as Q
import hashlib
import json
import math
import re

from .csail_breakout import TENSORS,dependency_identity,file_hash
from .native_weight_edits import NativeEditedPolicy
from .wspl_artifact import ReLUArtifact,canonical_bytes
from .wspl_compile import compile_scalar
from .wspl_parse import parse_module

ARMS=('weights','code','wspl')
INPUTS=tuple(f'x{i:03}' for i in range(600))
OUTPUTS=tuple(f'a{i}' for i in range(6))
BOUND=10**30
NODE_LIMIT=50000
WIDTHS=[600,32,32,6]


def matched_dependency_identity():
    """Include the solver's code and loaded-library files, not only its version."""
    from pathlib import Path
    import z3
    root=Path(z3.__file__).parent
    body=dict(core=dependency_identity(),solver_version=z3.get_full_version(),
        solver_files={p.relative_to(root).as_posix():file_hash(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in ('.py','.pyd','.dll','.so')})
    body['sha256']=hashlib.sha256(canonical_bytes(body)).hexdigest()
    return body


def ty(lo,hi):return dict(kind='Real',unit='1',lower=str(lo),upper=str(hi))
def ref(name):return dict(op='ref',name=name)
def constant(q):return dict(op='const',value=str(q),unit='1')
def affine(terms,bias=0):return dict(op='affine',terms=[dict(coefficient=str(c),value=v) for c,v in terms],bias=str(bias),unit='1')


def wspl_body(body):
    header='module Breakout {\n'+''.join(f'input {n}: Real[0,1] normalized_value;\n' for n in INPUTS)
    header+=''.join(f'output {n}: Real[-{BOUND},{BOUND}] normalized_value;\n' for n in OUTPUTS)
    parsed=parse_module(header+body+'\n}',max_nodes=NODE_LIMIT)
    if parsed.contracts:raise ValueError('candidate contracts are not study requirements; omit them')
    if tuple(parsed.program['inputs'])!=INPUTS or [o['name'] for o in parsed.program['outputs']]!=list(OUTPUTS):
        raise ValueError('fixed port interface cannot change')
    return parsed.program,header+body+'\n}'


def code_body(body):
    """Ordinary straight-line Python syntax, parsed into IR, never executed."""
    tree=ast.parse(body,mode='exec')
    if sum(1 for _ in ast.walk(tree))>200000:raise ValueError('code AST budget exceeded')
    env={n:(ref(n),Q(0),Q(1)) for n in INPUTS};lets=[];outputs=None
    def literal(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            token=ast.get_source_segment(body,node)
            if not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)',token or ''):raise ValueError('plain decimal constants only')
            q=Q(token)
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            q=literal(node.operand)*(-1 if isinstance(node.op,ast.USub) else 1)
        elif isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            a,b=literal(node.left),literal(node.right)
            if isinstance(node.op,ast.Add):q=a+b
            elif isinstance(node.op,ast.Sub):q=a-b
            elif isinstance(node.op,ast.Mult):q=a*b
            else:q=a/b
        else:raise ValueError('constant expression required')
        if abs(q)>BOUND:raise ValueError('constant budget exceeded')
        return q
    def scale(a,q):
        v,lo,hi=a
        if q==1:return a
        if q==0:return constant(0),Q(0),Q(0)
        if v['op']=='affine':
            value=affine([(q*Q(t['coefficient']),t['value']) for t in v['terms']],q*Q(v['bias']))
        else:value=affine([(q,v)])
        return value,min(lo*q,hi*q),max(lo*q,hi*q)
    def visit(node,depth=0):
        if depth>32:raise ValueError('expression nesting budget')
        if isinstance(node,ast.Name) and node.id in env:return env[node.id]
        try:
            q=literal(node);return constant(q),q,q
        except ValueError:pass
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            return scale(visit(node.operand,depth+1),-1 if isinstance(node.op,ast.USub) else 1)
        if isinstance(node,ast.BinOp):
            if isinstance(node.op,(ast.Add,ast.Sub)):
                # Flatten long ordinary Python sums, like the WSPL affine parser.
                stack=[(node,1)];terms=[];lo=hi=Q(0)
                while stack:
                    n,sign=stack.pop()
                    if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub)):
                        stack.append((n.right,sign*(-1 if isinstance(n.op,ast.Sub) else 1)));stack.append((n.left,sign))
                    else:
                        a=scale(visit(n,depth+1),sign);terms.append((1,a[0]));lo+=a[1];hi+=a[2]
                return affine(terms),lo,hi
            if isinstance(node.op,ast.Mult):
                try:return scale(visit(node.right,depth+1),literal(node.left))
                except ValueError:return scale(visit(node.left,depth+1),literal(node.right))
            if isinstance(node.op,ast.Div):return scale(visit(node.left,depth+1),1/literal(node.right))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
            op=node.func.id
            if op=='relu' and len(node.args)==1:
                a=visit(node.args[0],depth+1);return dict(op='relu',value=a[0]),max(Q(0),a[1]),max(Q(0),a[2])
            if op=='clip' and len(node.args)==3:
                a=visit(node.args[0],depth+1);lo,hi=literal(node.args[1]),literal(node.args[2])
                if lo>hi:raise ValueError('invalid clip')
                clamp=lambda x:min(hi,max(lo,x))
                return dict(op='clip',value=a[0],lower=str(lo),upper=str(hi)),clamp(a[1]),clamp(a[2])
            if op in ('min','max') and len(node.args)==2:
                a,b=(visit(n,depth+1) for n in node.args);f=min if op=='min' else max
                return dict(op=op,left=a[0],right=b[0]),f(a[1],b[1]),f(a[2],b[2])
        raise ValueError('unsupported Python subset; no execution/imports/attributes/indexing/loops/branches')
    for i,node in enumerate(tree.body):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in env or name in OUTPUTS or name in ('relu','clip','min','max') or not re.fullmatch('[a-z][a-z0-9_]*',name):raise ValueError('fresh scalar variable required')
            value,lo,hi=visit(node.value)
            if not -BOUND<=lo<=hi<=BOUND:raise ValueError('intermediate interval budget')
            lets.append(dict(name=name,type=ty(lo,hi),value=value));env[name]=(ref(name),lo,hi)
        elif isinstance(node,ast.Return) and i==len(tree.body)-1 and isinstance(node.value,(ast.List,ast.Tuple)) and len(node.value.elts)==6:
            outputs=[dict(name=n,type=ty(-BOUND,BOUND),value=visit(v)[0]) for n,v in zip(OUTPUTS,node.value.elts)]
        else:raise ValueError('scalar assignments followed by one six-value return required')
    if outputs is None:raise ValueError('six-value final return required')
    return dict(format='wspl.scalar_ir.v0',inputs={n:ty(0,1) for n in INPUTS},lets=lets,outputs=outputs)


def fixed_artifact(artifact):
    """Embed a depth <=2 source graph in 600-32-32-6, preserving signed values."""
    d=artifact.to_dict();hidden=d['layers'][:-1];last=d['layers'][-1]
    if len(hidden)>2 or any(len(l['bias'])>32 for l in hidden):raise ValueError('source graph exceeds two 32-wide hidden layers')
    if not hidden:
        # Relay positive and negative parts of each affine output, not 600 inputs.
        w=[];b=[]
        for row,bias in zip(last['weights'],last['bias']):
            for sign in (1,-1):w.append([str(sign*Q(x)) for x in row]);b.append(str(sign*Q(bias)))
        hidden=[dict(activation='relu',weights=w,bias=b)]
        last=dict(activation='linear',weights=[['1' if j==2*i else '-1' if j==2*i+1 else '0' for j in range(12)] for i in range(6)],bias=['0']*6)
    layers=[];previous=600
    for layer in hidden:
        rows=[row+['0']*(previous-len(row)) for row in layer['weights']]
        rows += [['0']*previous for _ in range(32-len(rows))]
        layers.append(dict(activation='relu',weights=rows,bias=layer['bias']+['0']*(32-len(layer['bias']))));previous=32
    while len(layers)<2:
        layers.append(dict(activation='relu',weights=[['1' if i==j else '0' for j in range(32)] for i in range(32)],bias=['0']*32))
    layers.append(dict(activation='linear',weights=[row+['0']*(32-len(row)) for row in last['weights']],bias=last['bias']))
    d.update(layers=layers,allocation=dict(mode='fresh_mlp',widths=WIDTHS,parameters=20486),
             source_map=dict(original_compilation=d['source_map'],padding='signed output / nonnegative hidden relays; fixed 2x32'))
    return ReLUArtifact(d)


def deployment_artifact(artifact):
    """Explicit FP32 coefficient rounding, retained as exact rationals thereafter."""
    import numpy as np
    d=artifact.to_dict();changed=0;maximum=Q(0)
    def convert(x):
        nonlocal changed,maximum
        q=Q(x)
        if abs(q)>1000000:raise ValueError('common coefficient magnitude limit exceeded')
        stored=float(np.float32(float(q)))
        if not math.isfinite(stored):raise ValueError('nonfinite FP32 coefficient')
        actual=Q.from_float(stored)
        if actual!=q:changed+=1;maximum=max(maximum,abs(actual-q))
        return str(actual)
    for l in d['layers']:
        l['weights']=[[convert(x) for x in row] for row in l['weights']];l['bias']=[convert(x) for x in l['bias']]
    d['compiler']+=';explicit_fp32_rounding.v1'
    d['source_map']={'pre_rounding_artifact_sha256':artifact.sha256,'rounding':'nearest FP32 via float64 conversion',
                     'coefficients_changed':changed,'max_absolute_change':str(maximum),'prior':d['source_map']}
    return ReLUArtifact(d),dict(coefficients_changed=changed,max_absolute_change=str(maximum),
        guarantee='stored FP32 coefficients represented exactly as rationals; arithmetic execution not proved')


def weight_artifact(policy,source_hash):
    layers=[];values=list(TENSORS)
    for i in range(3):
        w=policy.weights[values[2*i]];b=policy.weights[values[2*i+1]]
        layers.append(dict(activation='linear' if i==2 else 'relu',weights=[[str(Q.from_float(float(x))) for x in row] for row in w.T],bias=[str(Q.from_float(float(x))) for x in b]))
    return ReLUArtifact(dict(format='wspl.relu_mlp.v0',semantics='rational_real_graph',compiler='matched.native_fp32.v1',
        source_sha256=source_hash,inputs=[dict(name=n,type=ty(0,1)) for n in INPUTS],outputs=[dict(name=n,type=ty(-BOUND,BOUND)) for n in OUTPUTS],
        layers=layers,allocation=dict(mode='fresh_mlp',widths=WIDTHS,parameters=20486),source_map={}))


class MatchedBreakoutPolicy(NativeEditedPolicy):
    def load_answer(self,arm,answer):
        import numpy as np
        if arm not in ARMS or not isinstance(answer,str) or len(answer.encode())>1000000:raise ValueError('bounded reply and declared arm required')
        self.program=None;self.expanded_source=None;self.artifact=None
        if arm=='weights':
            info=self.apply_answer(answer)
            self.artifact=weight_artifact(self,hashlib.sha256(answer.encode()).hexdigest())
        else:
            if self._edited:raise ValueError('one reply per policy')
            self._edited=True
            fence='python' if arm=='code' else 'wspl'
            blocks=re.findall(r'```'+fence+r'[^\S\r\n]*\r?\n(.*?)```',answer,flags=re.S)
            if not blocks:raise ValueError('missing '+fence+' block')
            if arm=='code':self.program=code_body(blocks[-1])
            else:self.program,self.expanded_source=wspl_body(blocks[-1])
            compiled=compile_scalar(self.program,max_layers=2,max_units=128,max_parameters=100000,max_nodes=NODE_LIMIT)
            self.artifact,rounding=deployment_artifact(fixed_artifact(compiled))
            values=list(TENSORS)
            for i,l in enumerate(self.artifact.to_dict()['layers']):
                self.weights[values[2*i]]=np.array([[float(Q(x)) for x in row] for row in l['weights']],dtype=np.float32).T.copy()
                self.weights[values[2*i+1]]=np.array([float(Q(x)) for x in l['bias']],dtype=np.float32)
            for a in self.weights.values():a.flags.writeable=False
            info=dict(rounding=rounding,source_ir_sha256=hashlib.sha256(canonical_bytes(self.program)).hexdigest())
        self.mode='matched_relu_'+arm
        return dict(arm=arm,parameters=20486,nonzero=sum(int((a!=0).sum()) for a in self.weights.values()),
                    tensor_sha256=self.tensor_identity(),artifact_sha256=self.artifact.sha256,frontend=info)

    def logits(self,x):
        import numpy as np
        x=np.asarray(x,dtype=np.float32)
        if x.shape!=(600,) or not np.isfinite(x).all():raise ValueError('600 finite inputs required')
        values=list(TENSORS);h=x
        for i in range(3):
            h=h@self.weights[values[2*i]]+self.weights[values[2*i+1]]
            if i<2:h=np.maximum(h,np.float32(0))
        if not np.isfinite(h).all():raise ValueError('nonfinite ReLU logits')
        return h
