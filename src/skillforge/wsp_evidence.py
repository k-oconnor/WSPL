"""Frozen descriptions and scoped real-graph checks for WSP tensor artifacts.

No admission, no numerical-execution proof, and no semantic meaning inferred from
names. The symbolic encoder reads the actual stored bits, not a source compiler.
"""
import ast
from dataclasses import dataclass
from fractions import Fraction
import json
import re

from .wsp_artifact import TensorArtifact, canonical, digest, fraction
from .wsp_syntax import Reader, expression
from .wsp_values import Arithmetic, scalar, Limits


def _constant(node, text):
    segment = ast.get_source_segment(text, node)
    if segment is None:
        raise ValueError('constant expression source missing')
    q = scalar(Arithmetic({}, {}, {}, {}, Limits(max_operations=10_000)).run(segment))
    if not isinstance(q, Fraction):
        raise ValueError('rational constant required')
    return q


def _expr(node, text, env, z3=None):
    def walk(n): return _expr(n, text, env, z3)
    def rational(q): return z3.RealVal(str(q)) if z3 else q
    def conjunction(values): return z3.And(*values) if z3 else all(values)
    def disjunction(values): return z3.Or(*values) if z3 else any(values)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return rational(_constant(node, text))
    if isinstance(node, ast.Name):
        if node.id in ('true', 'false'):
            return node.id == 'true'
        if node.id not in env:
            raise ValueError(f'unknown description signal {node.id}')
        return env[node.id]
    if isinstance(node, ast.List):
        return [walk(x) for x in node.elts]
    if isinstance(node, ast.Subscript):
        values = walk(node.value)
        if not isinstance(values, list):
            raise ValueError('indexed signal must be a vector')
        q = _constant(node.slice, text)
        if q.denominator != 1 or not -len(values) <= q < len(values):
            raise ValueError('description index out of bounds')
        return values[int(q)]
    if isinstance(node, ast.UnaryOp):
        a = walk(node.operand)
        if isinstance(node.op, ast.USub): return -a
        if isinstance(node.op, ast.UAdd): return a
        if isinstance(node.op, ast.Not): return z3.Not(a) if z3 else not a
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Mult, ast.Div)):
            try:
                right = _constant(node.right, text)
                if isinstance(node.op, ast.Div):
                    if not right: raise ValueError('division by zero')
                    return walk(node.left)/rational(right)
                return walk(node.left)*rational(right)
            except ValueError:
                if isinstance(node.op, ast.Div):
                    raise ValueError('claim division requires nonzero constant denominator')
                try: left = _constant(node.left, text)
                except ValueError as exc: raise ValueError('UNSUPPORTED: nonlinear claim multiplication') from exc
                return rational(left)*walk(node.right)
        a, b = walk(node.left), walk(node.right)
        if isinstance(node.op, ast.Add): return a+b
        if isinstance(node.op, ast.Sub): return a-b
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name = node.func.id
        if name=='tap' and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):
            key='node:'+node.args[0].value
            if key not in env:raise ValueError('unknown graph node tap')
            return env[key]
        if name in ('pre','post') and len(node.args) == 1:
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                key = ('pre_' if name=='pre' else '')+arg.id
                if key not in env: raise ValueError('unknown tap')
                return env[key]
            if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
                key = ('pre_' if name=='pre' else '')+arg.value.id
                q = _constant(arg.slice, text)
                if key not in env or q.denominator!=1 or not -len(env[key]) <= q < len(env[key]):
                    raise ValueError('invalid tap index')
                return env[key][int(q)]
            raise ValueError('pre/post needs a signal or indexed signal')
        args = [walk(a) for a in node.args]
        def minimum(a,b): return z3.If(a <= b, a, b) if z3 else min(a,b)
        def maximum(a,b): return z3.If(a >= b, a, b) if z3 else max(a,b)
        if name == 'relu' and len(args)==1: return maximum(args[0], rational(Fraction(0)))
        if name == 'min' and len(args)==2: return minimum(*args)
        if name == 'max' and len(args)==2: return maximum(*args)
        if name == 'clip' and len(args)==3:
            lo,hi = _constant(node.args[1],text), _constant(node.args[2],text)
            if lo > hi: raise ValueError('clip bounds reversed')
            return minimum(maximum(args[0], rational(lo)), rational(hi))
    if isinstance(node, ast.Compare):
        left = walk(node.left); results = []
        for op,right_node in zip(node.ops,node.comparators):
            right = walk(right_node)
            if isinstance(op, ast.In):
                if not isinstance(right, list) or len(right)!=2: raise ValueError('in denotes a closed interval [lo,hi]')
                lo,hi = [_constant(n,text) for n in right_node.elts]
                if lo > hi: raise ValueError('interval bounds reversed')
                result = conjunction([left >= rational(lo), left <= rational(hi)])
            elif isinstance(op, ast.Eq): result = left == right
            elif isinstance(op, ast.NotEq): result = left != right
            elif isinstance(op, ast.Lt): result = left < right
            elif isinstance(op, ast.LtE): result = left <= right
            elif isinstance(op, ast.Gt): result = left > right
            elif isinstance(op, ast.GtE): result = left >= right
            else: raise ValueError('unsupported comparison')
            results.append(result); left = right
        return conjunction(results)
    if isinstance(node, ast.BoolOp):
        values = [walk(n) for n in node.values]
        return conjunction(values) if isinstance(node.op, ast.And) else disjunction(values)
    raise ValueError('unsupported description expression')


def evaluate(text, env, z3=None):
    return _expr(expression(text).body, text, env, z3)


def _with_views(env, views, z3=None):
    env = dict(env)
    for name,text in views.items():
        if name in env or name in {'relu','clip','min','max','pre','post','tap','true','false'}:
            raise ValueError('view shadows a signal or builtin')
        env[name] = evaluate(text, env, z3)
    return env


def _dummy(artifact):
    graph = artifact.to_dict()['graph']
    if graph['kind']=='dag':
        from .wsp_graph import validate_graph
        meta=validate_graph(graph,artifact.to_dict()['tensors'])
        env={k:[Fraction(0)]*v for k,v in graph['inputs'].items()}
        env.update({k:[Fraction(0)]*v for k,v in meta['widths'].items() if k.startswith('node:')})
        env.update({k:[Fraction(0)]*v for k,v in meta['output_widths'].items()})
        return env
    if graph['kind'] != 'dense':
        raise ValueError('UNSUPPORTED: descriptions need a signal graph adapter')
    env = {'x':[Fraction(0)]*graph['widths'][0]}
    for i,width in enumerate(graph['widths'][1:],1):
        name = 'y' if i==len(graph['widths'])-1 else f'h{i}'
        env[name] = [Fraction(0)]*width; env['pre_'+name] = [Fraction(0)]*width
    return env


@dataclass(frozen=True)
class Description:
    _text: str

    @property
    def sha256(self): return digest(self.to_dict())

    def to_dict(self): return json.loads(self._text)

    def save(self, path):
        with open(path,'x',encoding='utf-8',newline='\n') as f:
            f.write(canonical(dict(description_id=self.sha256, body=self.to_dict()))+'\n')

    @classmethod
    def load(cls,path):
        with open(path,encoding='utf-8') as f: text=f.read(2_000_001)
        if len(text)>2_000_000: raise ValueError('description file too large')
        d=json.loads(text)
        if set(d)!= {'description_id','body'} or digest(d['body'])!=d['description_id']:
            raise ValueError('description identity differs')
        return cls(canonical(d['body']))


def describe(artifact, source, *, origin='exploratory'):
    if origin not in ('exploratory','external_requirement'):
        raise ValueError('description origin must be explicit')
    if isinstance(source,str):
        reader=Reader(source); d=reader.description()
        if reader.tok.text!='<EOF>': raise ValueError('unexpected description suffix')
    else:
        d=json.loads(canonical(source))
    if not isinstance(d,dict) or set(d)!= {'target','views','hypotheses','claims'}:
        raise ValueError('invalid description structure')
    if (not isinstance(d['target'],str) or not isinstance(d['views'],dict)
            or not all(isinstance(k,str) and re.fullmatch('[A-Za-z_][A-Za-z_0-9]*',k) and isinstance(v,str) for k,v in d['views'].items())
            or not isinstance(d['hypotheses'],list) or not all(isinstance(v,str) for v in d['hypotheses'])
            or not isinstance(d['claims'],list)):
        raise ValueError('description field types differ')
    # Validate syntax and signal addressing; no assertion is made from dummy values.
    env=_with_views(_dummy(artifact),d['views'])
    names=set()
    for claim in d['claims']:
        if (not isinstance(claim,dict) or set(claim)!= {'name','semantics','assumptions','ensures'}
                or not isinstance(claim['name'],str) or not isinstance(claim['semantics'],str)
                or not isinstance(claim['assumptions'],list) or not isinstance(claim['ensures'],list)
                or not all(isinstance(x,str) for x in claim['assumptions']+claim['ensures'])
                or not claim['ensures'] or claim['name'] in names):
            raise ValueError('invalid/duplicate claim')
        names.add(claim['name'])
        for text in claim['assumptions']+claim['ensures']:
            value=evaluate(text,env)
            if type(value) is not bool:
                raise ValueError('assume/ensure must be Boolean formulas')
    return Description(canonical(dict(format='wsp.description.v1',artifact_id=artifact.sha256,origin=origin,
                                      content=d, executable_changes=0, hypotheses_status='UNVERIFIED')))


def _symbolic(artifact,z3):
    graph=artifact.to_dict()['graph']; arrays=artifact.arrays(bits=True)
    if graph['kind']=='dag':return _symbolic_graph(artifact,z3)
    values=[z3.Real(f'x_{i}') for i in range(graph['widths'][0])]; env={'x':values}
    for i,act in enumerate(graph['activations'],1):
        w,b=arrays[f'W{i}'],arrays[f'b{i}']
        if graph['layout']=='in_out': w=w.T
        pre=[z3.Sum([z3.RealVal(str(fraction(c)))*x for c,x in zip(row,values) if fraction(c)])+z3.RealVal(str(fraction(bias))) for row,bias in zip(w,b)]
        values=[z3.If(v>=0,v,z3.RealVal(0)) for v in pre] if act=='relu' else pre
        name='y' if i==len(graph['activations']) else f'h{i}'
        env['pre_'+name]=pre; env[name]=values
    return env


def _symbolic_graph(artifact,z3):
    from .wsp_graph import validate_graph
    d=artifact.to_dict();g=d['graph'];meta=validate_graph(g,d['tensors']);arrays=artifact.arrays(bits=True)
    env={k:[z3.Real(f'input_{k}_{i}') for i in range(width)] for k,width in g['inputs'].items()}
    values={'input:'+k:v for k,v in env.items()};nodes={n['id']:n for n in g['nodes']}
    def q(bits):return z3.RealVal(str(fraction(bits)))
    for key in meta['order']:
        n=nodes[key];v={p:values[s] for p,s in n['inputs'].items()};op=n['op']
        if op=='affine':
            w=arrays[n['params']['weight']];b=arrays[n['params']['bias']]
            if n['attrs']['layout']=='in_out':w=w.T
            out=[z3.Sum([q(c)*x for c,x in zip(row,v['x']) if fraction(c)])+q(bias) for row,bias in zip(w,b)]
        elif op=='scale':out=[q(arrays[n['params']['gain']][0])*x for x in v['x']]
        elif op=='add':out=[a+b for a,b in zip(v['a'],v['b'])]
        elif op=='concat':out=v['a']+v['b']
        elif op=='slice':out=v['x'][slice(n['attrs']['start'],n['attrs']['stop'],n['attrs']['step'])]
        elif op=='relu':out=[z3.If(x>=0,x,z3.RealVal(0)) for x in v['x']]
        elif op=='identity':out=list(v['x'])
        else:raise ValueError('UNSUPPORTED: graph formal operator')
        values['node:'+key]=out;env['node:'+key]=out
    env.update({k:values[s] for k,s in g['outputs'].items()})
    return env


def _input_terms(artifact,env):
    g=artifact.to_dict()['graph']
    return [x for k in g['inputs'] for x in env[k]] if g['kind']=='dag' else env['x']


def _input_record(artifact,flat):
    g=artifact.to_dict()['graph']
    if g['kind']!='dag':return flat
    result={};offset=0
    for name,width in g['inputs'].items():
        result[name]=flat[offset:offset+width];offset+=width
    return result


def _strings(value):
    if isinstance(value,dict):return {k:_strings(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_strings(v) for v in value]
    return str(value)


def _witness(model, inputs):
    out=[]
    for x in inputs:
        q=model.eval(x,model_completion=True)
        out.append(Fraction(q.numerator_as_long(),q.denominator_as_long()))
    return out


def check(artifact, description, *, timeout_ms=5000, max_parameters=100_000):
    d=description.to_dict()
    base=dict(artifact_id=artifact.sha256,description_id=description.sha256,
              scope='stored-FP32-coefficients interpreted as a rational real graph; not IEEE execution',
              admission='not_performed',hypotheses_status='UNVERIFIED',results=[])
    if not isinstance(d,dict) or d.get('artifact_id')!=artifact.sha256 or d.get('format')!='wsp.description.v1':
        return dict(base,status='INVALID',reason='description artifact identity differs')
    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 60_000:
        return dict(base,status='INVALID',reason='timeout must be 1..60000 ms per solver query')
    try:
        checked=describe(artifact,d['content'],origin=d['origin'])
        if checked.sha256!=description.sha256: raise ValueError('description fields differ')
    except (ValueError,TypeError,KeyError) as exc:
        return dict(base,status='INVALID',reason=str(exc))
    base['origin']=d['origin']
    claims=d['content']['claims']
    if not claims:
        return dict(base,status='NO_CLAIMS')
    graph=artifact.to_dict()['graph']
    if graph['kind']=='dag':
        from .wsp_graph import validate_graph
        meta=validate_graph(graph,artifact.to_dict()['tensors'])
        supported=meta['formal_supported']
        if meta['live_values']>100_000:return dict(base,status='UNKNOWN',reason='formal graph activation budget exceeded')
    else:supported=graph['kind']=='dense' and 'tanh' not in graph['activations']
    if not supported:
        return dict(base,status='UNSUPPORTED',reason='formal backend requires affine/ReLU graph')
    if sum(a.size for a in artifact.arrays().values())>max_parameters:
        return dict(base,status='UNKNOWN',reason='formal parameter budget exceeded')
    try: import z3
    except ImportError: return dict(base,status='UNSUPPORTED',reason='optional Z3 unavailable')
    env=_with_views(_symbolic(artifact,z3),d['content']['views'],z3)
    base['solver']=dict(name='Z3',version=z3.get_version_string(),timeout_ms_per_query=timeout_ms)
    from pathlib import Path
    base['checker_sha256']=hashlib_sha(Path(__file__).read_text(encoding='utf-8'))
    base['evaluator_sha256']=hashlib_sha(Path(__file__).with_name('wsp_artifact.py').read_text(encoding='utf-8'))
    for claim in claims:
        result=dict(name=claim['name'],claim_id=digest(claim))
        if claim['semantics']!='real_graph':
            base['results'].append(dict(result,status='UNSUPPORTED',reason='requested semantics unsupported'));continue
        try:
            assumptions=[evaluate(x,env,z3) for x in claim['assumptions']]
            ensures=[evaluate(x,env,z3) for x in claim['ensures']]
            solver=z3.SolverFor('QF_LRA');solver.set(timeout=timeout_ms);solver.add(*assumptions)
            domain=solver.check()
            if domain==z3.unsat:
                base['results'].append(dict(result,status='INVALID',reason='vacuous assumptions'));continue
            if domain==z3.unknown:
                base['results'].append(dict(result,status='UNKNOWN',reason=solver.reason_unknown()));continue
            feasible=_input_record(artifact,_witness(solver.model(),_input_terms(artifact,env)))
            replay_domain=_with_views(artifact.trace(feasible),d['content']['views'])
            if not all(evaluate(x,replay_domain) for x in claim['assumptions']):
                base['results'].append(dict(result,status='INVALID',reason='nonvacuity witness failed replay'));continue
            solver.add(z3.Not(z3.And(*ensures)))
            result['query_sha256']=hashlib_sha(solver.sexpr())
            outcome=solver.check()
            if outcome==z3.unsat:
                result.update(status='PROVED',nonvacuity_witness=_strings(feasible),nonvacuity_replayed=True)
            elif outcome==z3.unknown:
                result.update(status='UNKNOWN',reason=solver.reason_unknown())
            else:
                witness=_input_record(artifact,_witness(solver.model(),_input_terms(artifact,env)))
                actual=_with_views(artifact.trace(witness),d['content']['views'])
                replay=all(evaluate(x,actual) for x in claim['assumptions']) and not all(evaluate(x,actual) for x in claim['ensures'])
                outputs={k:actual[k] for k in graph['outputs']} if graph['kind']=='dag' else actual['y']
                result.update(status='REFUTED' if replay else 'INVALID',inputs=_strings(witness),outputs=_strings(outputs),witness_replayed=bool(replay))
        except (ValueError,TypeError,IndexError,ZeroDivisionError,z3.Z3Exception) as exc:
            result.update(status='INVALID',reason=str(exc))
        base['results'].append(result)
    statuses={r['status'] for r in base['results']}
    base['status']=next((s for s in ('INVALID','REFUTED','UNKNOWN','UNSUPPORTED') if s in statuses),'PROVED')
    return base


def hashlib_sha(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def probe(artifact, inputs, *, semantics='real_graph', max_rows=128):
    if not isinstance(inputs,list) or len(inputs)>max_rows:
        raise ValueError('bounded explicit input fixture required')
    rows=[]
    for values in inputs:
        taps=artifact.trace(values,semantics=semantics)
        rows.append({name:list(map(str,items)) for name,items in taps.items()})
    return dict(artifact_id=artifact.sha256,input_fixture_id=digest(_strings(inputs)),
                semantics=semantics,status='EMPIRICAL',samples=len(rows),taps=rows,universal_claim=False)


def equations(artifact, *, layer=None, node_id=None, offset=0, limit=16):
    graph=artifact.to_dict()['graph']
    if graph['kind']=='dag':
        from .wsp_graph import validate_graph
        meta=validate_graph(graph,artifact.to_dict()['tensors'])
        matches=[n for n in graph['nodes'] if n['id']==node_id]
        if not matches:raise ValueError('graph description needs a valid --node ID')
        n=matches[0]
        return dict(artifact_id=artifact.sha256,semantics='stored_graph_operator',node=n,
                    output_width=meta['widths']['node:'+node_id],parameter_identities={k:digest(artifact.to_dict()['tensors'][v]) for k,v in n['params'].items()},
                    semantic_interpretation='not_established',executable_changes=0)
    if graph['kind']!='dense' or type(layer) is not int or not 1 <= layer < len(graph['widths']):
        raise ValueError('valid dense layer required')
    if type(offset) is not int or offset<0 or type(limit) is not int or not 0<=limit<=256:
        raise ValueError('invalid equation page')
    arrays=artifact.arrays(bits=True);w,b=arrays[f'W{layer}'],arrays[f'b{layer}']
    if graph['layout']=='in_out':w=w.T
    source='x' if layer==1 else f'h{layer-1}'
    target='y' if layer==len(graph['activations']) else f'h{layer}'
    rows=[]
    for i in range(offset,min(offset+limit,len(b))):
        rows.append(dict(output=f'{target}[{i}]',activation=graph['activations'][layer-1],bias=str(fraction(b[i])),
                         terms=[dict(signal=f'{source}[{j}]',coefficient=str(fraction(c))) for j,c in enumerate(w[i]) if fraction(c)]))
    return dict(artifact_id=artifact.sha256,semantics='stored_coefficient_equations',equations=rows,
                semantic_interpretation='not_established',executable_changes=0)
