"""Explicit stateless vector DAGs, boundary analysis and functional graph edits.

Every parameter reference names canonical FP32 storage. Reusing that name is
explicit sharing; graph edits never infer or silently break such sharing.
"""
from dataclasses import dataclass, asdict
from fractions import Fraction
import copy
import heapq
import math
import re

import numpy as np

from .wsp_artifact import TensorArtifact, fraction, round_f32, digest


IDENT = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')
OPS = ('affine', 'relu', 'tanh', 'identity', 'add', 'scale', 'concat', 'slice')
MAX_NODES = 4096
MAX_VALUES = 2_000_000


def identifier(value):
    return isinstance(value, str) and bool(IDENT.fullmatch(value))


def node(id, op, inputs, *, params=None, attrs=None):
    return dict(id=id, op=op, inputs=dict(inputs), params=dict(params or {}), attrs=dict(attrs or {}))


def edges(graph):
    result = {f'{n["id"]}.{port}': source for n in graph['nodes'] for port, source in n['inputs'].items()}
    result.update({f'output.{name}': source for name, source in graph['outputs'].items()})
    return result


def parameter_uses(graph):
    uses = {}
    for n in graph['nodes']:
        for port, name in n['params'].items():
            uses.setdefault(name, []).append(f'{n["id"]}.{port}')
    return {k: sorted(v) for k, v in sorted(uses.items())}


def validate_graph(graph, tensors):
    """Validate all edges/parameters and infer vector widths in a stable topo order."""
    if not isinstance(graph, dict) or set(graph) != {'kind', 'inputs', 'nodes', 'outputs'} or graph['kind'] != 'dag':
        raise ValueError('DAG requires kind, inputs, nodes and outputs')
    ins, outs, nodes = graph['inputs'], graph['outputs'], graph['nodes']
    if (not isinstance(ins, dict) or not ins or not isinstance(outs, dict) or not outs
            or len(ins)>MAX_NODES or len(outs)>MAX_NODES
            or set(ins) & set(outs) or any(not identifier(k) for k in list(ins)+list(outs))
            or (set(ins)|set(outs)) & {'true','false','tap','pre','post','relu','clip','min','max'}
            or any(type(w) is not int or not 1 <= w <= MAX_VALUES for w in ins.values())):
        raise ValueError('distinct named inputs/outputs and positive vector widths required')
    if not isinstance(nodes, list) or len(nodes) > MAX_NODES:
        raise ValueError('DAG node limit exceeded')
    by_id = {}
    for n in nodes:
        if (not isinstance(n, dict) or set(n) != {'id','op','inputs','params','attrs'}
                or not identifier(n['id']) or n['id'] == 'output' or n['id'] in by_id
                or n['op'] not in OPS or not isinstance(n['inputs'],dict)
                or not isinstance(n['params'],dict) or not isinstance(n['attrs'],dict)):
            raise ValueError('invalid/duplicate DAG node or unsupported operator')
        if any(not identifier(k) for k in list(n['inputs'])+list(n['params'])):
            raise ValueError('invalid node port name')
        by_id[n['id']] = n
    def dependency(source):
        if not isinstance(source,str) or ':' not in source:
            raise ValueError('signal must be input:name or node:id')
        kind, key = source.split(':',1)
        if kind == 'input' and key in ins: return None
        if kind == 'node' and key in by_id: return key
        raise ValueError(f'dangling/unknown signal {source}')
    dependents = {k:set() for k in by_id}; incoming = {}; count = len(outs)
    for key, n in by_id.items():
        deps = {d for s in n['inputs'].values() if (d := dependency(s)) is not None}
        incoming[key] = len(deps); count += len(n['inputs'])
        for d in deps: dependents[d].add(key)
        for tensor in n['params'].values():
            if not isinstance(tensor,str) or tensor not in tensors:
                raise ValueError('node parameter refers to missing storage')
    if count > 8*MAX_NODES:
        raise ValueError('DAG edge limit exceeded')
    for source in outs.values(): dependency(source)
    ready = [k for k,v in incoming.items() if v==0]; heapq.heapify(ready); order=[]
    while ready:
        key = heapq.heappop(ready); order.append(key)
        for child in sorted(dependents[key]):
            incoming[child] -= 1
            if incoming[child] == 0: heapq.heappush(ready,child)
    if len(order) != len(nodes): raise ValueError('DAG cycle detected')
    widths = {f'input:{k}':v for k,v in ins.items()}; depths={k:0 for k in widths}
    values = sum(ins.values())
    if values>MAX_VALUES:raise ValueError('DAG input storage limit exceeded')
    for key in order:
        n=by_id[key]; op=n['op']; ports=n['inputs']; params=n['params']; attrs=n['attrs']
        expected_inputs = {'a','b'} if op in ('add','concat') else {'x'}
        expected_params = {'weight','bias'} if op=='affine' else {'gain'} if op=='scale' else set()
        expected_attrs = {'layout'} if op=='affine' else {'start','stop','step'} if op=='slice' else set()
        if set(ports)!=expected_inputs or set(params)!=expected_params or set(attrs)!=expected_attrs:
            raise ValueError(f'{key}: operator port/parameter/attribute schema differs')
        shape = {p:widths[s] for p,s in ports.items()}
        if op=='affine':
            w=tensors[params['weight']]['shape']; b=tensors[params['bias']]['shape']
            layout=attrs['layout']
            if layout not in ('in_out','out_in') or len(w)!=2 or len(b)!=1:
                raise ValueError('affine parameter rank/layout differs')
            input_width, width = w if layout=='in_out' else w[::-1]
            if input_width!=shape['x'] or b[0]!=width:
                raise ValueError(f'{key}: affine width mismatch')
        elif op=='add':
            if shape['a']!=shape['b']: raise ValueError(f'{key}: add widths differ; no implicit broadcasting')
            width=shape['a']
        elif op=='concat': width=shape['a']+shape['b']
        elif op=='slice':
            if any(type(v) is not int for v in attrs.values()) or attrs['step']==0:
                raise ValueError('slice needs integer bounds and nonzero stride')
            selected=range(shape['x'])[slice(attrs['start'],attrs['stop'],attrs['step'])]
            width=len(selected)
            if not width: raise ValueError('empty graph slice unsupported')
        else:
            width=shape['x']
            if op=='scale' and tensors[params['gain']]['shape'] != [1]:
                raise ValueError('scale gain must have shape [1]')
        values+=width
        if values>MAX_VALUES: raise ValueError('DAG activation storage limit exceeded')
        widths['node:'+key]=width
        depths['node:'+key]=1+max(depths[s] for s in ports.values())
    uses=parameter_uses(graph)
    return dict(order=order,widths=widths,output_widths={k:widths[v] for k,v in outs.items()},
                nodes=len(nodes),edges=count,depth=max(depths.values()),live_values=values,
                parameters=sum(len(t['bits']) for t in tensors.values()),
                parameter_uses=uses,shared_parameters={k:v for k,v in uses.items() if len(v)>1},
                unreferenced_tensors=sorted(set(tensors)-set(uses)),
                formal_supported=all(n['op']!='tanh' for n in nodes))


@dataclass(frozen=True)
class GraphBudget:
    max_nodes: int = 256
    max_edges: int = 1024
    max_parameters: int = 1_000_000
    max_depth: int = 128
    max_live_values: int = 1_000_000
    allowed_ops: tuple = OPS

    def __post_init__(self):
        for key,value in asdict(self).items():
            if key!='allowed_ops' and (type(value) is not int or not 1<=value<=MAX_VALUES):
                raise ValueError('positive bounded graph budget required')
        if not isinstance(self.allowed_ops,(list,tuple)) or len(set(self.allowed_ops))!=len(self.allowed_ops) or not set(self.allowed_ops)<=set(OPS):
            raise ValueError('unsupported graph operator budget')
        object.__setattr__(self,'allowed_ops',tuple(self.allowed_ops))

    def check(self, artifact, parent=None):
        d=artifact.to_dict(); stats=validate_graph(d['graph'],d['tensors'])
        for key in ('nodes','edges','parameters','depth','live_values'):
            if stats[key]>getattr(self,'max_'+key): raise ValueError(f'graph budget exceeded: {key}')
        if not {n['op'] for n in d['graph']['nodes']}<=set(self.allowed_ops):
            raise ValueError('operator forbidden by host graph budget')
        if parent is not None:
            p=parent.to_dict(); before=validate_graph(p['graph'],p['tensors'])
            if d['graph']['inputs']!=p['graph']['inputs'] or stats['output_widths']!=before['output_widths']:
                raise ValueError('public input/output interface changed')
        return stats


def to_dag(artifact):
    """Explicit conversion; identical tensor records, separate graph identity."""
    d=artifact.to_dict();g=d['graph']
    if g['kind']=='dag': return artifact
    if g['kind']!='dense': raise ValueError('graphify needs a supported dense artifact')
    nodes=[]; previous='input:x'
    for i,act in enumerate(g['activations'],1):
        affine=f'a{i}'; result='y' if i==len(g['activations']) else f'h{i}'
        nodes.append(node(affine,'affine',{'x':previous},params={'weight':f'W{i}','bias':f'b{i}'},attrs={'layout':g['layout']}))
        nodes.append(node(result,'identity' if act=='linear' else act,{'x':'node:'+affine}))
        previous='node:'+result
    d['graph']=dict(kind='dag',inputs={'x':g['widths'][0]},nodes=nodes,outputs={'y':previous})
    return TensorArtifact(d)


def trace_graph(artifact, inputs, semantics):
    if semantics not in ('real_graph','numpy_f32'): raise ValueError('unknown execution semantics')
    d=artifact.to_dict();g=d['graph'];meta=validate_graph(g,d['tensors'])
    if semantics=='real_graph' and not meta['formal_supported']: raise ValueError('UNSUPPORTED: rational tanh execution')
    if not isinstance(inputs,dict) or set(inputs)!=set(g['inputs']): raise ValueError('named graph input record required')
    values={};taps={};arrays=artifact.arrays(bits=semantics=='real_graph')
    for name,width in g['inputs'].items():
        if not isinstance(inputs[name],(list,tuple,np.ndarray)) or len(inputs[name])!=width: raise ValueError('graph input width differs')
        a=[Fraction(x) for x in inputs[name]] if semantics=='real_graph' else np.array(inputs[name],dtype=np.float32)
        if semantics=='numpy_f32' and (a.ndim!=1 or not np.isfinite(a).all()): raise ValueError('finite vector inputs required')
        values['input:'+name]=a;taps[name]=list(a)
    by_id={n['id']:n for n in g['nodes']}
    for key in meta['order']:
        n=by_id[key];op=n['op'];v={p:values[s] for p,s in n['inputs'].items()}
        with np.errstate(over='raise',invalid='raise'):
            if op=='affine':
                w=arrays[n['params']['weight']];b=arrays[n['params']['bias']]
                if n['attrs']['layout']=='in_out':w=w.T
                out=([sum((fraction(c)*x for c,x in zip(row,v['x'])),fraction(bias)) for row,bias in zip(w,b)]
                     if semantics=='real_graph' else w@v['x']+b)
            elif op=='scale':
                gain=arrays[n['params']['gain']][0]
                out=[fraction(gain)*x for x in v['x']] if semantics=='real_graph' else v['x']*gain
            elif op=='add':out=[a+b for a,b in zip(v['a'],v['b'])] if semantics=='real_graph' else v['a']+v['b']
            elif op=='concat':out=list(v['a'])+list(v['b']) if semantics=='real_graph' else np.concatenate((v['a'],v['b']))
            elif op=='slice':out=v['x'][slice(n['attrs']['start'],n['attrs']['stop'],n['attrs']['step'])]
            elif op=='relu':out=[max(Fraction(0),x) for x in v['x']] if semantics=='real_graph' else np.maximum(v['x'],np.float32(0))
            elif op=='tanh':out=np.tanh(v['x'])
            else:out=v['x'].copy()
        values['node:'+key]=out;taps['node:'+key]=list(out)
    taps.update({name:list(values[s]) for name,s in g['outputs'].items()})
    return taps


def region(artifact, selected):
    """Exact graph cut, including public outputs and shared parameter references."""
    d=artifact.to_dict();g=d['graph'];meta=validate_graph(g,d['tensors'])
    if not isinstance(selected,(list,tuple)) or not selected or len(set(selected))!=len(selected): raise ValueError('nonempty unique region nodes required')
    nodes={n['id']:n for n in g['nodes']};chosen=set(selected)
    if not chosen<=set(nodes):raise ValueError('region refers to missing nodes')
    incoming={};outgoing={}
    for address,source in edges(g).items():
        dest=address.split('.')[0];inside=source.startswith('node:') and source[5:] in chosen
        if dest in chosen and not inside:incoming[address]=source
        if dest not in chosen and inside:outgoing[address]=source
    used={t for key in chosen for t in nodes[key]['params'].values()}
    shared={t:meta['parameter_uses'][t] for t in sorted(used) if any(p.split('.')[0] not in chosen for p in meta['parameter_uses'][t])}
    return dict(nodes=sorted(chosen),inputs=dict(sorted(incoming.items())),outputs=dict(sorted(outgoing.items())),
                shared_parameters=shared,parameters=sorted(used))


def declared_region(artifact, selected, inputs, outputs, shared_parameters):
    cut=region(artifact,selected)
    for label,mapping in (('inputs',inputs),('outputs',outputs)):
        if (not isinstance(mapping,dict) or any(not identifier(k) or not isinstance(v,str) for k,v in mapping.items())
                or len(set(mapping.values()))!=len(mapping) or set(mapping.values())!=set(cut[label])):
            raise ValueError(f'incomplete or duplicate region {label}; actual boundary: {sorted(cut[label])}')
    if not isinstance(shared_parameters,list) or len(shared_parameters)!=len(set(shared_parameters)) or set(shared_parameters)!=set(cut['shared_parameters']):
        raise ValueError(f'shared parameter boundary differs: {sorted(cut["shared_parameters"])}')
    return dict(cut=cut,inputs=dict(inputs),outputs=dict(outputs))


def _set_edge(graph, address, source):
    if address not in edges(graph):raise ValueError(f'unknown edge {address}')
    owner,port=address.split('.')
    if owner=='output':graph['outputs'][port]=source
    else:next(n for n in graph['nodes'] if n['id']==owner)['inputs'][port]=source


def _import(d, patch, prefix, bindings, share):
    if not identifier(prefix):raise ValueError('identifier namespace required')
    p=patch.to_dict();pg=p['graph']
    if pg['kind']!='dag':raise ValueError('replacement/transplant must be explicitly graphified')
    validate_graph(pg,p['tensors'])
    if not isinstance(bindings,dict) or set(bindings)!=set(pg['inputs']):raise ValueError('patch input bindings differ')
    parent_meta=validate_graph(d['graph'],d['tensors'])
    for port,source in bindings.items():
        if not isinstance(source,str) or source not in parent_meta['widths'] or parent_meta['widths'][source]!=pg['inputs'][port]:
            raise ValueError('patch input signal/declared width differs')
    if not isinstance(share,dict) or not set(share)<=set(p['tensors']):raise ValueError('unknown shared patch tensor')
    node_map={n['id']:prefix+'__'+n['id'] for n in pg['nodes']}
    if set(node_map.values()) & {n['id'] for n in d['graph']['nodes']}:raise ValueError('transplant node namespace collision')
    tensor_map={};copied=[]
    for key,record in p['tensors'].items():
        name=share.get(key,prefix+'__'+key)
        if key in share:
            if name not in d['tensors'] or record!=d['tensors'][name]:raise ValueError('sharing needs identical existing tensor storage')
        else:
            if name in d['tensors']:raise ValueError('transplant tensor namespace collision')
            d['tensors'][name]=copy.deepcopy(record);copied.append(name)
        tensor_map[key]=name
    def mapped(s):return bindings[s[6:]] if s.startswith('input:') else 'node:'+node_map[s[5:]]
    for n in pg['nodes']:
        d['graph']['nodes'].append(node(node_map[n['id']],n['op'],{k:mapped(s) for k,s in n['inputs'].items()},
                                       params={k:tensor_map[t] for k,t in n['params'].items()},attrs=n['attrs']))
    return dict(outputs={k:mapped(s) for k,s in pg['outputs'].items()},nodes=node_map,tensors=tensor_map,copied_tensors=copied,shared_tensors=dict(share))


def _finish(parent, document, budget, report):
    candidate=TensorArtifact(document,max_elements=budget.max_parameters)
    stats=budget.check(candidate,parent)
    report.update(parent_id=parent.sha256,candidate_id=candidate.sha256,resources=stats,
                  evidence_reuse='requires_recheck',behavior_preservation='not_established')
    return candidate,report


def transplant(parent, patch, *, prefix, inputs, budget, share=None):
    d=parent.to_dict();budget.check(parent)
    imported=_import(d,patch,prefix,inputs,share or {})
    return _finish(parent,d,budget,dict(operation='transplant',imported=imported))


def replace_region(parent, declaration, patch, *, prefix, inputs, outputs, budget, drop_inputs=(), share=None):
    d=parent.to_dict();budget.check(parent)
    actual=region(parent,declaration['cut']['nodes'])
    if actual!=declaration['cut']:raise ValueError('stale region boundary or parameter sharing')
    incoming,outgoing=declaration['inputs'],declaration['outputs']
    declared_region(parent,actual['nodes'],incoming,outgoing,list(actual['shared_parameters']))
    if (not set(inputs.values())<=set(incoming) or len(set(drop_inputs))!=len(drop_inputs)
            or set(inputs.values()) & set(drop_inputs) or set(inputs.values())|set(drop_inputs)!=set(incoming)):
        raise ValueError('every region input must be bound or explicitly dropped')
    if set(outputs)!=set(outgoing):raise ValueError('every outgoing boundary edge must be mapped')
    binding={key:actual['inputs'][incoming[alias]] for key,alias in inputs.items()}
    imported=_import(d,patch,prefix,binding,share or {})
    if not set(outputs.values())<=set(imported['outputs']):raise ValueError('unknown patch output')
    for alias,port in outputs.items():_set_edge(d['graph'],outgoing[alias],imported['outputs'][port])
    chosen=set(actual['nodes']);d['graph']['nodes']=[n for n in d['graph']['nodes'] if n['id'] not in chosen]
    remaining=parameter_uses(d['graph']);removed=[]
    for name in actual['parameters']:
        if name not in remaining:
            del d['tensors'][name];removed.append(name)
    return _finish(parent,d,budget,dict(operation='replace',boundary=actual,imported=imported,
                                      removed_tensors=removed,dropped_inputs=list(drop_inputs)))


def redirect(parent, address, source, *, budget):
    budget.check(parent);d=parent.to_dict();old=edges(d['graph']).get(address)
    _set_edge(d['graph'],address,source)
    return _finish(parent,d,budget,dict(operation='redirect',edge=address,before=old,after=source))


def untie(parent, address, new_name, *, budget):
    budget.check(parent);d=parent.to_dict()
    if not identifier(new_name) or new_name in d['tensors']:raise ValueError('fresh tensor name required')
    if not isinstance(address,str) or address.count('.')!=1:raise ValueError('node.parameter address required')
    owner,port=address.split('.')
    nodes={n['id']:n for n in d['graph']['nodes']}
    if owner not in nodes or port not in nodes[owner]['params']:raise ValueError('unknown parameter reference')
    original=nodes[owner]['params'][port]
    d['tensors'][new_name]=copy.deepcopy(d['tensors'][original]);nodes[owner]['params'][port]=new_name
    return _finish(parent,d,budget,dict(operation='untie',reference=address,original=original,copy=new_name,remaining_users=parameter_uses(d['graph']).get(original,[])))


def add_residual(parent, patch, *, prefix, inputs, output, into, scale, budget, share=None):
    d=parent.to_dict();budget.check(parent)
    if into not in edges(d['graph']):raise ValueError('residual target edge missing')
    old=edges(d['graph'])[into]
    imported=_import(d,patch,prefix,inputs,share or {})
    if output not in imported['outputs']:raise ValueError('residual output missing')
    def fresh(base,used):
        name=base;index=0
        while name in used:index+=1;name=f'{base}_{index}'
        used.add(name);return name
    node_names={n['id'] for n in d['graph']['nodes']}
    gain=fresh(prefix+'__gain',set(d['tensors']))
    scaled=fresh(prefix+'__scaled',node_names);summed=fresh(prefix+'__sum',node_names)
    bits=round_f32(scale)
    d['tensors'][gain]=dict(shape=[1],dtype='f32',bits=[f'{bits:08x}'])
    d['graph']['nodes'].extend([node(scaled,'scale',{'x':imported['outputs'][output]},params={'gain':gain}),
                               node(summed,'add',{'a':old,'b':'node:'+scaled})])
    _set_edge(d['graph'],into,'node:'+summed)
    return _finish(parent,d,budget,dict(operation='add_residual',into=into,original_signal=old,imported=imported,
                                      scale_requested=str(scale),scale_stored=str(fraction(bits)),scale_bits=f'{bits:08x}',
                                      auxiliary_nodes=[scaled,summed],gain_tensor=gain,
                                      reduction_order='original + (branch * stored_gain)',augmentation=True))


def graph_diff(parent,candidate,*,offset=0,limit=64):
    a,b=parent.to_dict(),candidate.to_dict();changes=[];count=0;by_tensor={}
    for name in sorted(set(a['tensors'])|set(b['tensors'])):
        old=a['tensors'].get(name);new=b['tensors'].get(name)
        before=old['bits'] if old else [];after=new['bits'] if new else []
        local=0
        for i in range(max(len(before),len(after))):
            x=before[i] if i<len(before) else None;y=after[i] if i<len(after) else None
            if x==y:continue
            if offset<=count<offset+limit:changes.append(dict(tensor=name,flat_index=i,before_bits=x,after_bits=y))
            count+=1;local+=1
        if local:by_tensor[name]=local
    def nodes(d):return {n['id']:n for n in d['graph'].get('nodes',[])}
    an,bn=nodes(a),nodes(b)
    delta=dict(changed=a['graph']!=b['graph'],added_nodes=sorted(set(bn)-set(an)),removed_nodes=sorted(set(an)-set(bn)),
               modified_nodes=sorted(k for k in set(an)&set(bn) if an[k]!=bn[k]),
               added_tensors=sorted(set(b['tensors'])-set(a['tensors'])),removed_tensors=sorted(set(a['tensors'])-set(b['tensors'])))
    delta['tensor_metadata_changes']={k:dict(before={f:a['tensors'][k][f] for f in ('shape','dtype')},after={f:b['tensors'][k][f] for f in ('shape','dtype')})
        for k in sorted(set(a['tensors'])&set(b['tensors'])) if any(a['tensors'][k][f]!=b['tensors'][k][f] for f in ('shape','dtype'))}
    if a['graph']['kind']==b['graph']['kind']=='dag':
        ae,be=edges(a['graph']),edges(b['graph'])
        delta['edges']={k:dict(before=ae.get(k),after=be.get(k)) for k in sorted(set(ae)|set(be)) if ae.get(k)!=be.get(k)}
        affected=set(delta['added_nodes']+delta['removed_nodes']+delta['modified_nodes'])
        for graph in (a['graph'],b['graph']):
            for tensor,users in parameter_uses(graph).items():
                if a['tensors'].get(tensor)!=b['tensors'].get(tensor):affected.update(p.split('.')[0] for p in users)
        consumers={}
        for graph in (a['graph'],b['graph']):
            for n in graph['nodes']:
                for source in n['inputs'].values():
                    if source.startswith('node:'):consumers.setdefault(source[5:],set()).add(n['id'])
        pending=list(affected)
        while pending:
            for child in consumers.get(pending.pop(),()):
                if child not in affected:affected.add(child);pending.append(child)
        delta['potentially_affected_nodes']=sorted(affected)
        delta['unaffected_dependency_nodes']=sorted((set(an)&set(bn))-affected)
        delta['potentially_affected_outputs']=sorted(k for k in set(a['graph']['outputs'])|set(b['graph']['outputs'])
            if a['graph']['outputs'].get(k)!=b['graph']['outputs'].get(k)
            or any(s.startswith('node:') and s[5:] in affected for s in (a['graph']['outputs'].get(k,''),b['graph']['outputs'].get(k,''))))
    return dict(parent_id=parent.sha256,candidate_id=candidate.sha256,status='NO_CHANGE' if parent.sha256==candidate.sha256 else 'CHANGED',
                changed_elements=count,changed_by_tensor=by_tensor,offset=offset,changes=changes,graph=delta,
                behavior_preservation='not_established',evidence_reuse='requires_dependency_recheck')
