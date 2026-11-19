"""Bounded public adaptation interface; hidden mechanisms stay in the host runner."""
import copy
import json
import math
from fractions import Fraction

from .closed_loop_control import Episode, make_task
from .control_dsl import Workcell
from .controller_author import construct, predict, teacher as block_source
from .wspl_artifact import ReLUArtifact


class MappedWorld(Workcell):
    def apply(self, command):
        vx,vy=command['vx'],command['vy']
        a,b=self.mapping
        super().apply(dict(command,vx=a[0]*vx+a[1]*vy,vy=b[0]*vx+b[1]*vy))


def episode(task, blocks, specs, mapping):
    e=Episode(task,blocks,specs)
    e.world=MappedWorld(task['initial'],task['objects'],radius=specs['near']['radius'],bound=specs['safe']['bound'])
    e.world.mapping=mapping
    return e


def probe(commands,mapping):
    if not isinstance(commands,list) or not 1<=len(commands)<=8:raise ValueError('1..8 commands required')
    rows=[]
    for c in commands:
        if not isinstance(c,list) or len(c)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>.07 for v in c):
            raise ValueError('Probe command must be two finite numbers in [-.07,.07]')
        w=MappedWorld([0.,0.],{},radius=.037,bound=1.024);w.mapping=mapping
        w.apply(dict(vx=c[0],vy=c[1],grip=0.))
        rows.append(dict(command=c,observed_displacement=w.position))
    return rows


def candidate(action, original, spec):
    if action['tool']=='install_wspl':
        artifact=construct(action['source'],spec)
    else:
        d=original.to_dict();d['layers']=action['layers'];d['allocation']['mode']='imported';d['source_map']={}
        import hashlib
        d['source_sha256']=hashlib.sha256(json.dumps(action['layers'],sort_keys=True).encode()).hexdigest()
        d['compiler']='astra_adaptation.native_layers.v1';artifact=ReLUArtifact(d)
    if artifact.input_names!=original.input_names or artifact.output_names!=original.output_names:
        raise ValueError('Port names must be preserved')
    if any(artifact.to_dict()[k]!=original.to_dict()[k] for k in ('inputs','outputs')):
        raise ValueError('Port domains and units must be preserved')
    if artifact.to_dict()['allocation']['parameters']>27:raise ValueError('27-parameter budget exceeded')
    final=artifact.to_dict()['layers'][-1]
    if any(Fraction(v)!=0 for v in final['weights'][2]) or Fraction(final['bias'][2])!=0:
        raise ValueError('Gripper output must have an identically zero final row')
    # Engineering admission only, not a universal functional/safety certificate.
    for x,y in [(0,0),(-2,-2),(2,2),(-2,2),(2,-2),(.01,-.01)]:
        v=predict(artifact,dict(x=x,y=y))
        if any(not math.isfinite(z) for z in v.values()) or abs(v['grip'])>1e-7:
            raise ValueError('Nonfinite output or movement block changes gripper')
    return artifact


def parse_action(text):
    if len(text.encode())>32000:raise ValueError('Response exceeds 32KB')
    a=json.loads(text)
    if not isinstance(a,dict) or not isinstance(a.get('prediction'),str) or not a['prediction'].strip():
        raise ValueError('JSON object with prospective prediction required')
    schemas={
        'probe':{'commands'},'install_wspl':{'source'},'install_native':{'layers'},
        'author':{'request'},'fit':{'rows','updates'},'evaluate':set(),'finish':set()}
    if a.get('tool') not in schemas or set(a)!={'tool','prediction'}|schemas[a['tool']]:raise ValueError('Unknown tool or fields')
    if a['tool']=='fit':
        if type(a['updates']) is not int or not 1<=a['updates']<=4096:raise ValueError('Invalid updates')
        if not isinstance(a['rows'],list) or not 1<=len(a['rows'])<=128:raise ValueError('1..128 rows required')
        for row in a['rows']:
            if set(row)!={'inputs','outputs'} or set(row['inputs'])!={'x','y'} or set(row['outputs'])!={'vx','vy','grip'}:raise ValueError('Invalid fitting row')
            for k,v in {**row['inputs'],**row['outputs']}.items():
                if type(v) not in (int,float) or not math.isfinite(v) or abs(v)>(2 if k in ('x','y') else .07):raise ValueError('Fitting data bound')
            if row['outputs']['grip']!=0:raise ValueError('Movement-only fitting')
    return a


def tasks(start,counts):
    return [make_task(start+4*i,n,'none','objects_'+str(n),renamed=True) for i,n in enumerate(counts)]


def packet(original,specs,history,remaining):
    return ('You are Astra, engineering a reusable neural controller repair. Return exactly one JSON object, no fences. '
      'A frozen small transformer chooses actions through six neural blocks. The movement actuator has changed; '
      'identify its response from observations. Only the movement block may change in this first pilot. '
      'The executive and other five blocks are frozen. You will be absent during final deployment. '
      'Observation coordinates and action ports remain unchanged. No mechanism equation or solution is supplied. '
      'The actuator is deterministic in this pilot. You may query commands from an isolated origin, inspect supplied '
      'artifacts, ask the saved small WSPL author for source, fit the current block to your own labels, install source '
      'or literal weights, and evaluate on calibration missions. No external tool calls or filesystem access. '
      'This is a bounded tool protocol: the host executes the requested operation and returns its result next turn. '
      'Every request includes a prediction written BEFORE its outcome is measured. Invalid requests consume a turn. '
      'Last accepted block is retained, not validation-best. finish freezes it; exhaustion also freezes it. '
      'Original and repaired blocks will also be tested under original dynamics, so report expected interference. '
      'Tools: {"tool":"probe","prediction":"...","commands":[[vx,vy],...]}, at most8 per request; '
      '{"tool":"install_wspl","prediction":"...","source":"module ..."}; '
      '{"tool":"install_native","prediction":"...","layers":[...]}, same layer schema and shape as original, '
      'rational strings for coefficients; {"tool":"author","prediction":"...","request":"functional specification"}, '
      'saved Qwen author returns WSPL and the host attempts installation; '
      '{"tool":"fit","prediction":"...","updates":128,"rows":[{"inputs":{"x":0,"y":0},'
      '"outputs":{"vx":0,"vy":0,"grip":0}}]}, warm-start AdamW LR.003 decay0 batch32, '
      'same architecture; {"tool":"evaluate","prediction":"..."}; {"tool":"finish","prediction":"..."}. '
      'WSPL module ports must match original, at most27compiled parameters. Execution uses named geometric errors '
      'x=target_x-current_x,y=target_y-current_y. Commands must keep grip zero. '
      'Public original specification, artifact, budgets and COMPLETE visible history follow:\n'+
      json.dumps(dict(original_spec=specs['move'],original_source=block_source(specs['move']),original_weights=original.to_dict(),remaining=remaining,history=history),separators=(',',':')))
