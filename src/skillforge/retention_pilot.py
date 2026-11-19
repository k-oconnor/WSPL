"""Observation-only neural selection between immutable movement artifacts."""
import json
import math
from fractions import Fraction
from .wspl_parse import parse_module
from .wspl_compile import compile_scalar


DEFAULT_SOURCE='''module Select {
input u: Real[-1,1] scalar;
input v: Real[-1,1] scalar;
input dx: Real[-1,1] scalar;
input dy: Real[-1,1] scalar;
output original: Real[-8,8] scalar;
output repaired: Real[-8,8] scalar;
original = 1;
repaired = 0;
}'''


def compile_selector(source):
    if not isinstance(source,str) or len(source.encode())>16000:raise ValueError('Selector source limit')
    a=compile_scalar(parse_module(source,max_nodes=128).program,max_parameters=64,max_layers=4,max_units=16,max_nodes=128)
    if a.input_names!=('u','v','dx','dy') or a.output_names!=('original','repaired'):raise ValueError('Selector ports')
    d=a.to_dict()
    if any(Fraction(p['type']['lower'])>-1 or Fraction(p['type']['upper'])<1 for p in d['inputs']):raise ValueError('Selector input domain must cover [-1,1]')
    return a


def validate_probe(value):
    if not isinstance(value,list) or len(value)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>.01 for v in value):
        raise ValueError('Two finite diagnostic commands in [-.01,.01] required')
    return value


def diagnose(world,command,selector):
    """No mapping, regime label, scene identity, or prior selection enters the selector."""
    validate_probe(command);before=world.observe()
    world.apply(dict(vx=command[0],vy=command[1],grip=0.))
    after=world.observe()
    inputs=dict(u=command[0],v=command[1],dx=after['position'][0]-before['position'][0],dy=after['position'][1]-before['position'][1])
    scores=selector.evaluate({k:Fraction(str(v)) for k,v in inputs.items()})
    selected='repaired' if scores['repaired']>scores['original'] else 'original'
    return selected,dict(command=command,before=before,after=after,inputs=inputs,
                         scores={k:str(v) for k,v in scores.items()},selected=selected,physical_steps=1)


def parse_request(text):
    if len(text.encode())>24000:raise ValueError('Request size limit')
    a=json.loads(text);schemas={'install':{'command','source'},'evaluate':set(),'finish':set()}
    if not isinstance(a,dict) or a.get('tool') not in schemas or set(a)!={'tool','prediction'}|schemas[a['tool']]:raise ValueError('Request schema')
    if not isinstance(a['prediction'],str) or not a['prediction'].strip():raise ValueError('Prospective prediction required')
    if a['tool']=='install':validate_probe(a['command'])
    return a


def packet(original,repaired,history,remaining):
    return ('You are Astra. Build one deployable system that retains old-task performance AND handles changed '
      'actuator dynamics. The last pilot repaired changed dynamics but broke the original dynamics. That is NOT '
      'an acceptable final adaptation here. Both movement blocks and all three Qwen executives are frozen. '
      'You may program a small neural selector and choose one diagnostic motion command. Before EVERY mission '
      'the host executes that command physically, without resetting the resulting position, then feeds only '
      '(u,v,dx,dy) to your selector. u,v are your command; dx,dy are observed displacement. Select original or '
      'repaired by larger output score (ties choose original). No regime label, task identifier, previous selection '
      'or hidden dynamics enters this network. There is no frontier model at deployment. The same artifact is '
      'used through original/changed/original regime returns. Regime changes occur only at episode boundaries; '
      'there is no mid-mission switching in this pilot. Retention is measured against baseline-successful original '
      'missions. Diagnostic physical motion counts toward the episode and must not destroy retention. '
      'All calibration feedback includes both regimes, with scenario IDs but no runtime labels to your network. '
      'This is noiseless deterministic adaptation to two previously studied dynamics, not new mechanism discovery. '
      'Return exactly JSON, no external tools or code fences: '
      '{"tool":"install","prediction":"expected effect on BOTH regimes","command":[u,v],"source":"module Select {...}"}; '
      '{"tool":"evaluate","prediction":"..."}; or {"tool":"finish","prediction":"..."}. '
      'command components within [-.01,.01]. Source uses WSPL affine/ReLU/min/max/clip, four inputs named '
      'u,v,dx,dy declared Real[-1,1] scalar, outputs original,repaired Real[-8,8] scalar. At most64compiled parameters. '
      'Install is atomic; invalid requests consume a turn. Last accepted candidate freezes on finish or budget '
      'exhaustion, not validation-best. No gradient training. The default always selects original. '
      'Example syntax (not a solution): '+DEFAULT_SOURCE+'\n'+json.dumps(dict(original_weights=original.to_dict(),
       repaired_weights=repaired.to_dict(),remaining=remaining,history=history),separators=(',',':')))
