"""Typed neural blocks and a planar workcell for an LTLf author-model pilot."""
from fractions import Fraction
import hashlib
import json
import random
import re

from .automata import compile_dfa
from .ltlf import Expr, evaluate_closed_trace
from .planning import FiniteModel, synthesize
from .wspl_compile import compile_scalar
from .wspl_parse import parse_module

KINDS = ('move', 'near', 'holding', 'close', 'open', 'safe')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def identifier(spec):
    return hashlib.sha256(canonical(spec).encode()).hexdigest()


def make_spec(kind, index):
    if kind == 'move':
        return dict(kind=kind, gain=round(.2 + (index % 61) * .01, 3),
                    speed=round(.04 + (index // 61) * .01, 3))
    if kind == 'near':
        return dict(kind=kind, radius=round(.02 + (index % 161) * .001, 3),
                    metric='linf' if index < 161 else 'l1')
    if kind == 'holding':
        return dict(kind=kind, threshold=round(.25 + index * .001, 3))
    if kind in ('close', 'open'):
        return dict(kind=kind, amplitude=round(.501 + index * .001, 3))
    if kind == 'safe':
        return dict(kind=kind, bound=round(.8 + index * .001, 3))
    raise ValueError('Unknown block kind')


def ports(spec):
    kind = spec['kind']
    if kind == 'holding':
        return ('h',), ('score',)
    if kind in ('near', 'safe'):
        return ('x', 'y'), ('score',)
    return ('x', 'y'), ('vx', 'vy', 'grip')


def teacher(spec):
    inputs, outputs = ports(spec)
    lines = ['module Block {']
    for name in inputs:
        bounds = '[0,1]' if name == 'h' else '[-2,2]'
        lines.append(f'  input {name}: Real{bounds} normalized_{name};')
    for name in outputs:
        lines.append(f'  output {name}: Real[-8,8] normalized_{name};')
    kind = spec['kind']
    if kind == 'move':
        gain, speed = spec['gain'], spec['speed']
        lines.extend([f'  vx = clip({gain} * x, -{speed}, {speed});',
                      f'  vy = clip({gain} * y, -{speed}, {speed});', '  grip = 0;'])
    elif kind in ('close', 'open'):
        amplitude = spec['amplitude'] * (1 if kind == 'close' else -1)
        lines.extend(['  vx = 0;', '  vy = 0;', f'  grip = {amplitude};'])
    elif kind == 'holding':
        lines.append(f"  score = h - {spec['threshold']};")
    else:
        lines.extend(['  let ax: Real[-2,2] normalized_ax = max(x, -x);', '  let ay: Real[-2,2] normalized_ay = max(y, -y);'])
        distance = 'ax + ay' if spec.get('metric') == 'l1' else 'max(ax, ay)'
        bound = spec['radius'] if kind == 'near' else spec['bound']
        lines.append(f'  score = {bound} - ({distance});')
    return '\n'.join(lines + ['}']) + '\n'


def prompt(spec):
    inputs, outputs = ports(spec)
    return (
        'Write one WSPL Core module named Block for the specified neural controller block. '
        'Return only the module source. The host compiles it to ReLU network weights. '
        'Use input/output declarations Real[lower,upper] normalized_name; assignments end '
        'with semicolons. Supported operations are affine arithmetic, relu(a), min(a,b), '
        'max(a,b), and clip(a,lo,hi). Unary minus is allowed. Let bindings use let name: Real[lo,hi] normalized_name = expression;. '
        'All outputs must be declared Real[-8,8]. Inputs x,y are Real[-2,2]; h is Real[0,1]. '
        'Preserve the listed port order.\n'
        f'Inputs: {list(inputs)}. Outputs: {list(outputs)}.\n'
        'For move, x,y are goal-minus-position errors; emit gain times each error clipped '
        'to +/-speed and grip zero. For near, return radius minus the specified L1 or '
        'L-infinity norm of x,y. For holding, return h minus threshold. For close/open, '
        'emit zero velocities and positive/negative amplitude respectively. For safe, '
        'return bound minus the L-infinity norm of world position x,y. '
        'Predicate truth is score >= 0.\nSpecification: ' + canonical(spec)
    )


def oracle(spec, values):
    kind = spec['kind']
    if kind == 'holding':
        return {'score': float(values['h']) - spec['threshold']}
    x, y = float(values['x']), float(values['y'])
    if kind == 'move':
        cap = spec['speed']
        return dict(vx=max(-cap, min(cap, spec['gain'] * x)),
                    vy=max(-cap, min(cap, spec['gain'] * y)), grip=0.)
    if kind in ('close', 'open'):
        return dict(vx=0., vy=0., grip=spec['amplitude'] * (1 if kind == 'close' else -1))
    distance = abs(x) + abs(y) if spec.get('metric') == 'l1' else max(abs(x), abs(y))
    return {'score': spec.get('radius', spec.get('bound')) - distance}


def construct(source, spec):
    if len(source.encode()) > 16000:
        raise ValueError('Source limit exceeded')
    source = source.strip()
    fenced = re.fullmatch(r'```(?:wspl|text)?\s*\n([\s\S]*?)\n```', source)
    if fenced:
        source = fenced.group(1)
    parsed = parse_module(source, max_nodes=256)
    artifact = compile_scalar(parsed.program, max_parameters=8192, max_layers=16,
                              max_units=256, max_nodes=256)
    expected_inputs, expected_outputs = ports(spec)
    if artifact.input_names != expected_inputs or artifact.output_names != expected_outputs:
        raise ValueError('Typed block ports differ')
    return artifact


def predict(artifact, values):
    return {name: float(value) for name, value in artifact.evaluate(
        {key: Fraction(str(value)) for key, value in values.items()}).items()}


def probes(spec, seed=90, count=24):
    rng = random.Random(seed)
    if spec['kind'] == 'holding':
        values = [0., 1., spec['threshold'], spec['threshold'] - .001,
                  spec['threshold'] + .001]
        return [dict(h=value) for value in values]
    rows = [dict(x=x, y=y) for x, y in [(0, 0), (2, 2), (-2, -2), (2, -2), (-2, 2)]]
    rows.extend(dict(x=rng.uniform(-2, 2), y=rng.uniform(-2, 2)) for _ in range(count))
    boundary = spec.get('radius', spec.get('bound', 0.1))
    rows.extend(dict(x=boundary + shift, y=0.) for shift in [-.001, 0., .001])
    return rows


def check_block(artifact, spec):
    errors = [abs(predict(artifact, values)[name] - expected)
              for values in probes(spec) for name, expected in oracle(spec, values).items()]
    return dict(max_error=max(errors), passes=max(errors) <= 1e-6)


def dataset():
    """Disjoint full specifications; shared functional families are intentional."""
    splits = {name: [] for name in ('train', 'development', 'test')}
    for kind in KINDS:
        indices = list(range(320))
        random.Random('ltlf-controller-' + kind).shuffle(indices)
        for position, index in enumerate(indices[:288]):
            split = 'train' if position < 256 else 'development' if position < 272 else 'test'
            spec = make_spec(kind, index)
            splits[split].append(dict(id=identifier(spec), spec=spec,
                messages=[dict(role='user', content=prompt(spec)),
                          dict(role='assistant', content=teacher(spec))]))
    identities = [{row['id'] for row in rows} for rows in splits.values()]
    assert all(len(ids) == len(rows) for ids, rows in zip(identities, splits.values()))
    assert not (identities[0] & identities[1] or identities[0] & identities[2] or identities[1] & identities[2])
    return splits


def atom(name):
    return Expr('ap', name=name)


def mission(checkpoint):
    order = ['holding', 'visited', 'delivered'] if checkpoint else ['holding', 'delivered']
    eventual = atom(order[-1])
    for name in reversed(order[:-1]):
        eventual = Expr('and', (atom(name), Expr('X', (Expr('F', (eventual,)),))))
    return Expr('and', (Expr('G', (atom('safe'),)), Expr('F', (eventual,))))


def strategy(checkpoint):
    states = {name: dict(holding=name in ('held', 'visited', 'goal'),
                        visited=name == 'visited', delivered=name == 'done', safe=True)
              for name in ('start', 'near', 'held', 'visited', 'goal', 'done')}
    transitions = {'start': {'approach': ['near']}, 'near': {'grasp': ['held']},
                   'held': {'transport': ['goal'], 'waypoint': ['visited']},
                   'visited': {'transport': ['goal']}, 'goal': {'release': ['done']}}
    model = FiniteModel(states, transitions, {action: [] for action in
        ('approach', 'grasp', 'waypoint', 'transport', 'release')})
    return model, synthesize(compile_dfa(mission(checkpoint)), model, 'start')


def rollout(blocks, specs, checkpoint=False, initial=(-.7, -.6), target=(.6, .55)):
    """Execute compiled neural blocks; judge physical trace with independent geometry."""
    model, plan = strategy(checkpoint)
    position = list(initial)
    object_position = [-.25, .1]
    waypoint = [0., .65]
    held = False
    visited = False
    delivered = False
    world, automaton = plan.initial
    trace, actions = [], []
    radius = specs['near']['radius']
    bound = specs['safe']['bound']

    def observed():
        return dict(holding=held, visited=visited, delivered=delivered,
                    safe=max(abs(v) for v in position) <= bound)

    def neural(name, values):
        return predict(blocks[name], values)

    for _ in range(8):
        action = plan.policy.get((world, automaton), 'invalid')
        if action is None:
            trace.append(observed())
            return dict(success=evaluate_closed_trace(mission(checkpoint), trace),
                        trace=trace, actions=actions, terminal=list(position), status='stopped')
        if action == 'invalid':
            break
        symbolic = model.valuation(world)
        automaton = plan.dfa.step(automaton, symbolic)
        goal = object_position if action == 'approach' else waypoint if action == 'waypoint' else target
        completed = False
        for _ in range(160):
            trace.append(observed())
            if not trace[-1]['safe']:
                return dict(success=False, trace=trace, actions=actions, status='unsafe')
            if neural('safe', dict(x=position[0], y=position[1]))['score'] < 0:
                return dict(success=False, trace=trace, actions=actions, status='predicate_stop')
            dx, dy = goal[0] - position[0], goal[1] - position[1]
            if action in ('approach', 'waypoint', 'transport'):
                if neural('near', dict(x=dx, y=dy))['score'] >= 0:
                    completed = True
                    if action == 'waypoint':
                        visited = max(abs(dx), abs(dy)) <= radius + 1e-9
                    break
                command = neural('move', dict(x=dx, y=dy))
            else:
                command = neural('close' if action == 'grasp' else 'open', dict(x=dx, y=dy))
            if max(abs(command['vx']), abs(command['vy'])) > .2 or abs(command['grip']) > 1:
                return dict(success=False, trace=trace, actions=actions, status='actuator_violation')
            position[0] += command['vx']; position[1] += command['vy']
            if command['grip'] > .5 and max(abs(position[i] - object_position[i]) for i in (0, 1)) <= radius + 1e-9:
                held = True
            if held:
                object_position = list(position)
            if command['grip'] < -.5 and held:
                held = False
                delivered = max(abs(object_position[i] - target[i]) for i in (0, 1)) <= radius + 1e-9
            if action == 'grasp' and neural('holding', dict(h=int(held)))['score'] >= 0:
                completed = True
                break
            if action == 'release' and neural('holding', dict(h=int(held)))['score'] < 0:
                completed = True
                break
        actions.append(action)
        if not completed:
            break
        world = model.transitions[world][action][0]
    trace.append(observed())
    return dict(success=False, trace=trace, actions=actions, status='budget_or_contract_failure')
