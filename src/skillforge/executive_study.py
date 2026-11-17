"""Task distribution and independent execution scoring for the neural DSL pilot."""
import hashlib
import json
import random
import re
from pathlib import Path

from .control_dsl import Executor, Workcell, compile_program
from .controller_author import KINDS
from .wspl_artifact import ReLUArtifact


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def order_goals(task):
    pending = set(task['goals'])
    result = []
    while pending:
        ready = sorted(n for n in pending if not any(b == n and a in pending for a,b in task['before']))
        if not ready:
            raise ValueError('Cyclic task')
        result.append(ready[0])
        pending.remove(ready[0])
    return result


def teacher(task):
    return '\n'.join('transfer ' + n + ' to ' + task['goals'][n] for n in order_goals(task)) + '\nmove_to home\n'


def prompt(task):
    visible = {k:v for k,v in task.items() if k not in ('jammed', 'panel')}
    return ('Write only a neural control DSL program, one instruction per line. '
        'The language is: move_to OBJECT_OR_POINT; grasp OBJECT; release OBJECT; '
        'transfer OBJECT to POINT. These are line forms, do not write semicolons. '
        'transfer expands into approach, grasp, transport and release. Each skill completes before the next. '
        'Move every named object to its assigned goal and leave it there, obey every before pair [A,B] '
        '(A must be delivered before B), and return the empty gripper to home. '
        'Among currently available goals choose the alphabetically first object. '
        'Do not move objects again after delivery. All positions are normalized planar coordinates. '
        'The frozen motor skills handle trajectories. No explanations or code fences.\nScene and task: ' + canonical(visible))


def make_task(seed, count, renamed=False, jam=False, panel='engineering'):
    rng = random.Random(seed)
    names = rng.sample(['amber','birch','coral','dune','elm','fern','garnet','hazel'] if not renamed else
                       ['widget_z','parcel_q','unit_k','item_v','sample_m','part_x','token_j','piece_r'], count)
    rng.shuffle(names)
    sources = [(-.55, -.55),(-.55,0),(-.55,.55),(-.25,-.3),(-.25,.3)]
    destinations = [(.3,-.55),(.3,0),(.3,.55),(.65,-.3),(.65,.3)]
    rng.shuffle(sources)
    rng.shuffle(destinations)
    jitter = lambda p: [round(v+rng.uniform(-.035,.035),4) for v in p]
    objects = {n:jitter(sources[i]) for i,n in enumerate(names)}
    points = {'home': [-.75,-.75]}
    goals = {}
    labels = rng.sample(range(10,90), count)
    for i,n in enumerate(names):
        target = 'site_' + str(labels[i])
        points[target] = jitter(destinations[i])
        goals[n] = target
    # A sampled hidden ordering produces a DAG; the input order reveals no solution.
    ordering = list(names)
    rng.shuffle(ordering)
    edges = [[a,b] for i,a in enumerate(ordering) for b in ordering[i+1:] if rng.random()<.55]
    rng.shuffle(edges)
    task = dict(objects=objects, points=points, goals=goals, before=edges,
                initial=[-.75,-.75], jammed=[], panel=panel)
    if jam:
        task['jammed'] = [order_goals(task)[-1]]
    return task


def record(task):
    return dict(id=hashlib.sha256(canonical(task).encode()).hexdigest(), task=task,
                messages=[dict(role='user',content=prompt(task)),dict(role='assistant',content=teacher(task))])


def make_expanded_task(seed, count, renamed=False, jam=False, panel='engineering'):
    if not 2 <= count <= 8:
        raise ValueError('Expanded workcell supports two to eight objects')
    rng = random.Random(seed)
    def name():
        if renamed:
            return 'Unit_' + ''.join(rng.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4)) + '_Part_' + str(rng.randrange(100,1000))
        return rng.choice(['obj_', 'item_', 'part_', 'sample_']) + ''.join(rng.choices('abcdefghijklmnopqrstuvwxyz',k=4)) + str(rng.randrange(10,100))
    names=[]
    while len(names)<count:
        candidate=name()
        if candidate not in names:names.append(candidate)
    sources=[(x,y) for x in [-.7,-.35] for y in [-.6,-.2,.2,.6]]
    destinations=[(x,y) for x in [.3,.65] for y in [-.6,-.2,.2,.6]]
    rng.shuffle(sources);rng.shuffle(destinations)
    jitter=lambda p:[round(v+rng.uniform(-.025,.025),4) for v in p]
    objects={n:jitter(sources[i]) for i,n in enumerate(names)}
    points={'home':[-.8,-.8]}
    goals={}
    for i,n in enumerate(names):
        target='site_'+str(rng.randrange(1000,10000))
        while target in points:target='site_'+str(rng.randrange(1000,10000))
        points[target]=jitter(destinations[i]);goals[n]=target
    order=list(names);rng.shuffle(order)
    topology=(seed//4)%5
    if topology==0:edges=[[order[i],order[i+1]] for i in range(count-1)]
    elif topology==1:edges=[]
    elif topology==2:edges=[[order[0],n] for n in order[1:]]
    elif topology==3:edges=[[n,order[-1]] for n in order[:-1]]
    else:edges=[[a,b] for i,a in enumerate(order) for b in order[i+1:] if rng.random()<.55]
    rng.shuffle(edges)
    task=dict(objects=objects,points=points,goals=goals,before=edges,
              initial=[-.8,-.8],jammed=[],panel=panel)
    if jam:task['jammed']=[order_goals(task)[-1]]
    return task


def dataset():
    train = [record(make_expanded_task(1000000+i, 2+i%4, panel='training')) for i in range(16384)]
    dev = [record(make_expanded_task(2000000+i, 2+i%4, panel='development')) for i in range(256)]
    test = []
    for j,(panel,count,renamed,jam) in enumerate([
        ('familiar3',3,False,False), ('familiar4',4,False,False), ('familiar5',5,False,False),
        ('long6',6,False,False), ('long7',7,False,False), ('long8',8,False,False),
        ('renamed8',8,True,False), ('jammed8',8,False,True)]):
        test.extend(record(make_expanded_task(3000000+j*1000+i,count,renamed,jam,panel)) for i in range(32))
    return dict(train=train, development=dev, test=test)


def load_library(config):
    specs, blocks = {}, {}
    for kind in KINDS:
        entry = config[kind]
        blocks[kind] = ReLUArtifact.load(Path(entry['path']))
        specs[kind] = entry['spec']
    return blocks, specs


def score(source, task, blocks, specs, keep_trace=True):
    """Judge final geometry AND delivery order, not just interpreter success."""
    text = source.strip()
    fence = re.fullmatch(r'```(?:text)?\s*\n([\s\S]*?)\n```',text)
    if fence:
        text = fence.group(1)
    result = dict(constructed=False, task_success=False, safe_failure=False, panel=task['panel'])
    try:
        program = compile_program(text, task['objects'], task['points'], limit=64)
    except ValueError as error:
        return dict(result, error=str(error))
    world = Workcell(task['initial'], task['objects'], radius=specs['near']['radius'],
                     bound=specs['safe']['bound'], jammed=frozenset(task['jammed']))
    executor = Executor(program, task['points'], blocks, skill_budget=160,total_budget=2048)
    execution = executor.run(world)
    delivered, violations = {}, []
    for i,event in enumerate(executor.trace):
        before, after = event['before'], event['after']
        if not after['safe']:
            violations.append('unsafe')
        for name,target in task['goals'].items():
            near = max(abs(a-b) for a,b in zip(after['objects'][name], task['points'][target])) <= world.radius+1e-9
            if before['held'] == name and after['held'] is None and near:
                if name in delivered:
                    violations.append('repeat_delivery')
                delivered.setdefault(name,i)
    final_positions = all(max(abs(a-b) for a,b in zip(world.objects[n],task['points'][p]))<=world.radius+1e-9
                          for n,p in task['goals'].items())
    ordered = all(a in delivered and b in delivered and delivered[a]<delivered[b] for a,b in task['before'])
    # Reject regrasp after delivery, even if the object later returns to its goal.
    for n,t in delivered.items():
        if any(e['after']['held']==n for e in executor.trace[t+1:]):
            violations.append('moved_after_delivery')
    success = (execution['status']=='success' and final_positions and ordered and
               len(delivered)==len(task['objects']) and world.held is None and
               world.near(task['points']['home']) and not violations)
    unjammed = set(task['objects'])-set(task['jammed'])
    preserved = all(n in delivered and max(abs(a-b) for a,b in zip(world.objects[n],task['points'][task['goals'][n]]))<=world.radius+1e-9 for n in unjammed)
    partial_ordered = all(b not in delivered or (a in delivered and delivered[a]<delivered[b]) for a,b in task['before'])
    safe_failure = bool(task['jammed']) and execution['status']=='failure' and execution['reason']=='budget_exhausted' and preserved and partial_ordered and not violations and world.held is None
    result.update(constructed=True, task_success=success, safe_failure=safe_failure,
                  execution_status=execution['status'], reason=execution['reason'], ticks=execution['ticks'],
                  delivered=delivered, violations=violations, final=world.observe(),
                  missing_deliveries=sorted(set(task['objects'])-set(delivered)),
                  ordering_errors=[[a,b] for a,b in task['before'] if b in delivered and (a not in delivered or delivered[a]>=delivered[b])],
                  instructions=len(program), completed_instructions=executor.pc)
    if keep_trace:
        result['trace'] = execution['trace']
    return result
