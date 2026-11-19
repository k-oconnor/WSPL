"""Observed-state compact control and independent mission scoring.

Kinematic engineering environment, not contact physics. Disturbances change the
world; they never choose or repair the executive's action.
"""
from dataclasses import dataclass
import hashlib
import json
import random
import re

from .control_dsl import Workcell, Executor, compile_program
from .executive_study import canonical, order_goals


PANELS = [('familiar3',3,'none'),('familiar5',5,'none'),
          ('long8',8,'none'),('long12',12,'none'),('renamed12',12,'none'),
          ('slip8',8,'slip'),('moved8',8,'moved'),('blocked8',8,'blocked'),
          ('combined12',12,'combined')]


def make_task(seed,count,disturbance='none',panel='engineering',renamed=False):
    if not 2<=count<=12 or disturbance not in ('none','slip','moved','blocked','combined'):
        raise ValueError('Unsupported task configuration')
    rng=random.Random(seed);names=[]
    while len(names)<count:
        name=('Unit_' if renamed else 'obj_')+''.join(rng.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ' if renamed else 'abcdefghijklmnopqrstuvwxyz',k=4))+str(rng.randrange(10,100))
        if name not in names:names.append(name)
    sources=[(x,y) for x in (-.72,-.47,-.22) for y in (-.6,-.2,.2,.6)]
    targets=[(x,y) for x in (.22,.47,.72) for y in (-.6,-.2,.2,.6)]
    rng.shuffle(sources);rng.shuffle(targets)
    jitter=lambda p:[round(v+rng.uniform(-.012,.012),4) for v in p]
    objects={n:jitter(sources[i]) for i,n in enumerate(names)}
    points={'home':[-.8,-.8]};goals={}
    labels=rng.sample(range(1000,9999),count)
    for i,n in enumerate(names):
        dest='site_'+str(labels[i]);goals[n]=dest;points[dest]=jitter(targets[i])
    order=names.copy();rng.shuffle(order);family=(seed//4)%5
    if family==0:edges=[[order[i],order[i+1]] for i in range(count-1)]
    elif family==1:edges=[]
    elif family==2:edges=[[order[0],n] for n in order[1:]]
    elif family==3:edges=[[n,order[-1]] for n in order[:-1]]
    else:edges=[[a,b] for i,a in enumerate(order) for b in order[i+1:] if rng.random()<.4]
    # A temporarily blocked goal must leave another enabled delivery available.
    # Training and test disclose only the blocked point, not its release trigger.
    if disturbance in ('blocked','combined'):edges=[]
    task=dict(objects=objects,points=points,goals=goals,before=edges,
              initial=[-.8,-.8],panel=panel,disturbance=disturbance,
              dependency_family='independent' if disturbance in ('blocked','combined') else
              ('chain','independent','fan_out','fan_in','random_dag')[family])
    task['event_object']=order_goals(task)[0]
    return task


class ActuatorWorld(Workcell):
    motion_scale=1.

    def apply(self,command):
        # Command contract remains in actuator coordinates; this experiment uses +/-1.
        if self.motion_scale not in (-1.,1.):raise ValueError('Unsupported actuator mapping')
        super().apply(dict(command,vx=command['vx']*self.motion_scale,vy=command['vy']*self.motion_scale))


@dataclass
class Episode:
    task: dict
    blocks: dict
    specs: dict
    motion_scale: float = 1.

    def __post_init__(self):
        self.world=ActuatorWorld(self.task['initial'],self.task['objects'],radius=self.specs['near']['radius'],bound=self.specs['safe']['bound'])
        self.world.motion_scale=self.motion_scale
        self.delivered={};self.violations=[];self.trace=[];self.decisions=[]
        self.events=[];self.triggered=set();self.failures={};self.last=None
        self.finished=False;self.stopped=False;self.error=None;self.ticks=0
        self.decision_budget=2*len(self.task['objects'])+8

    def blocked(self):
        if self.task['disturbance'] in ('blocked','combined'):
            target=self.task['event_object']
            if not any(n!=target for n in self.delivered):return [self.task['goals'][target]]
        return []

    def observation(self):
        return dict(goals=self.task['goals'],before=self.task['before'],
                    delivered=sorted(self.delivered),held=self.world.held,
                    blocked_points=self.blocked(),at_home=self.world.near(self.task['points']['home']),
                    last=self.last,failed_attempts=dict(self.failures),safe=self.world.safe())

    def execute(self,source):
        if self.finished:raise ValueError('Episode already terminal')
        source=source.strip()
        self.decisions.append(dict(source=source,observation=self.observation()))
        # One compact action, never silently truncate a multi-action answer.
        if '\n' in source or '\r' in source:
            self.finished=True;self.error='multiple_actions';return
        if len(self.decisions)>self.decision_budget:
            self.finished=True;self.error='decision_budget';return
        if source=='stop':
            self.finished=True;self.stopped=True;return
        try:program=compile_program(source,self.task['objects'],self.task['points'],limit=4)
        except ValueError as error:
            self.finished=True;self.error='invalid_action: '+str(error);return
        match=re.fullmatch(r'transfer (\w+) to (\w+)',source)
        if match and match[2] in self.blocked():
            self.last=dict(action=source,status='failure',reason='destination_blocked')
            self.failures[source]=self.failures.get(source,0)+1
            self.decisions[-1]['outcome']=self.last;return
        # Prevent primitive movement into a declared blocked destination as well.
        if any(c.operator=='move_to' and c.argument in self.blocked() for c in program):
            self.last=dict(action=source,status='failure',reason='destination_blocked')
            self.failures[source]=self.failures.get(source,0)+1
            self.decisions[-1]['outcome']=self.last;return
        executor=Executor(program,self.task['points'],self.blocks,skill_budget=160,total_budget=max(1,4096-self.ticks))
        target=self.task['event_object'];grasp_attempt=False
        while executor.status=='running':
            current=program[executor.pc]
            if (current.operator=='grasp' and current.argument==target and
                self.task['disturbance'] in ('slip','combined') and 'slip' not in self.triggered
                and self.world.near(self.world.objects[target])):
                self.world.jammed=frozenset({target});grasp_attempt=True
            executor.tick(self.world)
            event=executor.trace[-1];before,after=event['before'],event['after']
            if not after['safe']:self.violations.append('unsafe')
            for name,point in self.task['goals'].items():
                near=max(abs(a-b) for a,b in zip(after['objects'][name],self.task['points'][point]))<=self.world.radius+1e-9
                if before['held']==name and after['held'] is None and near:
                    if name in self.delivered:self.violations.append('repeat_delivery')
                    if any(b==name and a not in self.delivered for a,b in self.task['before']):
                        self.violations.append('ordering')
                    self.delivered.setdefault(name,len(self.trace))
                if name in self.delivered and after['held']==name:self.violations.append('moved_after_delivery')
            self.trace.append(event);self.ticks+=1
            if (event['skill_status']=='success' and current.operator=='move_to' and current.argument==target
                and self.world.held is None and 'moved' not in self.triggered
                and self.task['disturbance'] in ('moved','combined')):
                x,y=self.world.objects[target]
                self.world.objects[target]=(x,y+(.13 if y<.5 else -.13))
                self.triggered.add('moved');self.events.append(dict(kind='moved',after_tick=self.ticks,object=target,
                                                                  before=[x,y],after=list(self.world.objects[target])))
        if grasp_attempt:
            self.triggered.add('slip');self.world.jammed=frozenset()
            self.events.append(dict(kind='slip',after_tick=self.ticks,object=target))
        self.last=dict(action=source,status=executor.status,reason=executor.reason)
        self.decisions[-1]['outcome']=self.last
        if executor.status=='failure':self.failures[source]=self.failures.get(source,0)+1
        if self.violations or self.ticks>=4096:
            self.finished=True;self.error='safety_or_tick_budget'

    def result(self,keep_trace=True):
        positions=all(max(abs(a-b) for a,b in zip(self.world.objects[n],self.task['points'][p]))<=self.world.radius+1e-9
                      for n,p in self.task['goals'].items())
        complete=(self.stopped and len(self.delivered)==len(self.task['objects']) and positions
                  and self.world.held is None and self.world.near(self.task['points']['home']) and not self.violations)
        result=dict(task_success=bool(complete),error=self.error,stopped=self.stopped,
                    all_actions_valid=bool(self.decisions) and not (self.error and self.error.startswith(('invalid_action','multiple_actions'))),
                    delivered=sorted(self.delivered),missing=sorted(set(self.task['objects'])-set(self.delivered)),
                    violations=self.violations,ticks=self.ticks,decisions=len(self.decisions),
                    failed_calls=sum(self.failures.values()),events=self.events,final=self.world.observe())
        if keep_trace:result.update(trace=self.trace,decision_trace=self.decisions)
        return result


def teacher(observation):
    """Reference policy sees exactly the learner observation, no hidden event state."""
    if not observation['safe']:return 'stop'
    goals=observation['goals'];done=set(observation['delivered']);held=observation['held']
    if held is not None:
        # Teacher trajectories use complete transfers; unsupported stranded states abort.
        return 'stop'
    pending=set(goals)-done
    ready=sorted(n for n in pending if goals[n] not in observation['blocked_points'] and
                 all(a in done for a,b in observation['before'] if b==n))
    if ready:return 'transfer '+ready[0]+' to '+goals[ready[0]]
    if pending:return 'stop'
    return 'stop' if observation['at_home'] else 'move_to home'


def prompt(observation):
    return ('Choose exactly ONE compact action: transfer OBJECT to POINT, move_to OBJECT_OR_POINT, '
        'grasp OBJECT, release OBJECT, or stop. No explanations, fences or extra lines. '
        'Deliver every object to its goal, obey before pairs [A,B], and never regrasp a delivered object. '
        'Among currently enabled undelivered goals with unblocked destinations choose the alphabetically first object. '
        'A failure can leave a goal unfinished: use the new observation to decide the next action. '
        'Return the empty gripper home after all deliveries, then stop. Do not stop with reachable unfinished goals. '
        'Observed state: '+canonical(observation))


def records(task,blocks,specs):
    episode=Episode(task,blocks,specs);rows=[]
    mission_id=hashlib.sha256(canonical(task).encode()).hexdigest()
    while not episode.finished:
        obs=episode.observation();action=teacher(obs)
        rows.append(dict(mission_id=mission_id,observation=obs,
                         messages=[dict(role='user',content=prompt(obs)),dict(role='assistant',content=action)]))
        episode.execute(action)
    if not episode.result(False)['task_success']:raise ValueError('Reference mission infeasible')
    return rows


def training_data(blocks,specs,train_rows=16384,dev_rows=256,guard=lambda:None,progress=lambda **kwargs:None):
    data={}
    for split,count,base in [('train',train_rows,7000000),('development',dev_rows,8000000)]:
        rows=[];i=0
        while len(rows)<count:
            guard()
            disturbance=['none','none','slip','moved','blocked'][i%5]
            rows.extend(records(make_task(base+i,2+i%4,disturbance,panel=split),blocks,specs));i+=1
            if i%16==0:progress(split=split,completed=min(len(rows),count),total=count,missions=i)
        # Select snapshots only after assigning complete missions to disjoint splits.
        random.Random(base).shuffle(rows);data[split]=rows[:count]
    return data


def evaluation_tasks():
    return [make_task(9000000+j*1000+4*i,n,dist,panel=panel,renamed=panel.startswith('renamed'))
            for j,(panel,n,dist) in enumerate(PANELS) for i in range(16)]
