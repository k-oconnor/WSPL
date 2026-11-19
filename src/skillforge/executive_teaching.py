"""Physical mission boundary and teacher provenance for student-owned control."""
import json
import random
from pathlib import Path
import numpy as np
from .factory_observation import PandaBackend,ObservationSession
from .student_control import StudentExecutiveEpisode,prompt
from .wspl_artifact import ReLUArtifact

CONDITIONS=('ordinary','changed','switch','return')
CHANGED=[[0,-.8,0],[1.2,0,0],[0,0,.9]]


class MonitoredBackend:
    def __init__(self,backend):
        self.backend=backend;self.true_trace=[];self.target=None;self.stable=0
        self.goals={};self.visits=[];self.order_violations=[]
    def sensors(self):return self.backend.sensors()
    def forward_kinematics(self,q):return self.backend.forward_kinematics(q)
    def render(self,camera):return self.backend.render(camera)
    def step(self,command):
        self.backend.step(command)
        position=self.backend.env.sim.data.site_xpos[self.backend._eef_site].copy()
        distance=None
        if self.target in self.goals:
            distance=float(np.linalg.norm(position-np.asarray(self.goals[self.target])))
            self.stable=self.stable+1 if distance<=.005 else 0
            if self.stable>=3 and self.target not in self.visits:
                expected=list(self.goals)[len(self.visits)] if len(self.visits)<len(self.goals) else None
                if self.target!=expected:self.order_violations.append(self.target)
                self.visits.append(self.target)
        else:self.stable=0
        self.true_trace.append(dict(timestamp_s=float(self.backend.sensors()['timestamp_s']),
            command=list(map(float,command)),position_m=position.tolist(),target=self.target,distance_m=distance))


class TeachingMission:
    def __init__(self,artifact_path,output,seed,count,condition,max_decisions=None,backend_factory=PandaBackend):
        if condition not in CONDITIONS or not 2<=count<=5:raise ValueError('Invalid mission')
        artifact=json.loads(Path(artifact_path).read_text())
        self.condition=condition;self.changed=False
        initial=CHANGED if condition in ('changed','return') else np.eye(3).tolist()
        self.backend=backend_factory(seed,initial);self.monitor=MonitoredBackend(self.backend)
        self.session=ObservationSession(self.monitor,output,seed+100000,max_steps=60*count+80,max_frames=0)
        rng=random.Random(seed);names=rng.sample(range(1000,9999),count+2)
        self.modules={f'tool_{names[0]}':ReLUArtifact(artifact['original']),f'tool_{names[1]}':ReLUArtifact(artifact['repair'])}
        self.nominal,self.compensated=list(self.modules)
        self.documentation={self.nominal:'Original motor block for nominal XYZ command calibration.',
            self.compensated:'Motor block compensating a cross-axis calibration: positive x command moves mainly +y; positive y command moves mainly -x; positive z remains +z.'}
        # Neither order nor ID suffix encodes the recommended module.
        if rng.random()<.5:
            self.modules=dict(reversed(list(self.modules.items())));self.documentation=dict(reversed(list(self.documentation.items())))
        origin=np.asarray(self.session.observe()['nominal_tool_position_m'])
        offsets=[(.04,0,.02),(-.04,0,.02),(0,.04,.02),(0,-.04,.02),(.03,.03,.02),(-.03,.03,.02),(.03,-.03,.02),(-.03,-.03,.02)]
        selected=rng.sample(offsets,count)
        self.goals={f'goal_{names[i+2]}':(origin+np.asarray(p)+np.array([rng.uniform(-.003,.003),rng.uniform(-.003,.003),0])).tolist() for i,p in enumerate(selected)}
        self.monitor.goals=self.goals
        self.episode=StudentExecutiveEpisode(self.session,self.modules,self.goals,max_decisions=max_decisions or 4*count+12)
        self.last_mission_state=None

    def public(self):
        return dict(observation=self.episode.observe(),documentation=self.documentation,finished=self.episode.finished)

    def execute(self,text):
        try:
            action=json.loads(text);target=action.get('goal') if isinstance(action,dict) and action.get('op')=='reach' else None
        except (ValueError,TypeError):target=None
        if target!=self.monitor.target:self.monitor.stable=0
        self.monitor.target=target
        outcome=self.episode.execute(text)
        if not self.changed and self.monitor.visits and self.condition in ('switch','return'):
            self.backend.mapping=np.array(CHANGED if self.condition=='switch' else np.eye(3),dtype=float)
            self.changed=True
        if self.session.steps>=self.session.max_steps or len(self.episode.decisions)>=self.episode.max_decisions:
            self.episode.finished=True
        return dict(outcome=outcome,**self.public())

    def result(self):
        stopped=self.episode.last is not None and self.episode.last.get('status')=='stopped'
        success=stopped and self.monitor.visits==list(self.goals) and not self.monitor.order_violations
        return dict(success=success,stopped=stopped,physical_visits=self.monitor.visits,
            requested_order=list(self.goals),order_violations=self.monitor.order_violations,
            physical_ticks=self.session.steps,decisions=len(self.episode.decisions),
            switches=sum(1 for d in self.episode.decisions if d['outcome'].get('status')=='selected'),
            probes=sum(1 for d in self.episode.decisions if d['outcome'].get('status')=='observed'),
            setup_settling_ticks=20,trace=self.monitor.true_trace,decision_trace=self.episode.decisions,
            sensor_trace=self.session.log)

    def close(self):self.backend.close()


def ordinary_action(public):
    """Rehearsal teacher ONLY for declared ordinary-data collection, never deployment."""
    observation=public['observation']
    nominal=next(k for k,v in public['documentation'].items() if v.startswith('Original motor'))
    if observation['selected_module']!=nominal:return dict(op='select',module=nominal)
    pending=[g for g in observation['requested_order'] if g not in observation['observed_visits']]
    return dict(op='reach',goal=pending[0]) if pending else dict(op='stop')


def teacher_prompt(public):
    # Teacher gets exactly the student observation/documentation. No private result/history.
    return ('You are Astra demonstrating robot executive decisions for a weaker language model. '
            'Return JSON with exactly prediction (a short prospective explanation) and action '
            '(one JSON object in the student action schema). No other tools. '
            'Your decisions will be executed physically and may be used as imitation labels. '
            'Choose using only the supplied observations. Do not assume the regime or schedule.\n'+
            prompt(public['observation'],public['documentation'])+
            '\nFor this TEACHER reply, wrap that one action as '
            '{"prediction":"short prospective prediction","action":{...}}. '
            'This wrapper overrides the student-only reply format; the inner action must follow it.')


def parse_teacher(text):
    item=json.loads(text)
    if not isinstance(item,dict) or set(item)!={'prediction','action'} or not isinstance(item['prediction'],str) or not item['prediction'].strip() or not isinstance(item['action'],dict):
        raise ValueError('Expected prospective prediction and one action object')
    return item['prediction'],json.dumps(item['action'],separators=(',',':'))


def eligible_rows(episodes):
    rows=[];counts={c:0 for c in CONDITIONS}
    for episode in episodes:
        if not episode['result']['success']:continue
        counts[episode['condition']]+=1
        for record in episode['records']:
            if record['accepted']:
                rows.append(dict(mission=episode['mission'],condition=episode['condition'],messages=[
                    dict(role='user',content=record['student_prompt']),dict(role='assistant',content=record['action'])]))
    if not all(counts.values()) or not rows:raise ValueError('No usable successful teacher coverage; stop before fitting')
    return rows,counts
