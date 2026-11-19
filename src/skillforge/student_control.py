"""Student-owned executive decisions over frozen motor modules.

No automatic diagnostic probe, selector, repair, goal selection or retry.
The session supplies the public observation boundary and bounded physical actions.
This module is engineering infrastructure; no training/evaluation entry point.
"""
import copy
import json
import numpy as np
from .factory_pilot import values


class StudentExecutiveEpisode:
    def __init__(self, session, modules, goals, max_decisions=16, reach_ticks=10):
        if not modules or not goals or not 1 <= reach_ticks <= 20:
            raise ValueError('Nonempty module/goal inventory and bounded reach chunk required')
        if type(max_decisions) is not int or max_decisions < 1:
            raise ValueError('Positive decision budget required')
        for name,module in modules.items():
            if not isinstance(name,str) or not name or module.input_names!=('x','y','z') or module.output_names!=('vx','vy','vz'):
                raise ValueError('Motor interface mismatch')
        for name,point in goals.items():
            if not isinstance(name,str) or not name or np.asarray(point).shape!=(3,) or not np.isfinite(point).all():
                raise ValueError('Finite named waypoints required')
        self.session=session;self.modules=dict(modules);self.goals=copy.deepcopy(goals)
        self.max_decisions=max_decisions;self.reach_ticks=reach_ticks
        self.selected=None;self.last=None;self.decisions=[];self.finished=False
        self.observed_visits=[]

    def observe(self):
        return dict(telemetry=self.session.observe(),goals=copy.deepcopy(self.goals),
                    requested_order=list(self.goals),observed_visits=list(self.observed_visits),
                    available_modules=list(self.modules),selected_module=self.selected,
                    last=copy.deepcopy(self.last),remaining_decisions=self.max_decisions-len(self.decisions))

    def execute(self,text):
        if self.finished:raise ValueError('Episode finished')
        before=self.observe();record=dict(source=text,before=before)
        self.decisions.append(record)
        if len(self.decisions)>self.max_decisions:
            self.finished=True;self.last=dict(status='error',reason='decision_budget');record['outcome']=self.last
            return copy.deepcopy(self.last)
        try:
            a=json.loads(text)
            schemas={'probe':{'command','ticks'},'select':{'module'},'reach':{'goal'},'stop':set()}
            if not isinstance(a,dict) or a.get('op') not in schemas or set(a)!={'op'}|schemas[a['op']]:
                raise ValueError('Exactly one supported action required')
            op=a['op']
            if op=='select':
                if a['module'] not in self.modules:raise ValueError('Unknown module')
                self.selected=a['module'];result=dict(status='selected',module=self.selected)
            elif op=='probe':
                command=np.asarray(a['command'],dtype=float)
                if command.shape!=(3,) or not np.isfinite(command).all():raise ValueError('Three probe components required')
                # The public session validates physical bounds/budget before executing.
                start=before['telemetry']['nominal_tool_position_m']
                self.session.act([*command,-1.],a['ticks'])
                end=self.session.observe()['nominal_tool_position_m']
                result=dict(status='observed',command=command.tolist(),ticks=a['ticks'],
                            displacement_m=(np.asarray(end)-np.asarray(start)).tolist())
            elif op=='reach':
                if self.selected is None:raise ValueError('Select a controller explicitly first')
                if a['goal'] not in self.goals:raise ValueError('Unknown goal')
                goal=np.asarray(self.goals[a['goal']],dtype=float);stable=0;used=0
                initial_error=float(np.linalg.norm(goal-np.asarray(before['telemetry']['nominal_tool_position_m'])))
                for _ in range(self.reach_ticks):
                    if self.session.steps>=self.session.max_steps:break
                    error=goal-np.asarray(self.session.observe()['nominal_tool_position_m'])
                    command=np.clip(values(self.modules[self.selected],error),-.2,.2)
                    self.session.act([*command,-1.],1);used+=1
                    error=goal-np.asarray(self.session.observe()['nominal_tool_position_m'])
                    stable=stable+1 if float(np.linalg.norm(error))<=.005 else 0
                    if stable>=3:break
                measured=float(np.linalg.norm(goal-np.asarray(self.session.observe()['nominal_tool_position_m'])))
                reached=stable>=3
                if reached:self.observed_visits.append(a['goal'])
                result=dict(status='observed_reached' if reached else 'chunk_ended',goal=a['goal'],
                            module=self.selected,ticks=used,initial_measured_error_m=initial_error,measured_error_m=measured)
            else:
                self.finished=True;result=dict(status='stopped')
            self.last=result
        except (ValueError,TypeError,KeyError) as error:
            self.last=dict(status='invalid_action',reason=str(error))
        record['outcome']=copy.deepcopy(self.last)
        return copy.deepcopy(self.last)


def prompt(observation,module_documentation):
    return ('You are the robot executive. Complete the requested waypoint sequence, then stop. '
            'Choose every action yourself using observed feedback. No external teacher is available. '
            'Return exactly one JSON action, no prose: '
            '{"op":"probe","command":[x,y,z],"ticks":N}, '
            '{"op":"select","module":"ID"}, {"op":"reach","goal":"ID"}, or {"op":"stop"}. '
            'Probe issues physical motion; it does not select a controller. Select changes no weights. '
            'Probe components must lie in [-0.2,0.2], ticks must be an integer1..20. '
            'Control runs at20Hz; a translation input of1 nominally requests a0.05m target increment, not guaranteed displacement. '
            'Reach attempts only the requested waypoint for a bounded chunk; it does not retry or advance goals. '
            'Status observed_reached is a sensor-based observation, not a perfect physical certificate. '
            'Calibration may differ or change; do not assume an error means a particular fault. '
            'Preserve correct behavior on ordinary missions.\nModule documentation:\n'+
            json.dumps(module_documentation)+'\nObservation:\n'+json.dumps(observation))
