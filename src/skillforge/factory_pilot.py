"""Bounded factory-interface authoring. No privileged state in public packets."""
import json
from fractions import Fraction
import numpy as np
from .factory_observation import PandaBackend, ObservationSession
from .wspl_parse import parse_module
from .wspl_compile import compile_scalar

MOTOR_SOURCE='''module Move {
input x: Real[-2,2] scalar;
input y: Real[-2,2] scalar;
input z: Real[-2,2] scalar;
output vx: Real[-1,1] scalar;
output vy: Real[-1,1] scalar;
output vz: Real[-1,1] scalar;
vx = clip(10*x,-0.2,0.2);
vy = clip(10*y,-0.2,0.2);
vz = clip(10*z,-0.2,0.2);
}'''
SELECT_SOURCE='''module Select {
input u: Real[-1,1] scalar;
input v: Real[-1,1] scalar;
input w: Real[-1,1] scalar;
input dx: Real[-1,1] scalar;
input dy: Real[-1,1] scalar;
input dz: Real[-1,1] scalar;
output original: Real[-8,8] scalar;
output repaired: Real[-8,8] scalar;
original = 1;
repaired = 0;
}'''


def compile_block(source, kind):
    if not isinstance(source,str) or len(source.encode())>24000:raise ValueError('Source limit')
    a=compile_scalar(parse_module(source,max_nodes=256).program,max_parameters=1024,max_layers=4,max_units=64,max_nodes=256)
    ins,outs=(('x','y','z'),('vx','vy','vz')) if kind=='motor' else (('u','v','w','dx','dy','dz'),('original','repaired'))
    if a.input_names!=ins or a.output_names!=outs:raise ValueError('Port mismatch')
    required=2 if kind=='motor' else 1
    if any(Fraction(p['type']['lower'])>-required or Fraction(p['type']['upper'])<required for p in a.to_dict()['inputs']):
        raise ValueError('Insufficient input domain')
    return a


def values(a,x):
    y=a.evaluate({k:Fraction(str(float(v))) for k,v in zip(a.input_names,x)})
    result=np.array([float(y[k]) for k in a.output_names])
    if not np.isfinite(result).all():raise ValueError('Nonfinite neural output')
    return result


class FactoryStudy:
    def __init__(self):
        self.original=compile_block(MOTOR_SOURCE,'motor')
        self.repair=compile_block(MOTOR_SOURCE,'motor')
        self.selector=compile_block(SELECT_SOURCE,'selector')
        self.probe=[0.,0.,0.];self.probe_ticks=1
        self.sources=dict(motor=MOTOR_SOURCE,selector=SELECT_SOURCE)

    def install(self,request):
        probe=np.asarray(request['probe'],dtype=float)
        if probe.shape!=(3,) or not np.isfinite(probe).all() or np.max(abs(probe))>.2:raise ValueError('Invalid probe')
        ticks=request['probe_ticks']
        if type(ticks) is not int or not 1<=ticks<=10:raise ValueError('Probe ticks 1..10')
        repair=compile_block(request['motor'],'motor');selector=compile_block(request['selector'],'selector')
        self.repair,self.selector,self.probe,self.probe_ticks=repair,selector,probe.tolist(),ticks
        self.sources=dict(motor=request['motor'],selector=request['selector'])

    def artifact(self):
        return dict(original=self.original.to_dict(),repair=self.repair.to_dict(),selector=self.selector.to_dict(),
                    probe=self.probe,probe_ticks=self.probe_ticks,sources=self.sources)


def rollout(study,folder,seed,mapping,offset,arm,cameras=()):
    """Taught relative waypoints; physics truth used ONLY for independent scoring."""
    backend=PandaBackend(seed,mapping)
    try:
        session=ObservationSession(backend,folder,seed+100000,max_steps=120,max_frames=8)
        initial=session.observe()
        goal=np.asarray(initial['nominal_tool_position_m'])+np.asarray(offset)
        for camera in cameras:session.capture(camera,'mission_start')
        motor=study.original;selected='original';diagnostic=None
        if arm in ('selected','matched_original','reference'):
            start=np.asarray(initial['nominal_tool_position_m'])
            session.act([*study.probe,-1.],study.probe_ticks)
            # Settling is real time and remains in the scene; no reset/teleport.
            session.act([0.,0.,0.,-1.],5)
            delta=np.asarray(session.observe()['nominal_tool_position_m'])-start
            scores=values(study.selector,[*study.probe,*delta])
            selected='repaired' if scores[1]>scores[0] else 'original'
            diagnostic=dict(displacement_m=delta.tolist(),scores=scores.tolist(),selected=selected)
            if arm=='selected' and selected=='repaired':motor=study.repair
        consecutive=0;success=False;errors=[]
        while session.steps<session.max_steps:
            error=goal-np.asarray(session.observe()['nominal_tool_position_m'])
            command=values(motor,error)
            if arm=='reference':command=np.linalg.solve(np.asarray(mapping),values(study.original,error))
            command=np.clip(command,-.2,.2)
            session.act([*command,-1.],1)
            actual=backend.env.sim.data.site_xpos[backend._eef_site].copy()
            distance=float(np.linalg.norm(goal-actual));errors.append(distance)
            session.log.append(dict(kind='evaluator_private',sequence=session.steps,
                                    actual_tool_position_m=actual.tolist(),distance_m=distance))
            consecutive=consecutive+1 if distance<=.005 else 0
            if consecutive>=3:success=True;break
        for camera in cameras:session.capture(camera,'mission_end')
        return dict(success=success,steps=session.steps,goal_m=goal.tolist(),
                    setup_settling_steps=20,
                    final_error_m=errors[-1],final_measured_error_m=float(np.linalg.norm(goal-np.asarray(session.observe()['nominal_tool_position_m']))),
                    final_telemetry=session.observe(),diagnostic=diagnostic,frames=session.frames,log=session.log)
    finally:backend.close()


def packet(history,remaining):
    instructions='''You are Astra, engineering a robot controller from observations.
Two robot calibration benches A/B have the same nominal Panda/OSC interface, but may have different command calibration.
You are not given their true mappings. Telemetry is noisy/quantized encoders, velocities, gripper joints, and nominal FK pose.
Images are real simulated frontview or robot0_eye_in_hand camera frames. No simulator object coordinates or fault labels are supplied.
Factory task: move the nominal tool to a taught waypoint, retain performance under original calibration, and adapt to changed calibration.
This is a robot motion-interface pilot, not pick-and-place or transfer of our planar Qwen executive.
No local optimizer is available. Write WSPL neural circuits, not external inference code.
Original controller is frozen. Install a new error-to-action block and a selector plus a diagnostic probe.
At deployment Astra is absent. Each mission runs your probe, then five zero-translation settling ticks.
The selector receives only probe u,v,w and noisy measured dx,dy,dz. It selects original or repaired for that mission.
The selected motor receives goal-minus-nominal-tool-position x,y,z. Host clips translation outputs to [-.2,.2]; gripper stays open.
No camera interpretation or bench label is supplied to the deployed circuits. Images help YOU investigate; they are not deployment features.
Each control tick is 0.05s. OSC translation input 1 corresponds nominally to a 0.05m target increment; measured motion has dynamics.
Probes, settling, and all movement count against the 120-tick mission budget. Never assume infinitesimal probes are observable.
WSPL uses module declarations, Real[lo,hi] scalar ports, semicolons, affine arithmetic, relu, min, max, clip; no variable products or division.
Motor inputs x,y,z must cover [-2,2], outputs vx,vy,vz. Selector inputs u,v,w,dx,dy,dz cover [-1,1], outputs original,repaired.
Each block at most1024 parameters, four layers,64 units. Repaired wins iff its score is larger; ties choose original.
You may acquire observations from A/B, but final evaluation uses fresh scenes/sensor seeds and no bench identifier.
Only request prospective captures; there is no retrospective buffer. Event triggers are command_start,command_end,gripper_command_change.
A gripper command is NOT evidence of a successful grasp. Images carry timestamps and camera labels. Old images are not automatically reattached.
Return exactly one JSON request with tool and a prediction written BEFORE the requested operation:
observe: {tool,prediction,bench,cameras} cameras subset of [frontview,robot0_eye_in_hand].
act: {tool,prediction,bench,command,ticks,cameras,events,clip_stride}; command=[vx,vy,vz,grip], abs translation<=.2, abs grip<=1; ticks1..20; clip_stride null or1..ticks.
install: {tool,prediction,probe,probe_ticks,motor,selector}; probe three translations <=.2; ticks1..10; source strings.
evaluate: {tool,prediction,cameras}; checks last installed artifact on two calibration waypoint missions.
finish: {tool,prediction}; freezes LAST accepted artifact, no best-of selection.
Per bench:160 action ticks,48 frames. At most12 images per reply. Two evaluate calls. Twelve total responses.
No external tools, shell, data access, or requests to the human. All failures consume a response. Preserve old-task successes.
Original block:\n'''
    return instructions+MOTOR_SOURCE+'\nDefault selector:\n'+SELECT_SOURCE+'\nRemaining:\n'+json.dumps(remaining)+'\nVisible history:\n'+json.dumps(history)


def request(text):
    a=json.loads(text)
    schemas={'observe':{'bench','cameras'},'act':{'bench','command','ticks','cameras','events','clip_stride'},
             'install':{'probe','probe_ticks','motor','selector'},'evaluate':{'cameras'},'finish':set()}
    if not isinstance(a,dict) or a.get('tool') not in schemas or set(a)!={'tool','prediction'}|schemas[a['tool']]:raise ValueError('Request schema')
    if not isinstance(a['prediction'],str) or not a['prediction'].strip():raise ValueError('Prediction required')
    if 'bench' in a and a['bench'] not in ('A','B'):raise ValueError('Unknown bench')
    return a
