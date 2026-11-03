"""Approach-only study construction and finite-budget rollout bookkeeping."""
import hashlib
import json
import math
import torch
from .neural import MotorMLP, MotorRMV, authored_approach, reference_command
from .motor import SkillObserver
from .libero_adapter import LiberoMotorAdapter


class ApproachStudy:
    def __init__(self, seed=7, gain=20.):
        self.gain = gain
        self.models = {"mlp_authored": authored_approach(MotorMLP(seed), gain),
                       "rmv_authored": authored_approach(MotorRMV(seed), gain),
                       "mlp_adamw": MotorMLP(seed), "rmv_adamw": MotorRMV(seed)}

    def command(self, name, channels):
        x = torch.tensor(channels, dtype=torch.float32)
        with torch.no_grad():
            if name == "reference": return reference_command(x, self.gain).tolist()
            if name == "zero": return [0., 0., 0., -1.]
            return self.models[name](x).tolist()


def frame_record(frame):
    return dict(time=frame.time, eef=frame.eef, object=frame.object,
                target=frame.target, channels=frame.channels)


def rollout(env, study, method, seed, max_steps=120, settling_steps=20, standoff_m=.12):
    """One seeded physical episode; caller owns env. No implicit retry/exclusion.

    Twenty open-gripper zero-translation preparation ticks settle the scene.
    Target is a supplied world point above the settled object's origin, not a
    certified grasp pose. Every preparation tick and motor call is recorded.
    """
    result = dict(seed=seed, method=method, status="running", success=False,
                  reset_calls=0, preparation_calls=0, motor_calls=0, trace=[], preparation=[])
    adapter = None
    try:
        import numpy as np
        env.seed(seed)
        result['reset_calls'] += 1
        obs = env.env.reset()  # fail on invalid reset; never wrapper's infinite retry
        for _ in range(settling_steps):
            result['preparation_calls'] += 1
            obs, reward, done, info = env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
            result['preparation'].append(dict(time=float(env.sim.data.time), native_done=bool(done)))
            if done: raise ValueError("native task terminal during scene preparation")
        target = [float(x) for x in obs['cream_cheese_1_pos']]
        target[2] += standoff_m
        adapter = LiberoMotorAdapter(env, 'cream_cheese_1', target, max_steps=max_steps)
        frame = adapter.observe(obs)
        start = frame_record(frame)
        # Compare paired reset identity without method names or irrelevant info.
        result['start'] = start
        result['start_sha256'] = hashlib.sha256(json.dumps(start, sort_keys=True).encode()).hexdigest()
        result['initial_distance_m'] = math.dist(frame.eef, frame.target)
        if result['initial_distance_m'] < .05:
            raise ValueError("trivial start: target less than 5 cm from end effector")
        observer = SkillObserver(); observer.begin('approach', 'approach', frame)
        for _ in range(max_steps):
            channels = frame.channels
            command = study.command(method, channels)
            frame, native = adapter.step(command)
            labels = observer.update(frame)
            result['trace'].append(dict(before_channels=channels, command=command,
                                        after=frame_record(frame), labels=labels, native=native))
            if labels['approach_done']:
                result.update(success=True, status='success'); break
            if native['native_done']:
                result['status'] = 'native_terminal'; break
        else:
            result['status'] = 'timeout'
        result['final_distance_m'] = math.dist(frame.eef, frame.target)
    except Exception as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
    finally:
        result['motor_calls'] = 0 if adapter is None else adapter.step_calls
    return result


def check_feasibility(reference, zero):
    if reference['start_sha256'] != zero['start_sha256']:
        raise ValueError('paired simulator starts differ')
    return reference['success'] and not zero['success']


def audit_initialization(model, x):
    """Read/backward audit; no optimizer step or parameter modification."""
    model.zero_grad(set_to_none=True)
    probe = x.detach().clone().requires_grad_()
    loss = ((model(probe)-reference_command(probe))**2).mean()
    loss.backward()
    report = dict(input_gradient_l1=float(probe.grad.abs().sum()),
                  parameter_gradient_l1={name: float(p.grad.abs().sum()) for name, p in model.named_parameters()})
    if isinstance(model, MotorRMV):
        with torch.no_grad():
            d = model.internals(x)
            report['clause_interior_fraction'] = float(((d['clauses'] > 0) & (d['clauses'] < 1)).float().mean())
            report['head_interior_fraction'] = float(((d['heads'] > 0) & (d['heads'] < 1)).float().mean())
    model.zero_grad(set_to_none=True)
    if not math.isfinite(report['input_gradient_l1']) or report['input_gradient_l1'] <= 1e-9:
        raise ValueError('input-insensitive/nonfinite initialization on source data')
    if any(not math.isfinite(v) or v == 0 for v in report['parameter_gradient_l1'].values()):
        raise ValueError('inactive/nonfinite parameter group on source data')
    return report
