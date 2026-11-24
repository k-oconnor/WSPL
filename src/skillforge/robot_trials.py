"""Shared robot evaluation and public-feedback teaching for the author study."""
import json
from pathlib import Path
from .fresh_simulator import Simulator
from .fresh_robot import public_hash, prompt_record, predict
from .robot_codec import render, decode_action, encode_action
from .reliable_json import save


CONDITIONS = ('ordinary', 'changed', 'switch', 'return')


def panel(seed, per_condition):
    return [dict(seed=seed+1000*c+i, count=3+i % 3, condition=condition, max_decisions=32)
            for c, condition in enumerate(CONDITIONS) for i in range(per_condition)]


def feedback_teacher(public):
    """Supplied training teacher; uses only measured public feedback, no regime."""
    o = public['observation']
    last = o.get('last') or {}
    pending = [g for g in o['requested_order'] if g not in o['observed_visits']]
    if not pending:
        return dict(op='stop')
    if last.get('status') == 'observed' and 'displacement_m' in last:
        dx, dy, _ = last['displacement_m']
        text = 'Original motor' if dx > dy else 'Motor block compensating'
        needed = next(k for k, v in public['documentation'].items() if v.startswith(text))
        return dict(op='select', module=needed)
    if o['selected_module'] is None or (last.get('status') == 'chunk_ended' and
            last['measured_error_m'] >= .95*last['initial_measured_error_m']):
        return dict(op='probe', command=[.12, 0, 0], ticks=6)
    return dict(op='reach', goal=pending[0])


def episode(config, out, manifest, identity, mission, action, guard):
    """The action callback receives public state only; no private simulator object."""
    out = Path(out)
    out.mkdir()
    sim = Simulator(config, out, manifest, identity)
    try:
        public = sim.call(dict(op='start', artifact=config['motor_artifact'],
                              output=str(out/'physical'), **mission))
        start_id = public_hash(public)
        records = []
        while not public['finished']:
            guard()
            command, metadata = action(public)
            records.append(dict(public=public, action=command, **metadata))
            public = sim.call(dict(op='act', action=json.dumps(command)))
        result = sim.call(dict(op='finish', trace_path=str(out/'trace.json.gz')))
        save(out/'decisions.json', records)
        result.update(start_sha256=start_id, **mission)
        save(out/'result.json', result)
        return result, records
    finally:
        sim.close()


def evaluate(model, tokenizer, config, out, manifest, identity, missions, guard, report):
    model.eval()
    model.gradient_checkpointing_disable()
    out = Path(out)
    out.mkdir()
    rows = []
    for i, mission in enumerate(missions):
        report(mission=i, condition=mission['condition'], missions=len(missions))
        def action(public):
            prompt = render(public)
            token, code = predict(model, tokenizer, prompt_record(tokenizer, prompt))
            invalid = False
            try:
                a = decode_action(public, code)
            except ValueError:
                a = dict(op='invalid_model_token', token=token)
                invalid = True
            return a, dict(prompt=prompt, token=token, code=code, invalid=invalid)
        result, records = episode(config, out/f'mission_{i:03d}', manifest, identity,
                                  mission, action, guard)
        rows.append(dict(mission=i, model_calls=len(records),
                         invalid_tokens=sum(r['invalid'] for r in records), **result))
        save(out/'episodes.json', rows)
    return rows


def acquire(tokenizer, config, out, manifest, identity, missions, guard, report):
    out = Path(out)
    out.mkdir()
    rows, results = [], []
    for i, mission in enumerate(missions):
        report(mission=i, condition=mission['condition'], missions=len(missions))
        def action(public):
            a = feedback_teacher(public)
            prompt, code = render(public), encode_action(public, a)
            return a, dict(prompt=prompt, code=code, record=prompt_record(tokenizer, prompt, code))
        result, records = episode(config, out/f'mission_{i:03d}', manifest, identity,
                                  mission, action, guard)
        results.append(result)
        save(out/'teacher_episodes.json', results)
        if not result['success']:
            raise ValueError('Public-feedback teacher failed; stop before fitting/author calls')
        for row in records:
            rows.append(dict(mission=i, condition=mission['condition'], **row))
        save(out/'rows.json', rows)
    return rows


def compare(parent, candidate):
    if len(parent) != len(candidate) or not parent:
        raise ValueError('Complete paired panels required')
    result = {}
    for condition in CONDITIONS:
        pairs = [(a, b) for a, b in zip(parent, candidate) if a['condition'] == condition]
        if not pairs:
            raise ValueError('Missing condition')
        for a, b in pairs:
            if any(a[k] != b[k] for k in ('seed', 'count', 'condition', 'start_sha256')):
                raise ValueError('Mismatched paired mission')
        result[condition] = dict(n=len(pairs), parent=sum(a['success'] for a, b in pairs),
            candidate=sum(b['success'] for a, b in pairs),
            corrections=sum(not a['success'] and b['success'] for a, b in pairs),
            regressions=sum(a['success'] and not b['success'] for a, b in pairs))
    ordinary = result['ordinary']
    adaptation = [result[c] for c in CONDITIONS[1:]]
    return dict(conditions=result, parent_ready=ordinary['parent'] >= 12,
        exploratory_gate=ordinary['parent'] >= 12 and ordinary['candidate'] >= 14
            and ordinary['regressions'] <= 1
            and sum(r['corrections']-r['regressions'] for r in adaptation) >= 8
            and all(r['corrections'] >= r['regressions'] for r in adaptation))
