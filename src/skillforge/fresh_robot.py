"""Fresh-machine prerequisite, not a continuation of archived checkpoints.

Analytic motor fixtures are supplied infrastructure. Only the ordinary executive
is trained by the separately source-gated runner. No teacher at deployment.
"""
import hashlib
import json
from pathlib import Path

from .factory_pilot import MOTOR_SOURCE, compile_block


def motor_artifact():
    # Inverse of CHANGED: x_command=y/1.2, y_command=-x/.8, z_command=z/.9.
    # These coefficients are an explicit engineering construction, not learned.
    repaired = MOTOR_SOURCE.replace('10*x', '(25/3)*y').replace(
        '10*y', '(-25/2)*x').replace('10*z', '(100/9)*z')
    return dict(schema=1, provenance='supplied_analytic_motors_v1',
                original=compile_block(MOTOR_SOURCE, 'motor').to_dict(),
                repair=compile_block(repaired, 'motor').to_dict(),
                sources=dict(original=MOTOR_SOURCE, repair=repaired))


def panel(start, size):
    if type(start) is not int or type(size) is not int or start < 0 or size < 1:
        raise ValueError('Positive panel size and nonnegative seed required')
    return [dict(seed=start+i, count=3+i % 3, condition='ordinary', max_decisions=32)
            for i in range(size)]


def validate_panels(training, reserved):
    seeds = [r['seed'] for r in training+reserved]
    if len(seeds) != len(set(seeds)):
        raise ValueError('Overlapping mission seeds')
    if any(r['condition'] != 'ordinary' for r in training+reserved):
        raise ValueError('Prerequisite is ordinary-only')


def prompt_record(tokenizer, prompt, code=None, max_tokens=2048):
    rendered = tokenizer.apply_chat_template(
        [dict(role='user', content=prompt)], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, add_special_tokens=False)['input_ids']
    if not ids or len(ids) > max_tokens:
        raise ValueError('Prompt length outside contract; no truncation')
    result = dict(ids=ids)
    if code is not None:
        target = tokenizer(code, add_special_tokens=False)['input_ids']
        if len(target) != 1 or tokenizer.decode(target, skip_special_tokens=False) != code:
            raise ValueError('Action must be exactly one reversible token')
        result['target'] = target[0]
    return result


def token_loss(model, record):
    import torch
    import torch.nn.functional as F
    device = next(model.parameters()).device
    ids = torch.tensor([record['ids']], dtype=torch.long, device=device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   use_cache=False, logits_to_keep=1).logits[:, -1].float()
    return F.cross_entropy(logits, torch.tensor([record['target']], device=device))


def predict(model, tokenizer, record):
    import torch
    device = next(model.parameters()).device
    ids = torch.tensor([record['ids']], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       use_cache=False, logits_to_keep=1).logits[0, -1]
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite deployment logits')
        token = int(logits.argmax())  # Full vocabulary; no action-code mask.
    return token, tokenizer.decode([token], skip_special_tokens=False)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def public_hash(public):
    return hashlib.sha256(json.dumps(public, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def summarize(rows, minimum_success=14):
    arms = {a: {r['mission']: r for r in rows if r['arm'] == a}
            for a in ('base', 'ordinary')}
    a, b = arms['base'], arms['ordinary']
    if not a or a.keys() != b.keys() or len(rows) != 2*len(a):
        raise ValueError('Incomplete or duplicate paired evaluation')
    if any(a[i]['start_sha256'] != b[i]['start_sha256'] for i in a):
        raise ValueError('Initial states differ across arms')
    successes = {arm: sum(bool(r['success']) for r in bank.values())
                 for arm, bank in arms.items()}
    return dict(missions_per_arm=len(a), successes=successes,
                corrections=sum(not a[i]['success'] and b[i]['success'] for i in a),
                regressions=sum(a[i]['success'] and not b[i]['success'] for i in a),
                parent_ready=successes['ordinary'] >= minimum_success,
                minimum_success=minimum_success, no_adaptation_test=True)
