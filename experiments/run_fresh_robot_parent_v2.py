"""Source-gated fresh ordinary parent: collect, fit, freeze, paired readiness.

No Astra calls, adaptation fitting, author study, or automatic follow-on.
"""
import json
import math
import random
import time
from pathlib import Path

import torch

from skillforge.fresh_simulator import Simulator
from skillforge.authoring_sft import adapter_state, checkpoint, frozen_hash, restore_adapter
from skillforge.executive_teaching import ordinary_action
from skillforge.fresh_robot import (panel, validate_panels, prompt_record, token_loss,
                                   predict, sha, public_hash, summarize)
from skillforge.robot_codec import render, encode_action, decode_action
from skillforge.reliable_json import save, status
from skillforge.teaching_training import state_hash


def run(constructor, output_dir, config):
    out = Path(output_dir)
    started = time.monotonic()
    identity = json.loads((out/'source_receipt.json').read_text())['manifest_sha256']
    sim = optimizer = model = None
    step = targets = 0
    base_hash = None
    live = {}

    def report(**values):
        live.update(values, elapsed_seconds=time.monotonic()-started)
        status(out/'status.json', live)

    def guard():
        control = out/'control.json'
        if time.monotonic()-started > config['wall_guard_seconds'] or (
                control.exists() and json.loads(control.read_text()).get('stop')):
            raise InterruptedError('Study stop/guard; no automatic retry')

    def preserve():
        checkpoint(out/'rolling.pt', model, optimizer,
                   dict(step=step, targets=targets, base_hash=base_hash,
                        manifest_sha256=identity, purpose='fresh_ordinary_parent'))

    def begin(label, mission):
        nonlocal sim
        guard()
        if sim is not None:
            raise RuntimeError("Previous simulator was not closed")
        rpc_out = out/(label+".rpc")
        rpc_out.mkdir()
        sim = Simulator(config, rpc_out, out/"approved_manifest.json", identity)
        return sim.call(dict(op='start', artifact=config['motor_artifact'],
                             output=str(out/label), **mission))

    def finish(label):
        nonlocal sim
        worker, sim = sim, None
        try:
            return worker.call(dict(op='finish', trace_path=str(out/(label+'.trace.json.gz'))))
        finally:
            worker.close()

    try:
        report(status='verifying_resources')
        for path, expected in config['resources'].items():
            guard()
            if sha(path) != expected:
                raise ValueError('Bound resource changed: '+path)
        training = panel(config['train_seed'], config['train_missions'])
        reserved = panel(config['reserved_seed'], config['reserved_missions'])
        validate_panels(training, reserved)
        study = constructor()
        model, tokenizer = study.model, study.tokenizer
        base_hash = frozen_hash(model)
        initial = adapter_state(model)
        rows, episodes = [], []
        with (out/'training_data.jsonl').open('x', encoding='utf-8') as data_log:
            for i, mission in enumerate(training):
                report(status='collecting_ordinary_teacher', mission=i, total=len(training))
                label = f'train_{i:03d}'
                public = begin(label, mission)
                pending = []
                while not public['finished']:
                    guard()
                    action = ordinary_action(public)
                    prompt, code = render(public), encode_action(public, action)
                    pending.append(dict(mission=i, prompt=prompt, code=code,
                                        record=prompt_record(tokenizer, prompt, code)))
                    public = sim.call(dict(op='act', action=json.dumps(action)))
                result = finish(label)
                episodes.append(dict(mission=i, **mission, **result))
                save(out/'teacher_episodes.json', episodes)
                if not result['success']:
                    raise ValueError('Supplied ordinary teacher failed; stop before training')
                for row in pending:
                    data_log.write(json.dumps(row)+'\n')
                data_log.flush()
                rows.extend(pending)
        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=config['learning_rate'],
                                     weight_decay=.01, foreach=False)
        rng = random.Random(config['schedule_seed'])
        schedule = [[rng.randrange(len(rows)) for _ in range(config['batch_size'])]
                    for _ in range(config['updates'])]
        save(out/'schedule.json', schedule)
        last_save = time.monotonic()
        with (out/'training.jsonl').open('x') as log:
            for batch in schedule:
                guard()
                optimizer.zero_grad(set_to_none=True)
                value = 0.
                for index in batch:
                    guard()
                    loss = token_loss(model, rows[index]['record'])
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss')
                    value += float(loss.detach())
                    (loss/len(batch)).backward()
                    del loss
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
                progress = max(0., (step+1-16)/max(1, config['updates']-16))
                lr = config['learning_rate']*min(1., (step+1)/16)*(
                    .1+.9*.5*(1+math.cos(math.pi*progress)))
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.step()
                step += 1
                targets += len(batch)
                item = dict(step=step, nll=value/len(batch), lr=lr,
                            grad_norm=float(norm), targets=targets)
                log.write(json.dumps(item)+'\n')
                log.flush()
                report(status='training', total=config['updates'], **item)
                if time.monotonic()-last_save >= 1800:
                    preserve()
                    last_save = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        if frozen_hash(model) != base_hash:
            raise ValueError('Frozen base changed')
        preserve()
        final = adapter_state(model)
        torch.save(dict(adapter=final, base_hash=base_hash, manifest_sha256=identity,
                        adapter_sha256=state_hash(final), step=step, targets=targets,
                        origin='fresh_ordinary_teacher'), out/'adapter_final.pt')
        save(out/'frozen.json', dict(manifest_sha256=identity, base_hash=base_hash,
             adapter_sha256=sha(out/'adapter_final.pt'), factor_sha256=state_hash(final),
             training_data_sha256=sha(out/'training_data.jsonl'),
             reserved_evaluation_started=False))
        # Both conditions freeze before opening any reserved physical mission.
        model.eval()
        model.gradient_checkpointing_disable()
        results = []
        with (out/'predictions.jsonl').open('x', encoding='utf-8') as predictions:
            for arm, bank in [('base', initial), ('ordinary', final)]:
                restore_adapter(model, bank)
                for i, mission in enumerate(reserved):
                    report(status='reserved_readiness', arm=arm, mission=i, total=len(reserved))
                    label = f'{arm}_{i:03d}'
                    public = begin(label, mission)
                    start_id = public_hash(public)
                    calls = invalid = 0
                    while not public['finished']:
                        guard()
                        prompt = render(public)
                        token, code = predict(model, tokenizer, prompt_record(tokenizer, prompt))
                        try:
                            action = decode_action(public, code)
                        except ValueError:
                            # Invalid output consumes a decision; no automatic repair.
                            action = {'op': 'invalid_model_token', 'token': token}
                            invalid += 1
                        predictions.write(json.dumps(dict(arm=arm, mission=i, decision=calls,
                            prompt=prompt, token=token, code=code, action=action))+'\n')
                        predictions.flush()
                        calls += 1
                        public = sim.call(dict(op='act', action=json.dumps(action)))
                    results.append(dict(arm=arm, mission=i, **mission, start_sha256=start_id,
                                        model_calls=calls, invalid_tokens=invalid, **finish(label)))
                    save(out/'episodes.json', results)
        restore_adapter(model, final)
        if frozen_hash(model) != base_hash or state_hash(adapter_state(model)) != state_hash(final):
            raise ValueError('Final restoration/base identity failed')
        result = dict(status='complete', manifest_sha256=identity, author_calls=0,
                      updates=step, targets=targets, training_missions=len(training),
                      summary=summarize(results, config['minimum_success']),
                      elapsed_seconds=time.monotonic()-started, follow_on_launched=False)
        save(out/'results.json', result)
        report(status='complete', results=result)
    except BaseException as error:
        report(status='failed', error=repr(error), step=step, targets=targets)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            preserve()
        raise
    finally:
        if sim is not None:
            sim.close()


if __name__ == '__main__':
    from source_consistency import require_registered_runner
    require_registered_runner(__file__)
