"""Approved-source runner for compiler-supervised neural controller authoring."""
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from skillforge.authoring_sft import (adapter_state, checkpoint, encode, frozen_hash,
                                     loss_sum, restore_adapter)
from skillforge.controller_author import (KINDS, check_block, construct, dataset,
                                         rollout, teacher)
from skillforge.reliable_json import save, status


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(constructor, output_dir, config):
    out = Path(output_dir)
    started = time.monotonic()
    state = dict(status='resource_verification', step=0)

    def update(**values):
        state.update(values, elapsed_seconds=time.monotonic() - started)
        status(out / 'status.json', state)

    def guard():
        control = out / 'control.json'
        if time.monotonic() - started > config['wall_guard_seconds']:
            raise InterruptedError('Study wall guard reached')
        if control.exists() and json.loads(control.read_text()).get('stop'):
            raise InterruptedError('User requested stop')

    for path, digest in config['resources'].items():
        if file_hash(path) != digest:
            raise ValueError('Resource identity differs: ' + path)
    splits = dataset()
    for name, rows in splits.items():
        with (out / (name + '.jsonl')).open('x', encoding='utf8') as stream:
            for row in rows:
                guard()
                artifact = construct(row['messages'][1]['content'], row['spec'])
                if not check_block(artifact, row['spec'])['passes']:
                    raise ValueError('Compiler teacher failed independent numerical check')
                stream.write(json.dumps(row) + '\n')
    save(out / 'corpus_identity.json', {name: dict(rows=len(rows), sha256=file_hash(out / (name + '.jsonl')))
                                      for name, rows in splits.items()})
    reference_missions = []
    reserved_by_kind = {kind: [row for row in splits['test'] if row['spec']['kind'] == kind] for kind in KINDS}
    for index in range(16):
        specs = {kind: reserved_by_kind[kind][index]['spec'] for kind in KINDS}
        blocks = {kind: construct(teacher(spec), spec) for kind, spec in specs.items()}
        for via in [False, True]:
            reference = rollout(blocks, specs, via,
                initial=(-.7, -.6 if index % 2 == 0 else .6),
                target=(.6, .55 if index % 2 == 0 else -.55))
            if not reference['success']:
                raise ValueError('Reference controller cannot execute a reserved mission')
            reference_missions.append(dict(index=index, checkpoint=via, **reference))
    save(out / 'reference_missions.json', reference_missions)
    update(status='loading_model')
    study = constructor()
    model, tokenizer = study.model, study.tokenizer
    base_hash = frozen_hash(model)
    initial = adapter_state(model)
    train = [encode(tokenizer, row['messages'], config['context']) for row in splits['train']]
    development = [encode(tokenizer, row['messages'], config['context']) for row in splits['development']]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['lr'], weight_decay=.01, foreach=False)
    step, targets, last_save = 0, 0, time.monotonic()
    optimizer_in_progress = False
    training_finished = False
    validation = []
    manifest = json.loads((out / 'source_receipt.json').read_text())['manifest_sha256']

    def save_checkpoint():
        checkpoint(out / 'rolling.pt', model, optimizer,
                   dict(step=step, targets=targets, base_hash=base_hash,
                        manifest_sha256=manifest, config=config))

    def validate():
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        model.eval()
        total, count = 0., 0
        with torch.no_grad():
            for i, record in enumerate(development):
                guard()
                total += float(loss_sum(model, record))
                count += record['targets']
                if i % 16 == 0:
                    update(status='validation', completed=i, step=step)
        validation.append(dict(step=step, nll=total / count, targets=count))
        save(out / 'validation.json', validation)
        model.train()
        gc.collect()
        torch.cuda.empty_cache()

    try:
        validate()
        total_steps = config['epochs'] * math.ceil(len(train) / config['accumulation'])
        with (out / 'training.jsonl').open('x', encoding='utf8') as log:
            for epoch in range(config['epochs']):
                order = list(range(len(train)))
                random.Random(config['order_seed'] + epoch).shuffle(order)
                for offset in range(0, len(order), config['accumulation']):
                    guard()
                    group = [train[i] for i in order[offset:offset + config['accumulation']]]
                    count = sum(record['targets'] for record in group)
                    optimizer.zero_grad(set_to_none=True)
                    total = 0.
                    for record in group:
                        loss = loss_sum(model, record)
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite training loss')
                        (loss / count).backward()
                        total += float(loss.detach())
                        del loss
                    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                         1., error_if_nonfinite=True)
                    fraction = max(0., (step + 1 - config['warmup']) / max(1, total_steps - config['warmup']))
                    lr = config['lr'] * min(1., (step + 1) / config['warmup']) * (.1 + .9 * .5 * (1 + math.cos(math.pi * fraction)))
                    for group_config in optimizer.param_groups:
                        group_config['lr'] = lr
                    optimizer_in_progress = True
                    optimizer.step()
                    optimizer_in_progress = False
                    step += 1
                    targets += count
                    optimizer.zero_grad(set_to_none=True)
                    update(status='training', step=step, total_steps=total_steps, epoch=epoch + 1,
                           training_nll=total / count, grad_norm=float(norm), lr=lr, targets=targets)
                    log.write(json.dumps(state) + '\n')
                    log.flush()
                    if time.monotonic() - last_save >= 1800:
                        save_checkpoint()
                        last_save = time.monotonic()
                save_checkpoint()
                last_save = time.monotonic()
                validate()
        if frozen_hash(model) != base_hash:
            raise ValueError('Frozen base parameters changed')
        save_checkpoint()
        final = adapter_state(model)
        torch.save(dict(adapter=final, base_hash=base_hash, manifest_sha256=manifest,
                        projections=study.projections, step=step), out / 'adapter_final.pt')
        training_finished = True
        model.gradient_checkpointing_disable()
        model.eval()
        # Both weights are fixed before any reserved generation is made.
        replies = []
        with torch.inference_mode():
            for arm, weights in [('trained', final), ('base', initial)]:
                restore_adapter(model, weights)
                for i, row in enumerate(splits['test']):
                    guard()
                    update(status='generation', arm=arm, completed=i, total=len(splits['test']))
                    prompt = tokenizer.apply_chat_template(row['messages'][:1], tokenize=False,
                                                           add_generation_prompt=True)
                    ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to('cuda')
                    generation_started = time.monotonic()
                    generated = model.generate(ids, do_sample=False, max_new_tokens=768, use_cache=True,
                                               pad_token_id=tokenizer.eos_token_id,
                                               attention_mask=torch.ones_like(ids))
                    tokens = generated[0, ids.shape[1]:].tolist()
                    reply = dict(arm=arm, id=row['id'], spec=row['spec'], tokens=tokens,
                                 source=tokenizer.decode(tokens, skip_special_tokens=True), hit_cap=len(tokens) == 768,
                                 generation_seconds=time.monotonic() - generation_started)
                    replies.append(reply)
                    with (out / 'replies.jsonl').open('a', encoding='utf8') as stream:
                        stream.write(json.dumps(reply) + '\n')
        update(status='evaluating_frozen_programs')
        scores, artifacts = [], {}
        for reply in replies:
            record = dict(arm=reply['arm'], id=reply['id'], spec=reply['spec'], constructed=False)
            try:
                artifact = construct(reply['source'], reply['spec'])
                record.update(constructed=True, parameters=artifact.to_dict()['allocation']['parameters'],
                              **check_block(artifact, reply['spec']))
                artifacts[reply['arm'], reply['id']] = artifact
                artifact.save(out / (reply['arm'] + '_' + reply['id'][:16] + '.weights.json'))
            except ValueError as error:
                record.update(passes=False, error=str(error))
            scores.append(record)
        save(out / 'block_scores.json', scores)
        missions = []
        by_kind = {kind: [row for row in splits['test'] if row['spec']['kind'] == kind] for kind in KINDS}
        for arm in ['base', 'trained']:
            for index in range(16):
                selected = {kind: by_kind[kind][index] for kind in KINDS}
                specs = {kind: row['spec'] for kind, row in selected.items()}
                for via in [False, True]:
                    guard()
                    record = dict(arm=arm, index=index, checkpoint=via,
                                  block_ids={kind: row['id'] for kind, row in selected.items()})
                    try:
                        blocks = {kind: artifacts[arm, row['id']] for kind, row in selected.items()}
                        record.update(rollout(blocks, specs, via,
                            initial=(-.7, -.6 if index % 2 == 0 else .6),
                            target=(.6, .55 if index % 2 == 0 else -.55)))
                    except (KeyError, ValueError) as error:
                        record.update(success=False, status='invalid_program', error=str(error))
                    missions.append(record)
        save(out / 'missions.json', missions)
        if frozen_hash(model) != base_hash:
            raise ValueError('Frozen base changed during evaluation')
        results = dict(status='complete', base_hash=base_hash, manifest_sha256=manifest,
                       trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                       validation=validation, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       arms={arm: dict(blocks=96, generation_seconds=sum(r['generation_seconds'] for r in replies if r['arm'] == arm),
                            generated_tokens=sum(len(r['tokens']) for r in replies if r['arm'] == arm),
                            source_bytes=sum(len(r['source'].encode()) for r in replies if r['arm'] == arm), constructed=sum(s['constructed'] for s in scores if s['arm'] == arm),
                            behavior_correct=sum(s['passes'] for s in scores if s['arm'] == arm),
                            missions=32, successful_missions=sum(m['success'] for m in missions if m['arm'] == arm))
                             for arm in ['base', 'trained']})
        save(out / 'results.json', results)
        update(status='complete', results=results)
    except BaseException as error:
        if not optimizer_in_progress and not training_finished:
            optimizer.zero_grad(set_to_none=True)
            save_checkpoint()
        update(status='paused' if isinstance(error, InterruptedError) else 'failed', error=repr(error))
        raise
