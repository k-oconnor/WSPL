"""Synthetic-token Qwen wiring/backward check; no task scoring or optimizer."""
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from skillforge.authoring_sft import AuthoringSFTStudy, frozen_hash
from skillforge.fresh_robot import token_loss, prompt_record
from skillforge.robot_codec import CODES
from skillforge.wspl_lora import (model_state, identity, describe, source_for_pair,
                                 construct_wspl, construct_native, install_state)


def main():
    started = time.monotonic()
    study = AuthoringSFTStudy(sys.argv[1])
    model, tokenizer = study.model, study.tokenizer
    base_hash = frozen_hash(model)
    state = model_state(model)
    parent_id = identity(state)
    # Cataloging all pairs checks the real topology against the bridge limits.
    catalog = describe(state)
    assert catalog['element_count'] == 18_464_768
    p = 'model.layers.0.self_attn.q_proj'
    source = source_for_pair(catalog['projections'][p]['pair_id'], 'B[0,0] = 1/64;')
    edited, receipt = construct_wspl(state, {p: source}, expected_parent_id=parent_id)
    replacement = {f: state[p+'.lora_'+f].copy() for f in ('A', 'B')}
    replacement['B'][0, 0] = np.float32(1/64)
    native, _ = construct_native(state, {p: replacement}, expected_parent_id=parent_id)
    assert identity(native) == identity(edited)
    synthetic = dict(ids=[100, 101, 102, 103]*512, target=104)
    model.eval()
    with torch.no_grad():
        before = float(token_loss(model, synthetic))
    install_state(model, edited, expected_current_id=parent_id)
    model.train()
    loss = token_loss(model, synthetic)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    model.zero_grad(set_to_none=True)
    install_state(model, state, expected_current_id=receipt['candidate_id'])
    model.eval()
    with torch.no_grad():
        after = float(token_loss(model, synthetic))
    assert before == after
    assert frozen_hash(model) == base_hash and identity(model_state(model)) == parent_id
    for code in CODES:
        prompt_record(tokenizer, 'Synthetic interface validation.', code)
    report = dict(status='passed', engineering_only=True, model_revision=json.loads(
        (Path(sys.argv[1])/'download_provenance.json').read_text())['revision'],
        projections=len(catalog['projections']), adapter_elements=catalog['element_count'],
        factor_id=parent_id, base_hash=base_hash, synthetic_backward=True,
        exact_restoration=True, frozen_base=True, task_predictions=0, optimizer_updates=0,
        synthetic_context_tokens=len(synthetic['ids']),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/wspl/fresh_robot_student_preflight.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
