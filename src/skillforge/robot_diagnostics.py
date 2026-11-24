"""Acquisition-only last-block diagnostics, shared equally across author arms."""
import numpy as np
import torch
from .robot_codec import CODES


def features(model, tokenizer, rows, guard, report):
    if len(rows) > 512:
        raise ValueError('At most 512 acquisition decisions')
    layer = model.config.num_hidden_layers-1
    name = f'model.layers.{layer}.mlp.down_proj'
    captured = {}
    handles = []
    def hook(key):
        def capture(module, args):
            captured[key] = args[0][:, -1].detach().float().cpu().numpy().copy()[0]
        return capture
    handles.append(model.get_submodule(name).register_forward_pre_hook(hook('X')))
    handles.append(model.model.norm.register_forward_pre_hook(hook('R')))
    code_ids = [tokenizer(c, add_special_tokens=False)['input_ids'] for c in CODES]
    if any(len(x) != 1 for x in code_ids):
        raise ValueError('Single-token action contract')
    code_ids = [x[0] for x in code_ids]
    collected = {k: [] for k in ('X', 'R', 'L', 'Y')}
    metadata = []
    model.eval()
    model.gradient_checkpointing_disable()
    try:
        for i, row in enumerate(rows):
            guard(); report(diagnostic_row=i, diagnostic_rows=len(rows))
            ids = torch.tensor([row['record']['ids']], device='cuda')
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    use_cache=False, logits_to_keep=1).logits[0, -1].float()
            for key in ('X', 'R'):
                collected[key].append(captured[key])
            collected['L'].append(logits[code_ids].cpu().numpy())
            y = np.zeros(len(CODES), np.float32)
            y[CODES.index(row['code'])] = 1
            collected['Y'].append(y)
            metadata.append(dict(row=i, condition=row['condition'], prompt=row['prompt'],
                label=row['code'], parent_token=int(logits.argmax()),
                parent_code=tokenizer.decode([int(logits.argmax())], skip_special_tokens=False)))
    finally:
        for handle in handles:
            handle.remove()
    bank = {k: np.stack(v).astype(np.float32) for k, v in collected.items()}
    bank['W'] = model.lm_head.weight[code_ids].detach().float().cpu().numpy().copy()
    bank['N'] = model.model.norm.weight.detach().float().cpu().numpy().copy()
    return bank, metadata, name
