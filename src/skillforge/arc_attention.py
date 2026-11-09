"""Full-answer ARC scoring with one intervention at the question boundary.

Teacher-forced answer tokens never control routing: only the final prefix query
is edited, under the model's causal mask. Later positions receive any effect
through the ordinary downstream computation. No repeated answer-token injection.
"""
import numpy as np
from .arc_eval import encode_choices, validate_row
from .attention_probe import intervention


def score_boundary(study, row, bank=None, mode='off', device='cuda'):
    import torch
    encoded = encode_choices(study.tokenizer, row)
    starts = {start for _, start in encoded}
    if len(starts) != 1: raise ValueError('choice prefixes differ')
    start = next(iter(starts))
    prefix = encoded[0][0][:start]
    if any(ids[:n] != prefix for ids, n in encoded):
        raise ValueError('choice prefixes differ')
    totals, counts, all_masses = [], [], []
    for tokens, start in encoded:
        ids = torch.tensor([tokens], device=device)
        masses = []
        with torch.inference_mode(), intervention(bank, mode, masses=masses, query_index=start-1):
            logits = study.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0].float()
            lp = torch.log_softmax(logits[start-1:len(tokens)-1], -1)
            selected = lp.gather(-1, ids[0, start:, None]).squeeze(-1)
            totals.append(float(selected.sum()))
        counts.append(len(tokens)-start)
        all_masses.append(masses)
    if not np.isfinite(totals).all(): raise ValueError('nonfinite scores')
    normalized = [s/n for s,n in zip(totals, counts)]
    gold = validate_row(row)
    pred, raw = int(np.argmax(normalized)), int(np.argmax(totals))
    return dict(id=row['id'], gold=gold, prediction=pred, raw_prediction=raw,
                correct=pred==gold, raw_correct=raw==gold,
                answer_logprobs=totals, answer_token_counts=counts,
                normalized_logprobs=normalized, virtual_mass_by_choice=all_masses)
