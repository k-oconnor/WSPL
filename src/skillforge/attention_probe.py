"""Native, frozen single-slot attention intervention. No author code in inference."""
from contextlib import contextmanager
import numpy as np

LAYERS = tuple(range(24, 30))
EDITABLE = {'K', 'V', 'alpha'}


def validate(bank, reference, stage, frozen=None):
    if set(bank) != set(reference):
        raise ValueError('tensor names differ')
    for name, old in reference.items():
        a = bank[name]
        if a.shape != old.shape or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError('shape/dtype/finite: ' + name)
        if name not in EDITABLE and a.tobytes() != old.tobytes():
            raise ValueError('read-only tensor: ' + name)
    alpha = bank['alpha']
    if np.any(alpha < 0) or np.any(alpha > 1) or np.count_nonzero(alpha) > 1:
        raise ValueError('at most one layer/KV group; alpha in [0,1]')
    for name in ('K', 'V'):
        norms = np.linalg.norm(bank[name].astype(np.float64), axis=-1)
        if np.any(norms > 10 * reference[name + '_scale'] + 1e-5):
            raise ValueError(name + ' norm exceeds ten native mean norms')
    if stage == 'forced':
        if np.any(bank['K']): raise ValueError('keys stay zero during forced delivery')
    elif stage == 'retrieval':
        if frozen is None: raise ValueError('frozen forced artifact required')
        for name in ('V', 'alpha'):
            if bank[name].tobytes() != frozen[name].tobytes():
                raise ValueError('retrieval may edit K only')
    else: raise ValueError('unknown stage')


@contextmanager
def intervention(bank=None, mode='off', capture=None, masses=None, query_index=-1):
    """Single-threaded eager Llama attention hook at one declared query position.

    K is in post-RoPE coordinates (fixed virtual coordinate zero). Native positions
    are untouched. GQA grouping is preserved: one slot per KV group, shared by its
    three query heads. Disabled calls delegate to the original function exactly.
    """
    import torch
    import transformers.models.llama.modeling_llama as llama
    original = llama.eager_attention_forward
    tensors = None if bank is None else {k: torch.from_numpy(bank[k]).to('cuda') for k in EDITABLE}

    def wrapped(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        output, weights = original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        if module.layer_idx not in LAYERS: return output, weights
        idx = LAYERS.index(module.layer_idx)
        position = query_index if query_index >= 0 else query.shape[-2] + query_index
        if not 0 <= position < query.shape[-2]: raise ValueError('query position outside sequence')
        if capture is not None:
            for name, a in [('Q', query[0, :, position]), ('native_K', key[0]), ('native_V', value[0]), ('attention', weights[0, :, position])]:
                capture.setdefault(name, []).append(a.detach().cpu().float().numpy())
        if mode == 'off' or tensors is None or not torch.count_nonzero(tensors['alpha'][idx]): return output, weights
        if dropout or module.training: raise ValueError('inference only')
        group = int(torch.nonzero(tensors['alpha'][idx]).item())
        first = group * module.num_key_value_groups
        heads = slice(first, first + module.num_key_value_groups)
        v = tensors['V'][idx, group]
        if mode == 'forced':
            mass = tensors['alpha'][idx, group].expand(query.shape[0], module.num_key_value_groups)
        elif mode in ('retrieval', 'zero_key'):
            k = tensors['K'][idx, group] if mode == 'retrieval' else torch.zeros_like(v)
            native_keys = key[:, group]
            q = query[:, heads, position]
            scores = torch.matmul(q, native_keys.transpose(-1, -2)) * scaling
            if attention_mask is not None:
                mask = attention_mask[..., position, :]
                if mask.shape[1] != 1: mask = mask[:, heads]
                scores = scores + mask
            virtual = torch.sum(q * k, dim=-1) * scaling
            mass = torch.sigmoid(virtual - torch.logsumexp(scores, dim=-1))
        else: raise ValueError('unknown intervention mode')
        output = output.clone()
        old = output[:, position, heads]
        output[:, position, heads] = (1 - mass[..., None]) * old + mass[..., None] * v
        if masses is not None: masses.append(dict(layer=module.layer_idx, kv_group=group, mass=mass.detach().cpu().tolist()))
        # weights describe original native attention; injected mass is separately logged.
        return output, weights
    llama.eager_attention_forward = wrapped
    try: yield
    finally: llama.eager_attention_forward = original


def prefix_score(study, row, bank=None, mode='off', capture=None):
    import torch
    from .arc_eval import encode_choices, validate_row
    encoded = encode_choices(study.tokenizer, row)
    prefix = encoded[0][0][:encoded[0][1]]
    suffixes = [ids[start:] for ids, start in encoded]
    if any(len(s) != 1 for s in suffixes): raise ValueError('single-token choices required')
    ids = torch.tensor([prefix], device='cuda'); masses = []
    with torch.inference_mode(), intervention(bank, mode, capture, masses):
        logits = study.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
    values = [float(lp[s[0]]) for s in suffixes]
    if not np.isfinite(values).all(): raise ValueError('nonfinite scores')
    gold = validate_row(row); pred = int(np.argmax(values))
    return dict(id=row['id'],gold=gold,prediction=pred,correct=pred==gold,logprobs=values,
                margin=values[gold]-max(v for i,v in enumerate(values) if i!=gold),virtual_mass=masses)


def make_bank(study, row):
    capture = {}; score = prefix_score(study, row, capture=capture)
    bank = {k:np.stack(v).astype(np.float32) for k,v in capture.items()}
    bank.update(K=np.zeros((6,3,64),np.float32), V=np.zeros((6,3,64),np.float32), alpha=np.zeros((6,3),np.float32))
    for name in ('K','V'):
        bank[name+'_scale'] = np.maximum(np.linalg.norm(bank['native_'+name].astype(np.float64),axis=-1).mean(axis=-1),1e-6).astype(np.float32)
    bank['O'] = np.stack([study.model.model.layers[i].self_attn.o_proj.weight.detach().cpu().numpy() for i in LAYERS]).astype(np.float32)
    options = [study.tokenizer.encode(' '+s,add_special_tokens=False)[0] for s in row['choices']['text']]
    bank['answer_embeddings'] = study.model.get_output_embeddings().weight[options].detach().cpu().float().numpy()
    bank['final_norm'] = study.model.model.norm.weight.detach().cpu().float().numpy()
    return bank, score


def author_prompt(guide, target, stage, history, step):
    import json
    text = guide + '\nTARGET:\n' + json.dumps(target) + '\nSTAGE: ' + stage
    text += '\nVISIBLE TRANSCRIPT:\n' + json.dumps(history)
    text += f'\nResponse {step+1}/6. Return one fenced python program or exactly FINAL.'
    if len(text.encode())>800000: raise ValueError('prompt budget exceeded')
    return text
