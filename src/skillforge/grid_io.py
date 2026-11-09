"""Only token/position serialization and ordinary frozen-model generation."""
import numpy as np


def render_prompt(grid,task,demos):
    # Import only the serializer, never the domain's movement computation.
    from .grid_domain import encode_grid
    if task not in ('copy','movement'):raise ValueError('task')
    rule=('Copy every input cell unchanged.' if task=='copy' else
          'Move the connected object (color 1 or 2) one cell toward marker 3. The marker is outside the object bounding box and aligned with its rows or columns, indicating exactly one of up, down, left or right. Preserve object shape and color, keep marker 3 fixed, and leave background 0 elsewhere.')
    text='A grid has 4 rows and 4 columns. Encode it as exactly 16 digits in row-major order, with no separators. Colors are 0, 1, 2, 3. '+rule+'\n'
    for d in demos:
        target=d['input'] if task=='copy' else d['output']
        text+='Input: '+encode_grid(d['input'])+'\nOutput: '+encode_grid(target)+'\n'
    return text+'Input: '+encode_grid(grid)+'\nOutput:'


def palette_ids(tokenizer):
    ids=[]
    for c in range(4):
        tokens=tokenizer.encode(str(c),add_special_tokens=False)
        if len(tokens)!=1 or tokenizer.decode(tokens)!=str(c):raise ValueError('single digit tokenizer contract')
        ids.append(tokens[0])
    if len(set(ids))!=4:raise ValueError('palette collision')
    return ids


def generate(study,grid,task,demos,mode):
    import torch
    if mode not in ('unconstrained','palette_only'):raise ValueError('mode')
    legal=palette_ids(study.tokenizer);lookup={token:i for i,token in enumerate(legal)}
    prefix=study.tokenizer.encode(render_prompt(grid,task,demos),add_special_tokens=False)
    if len(prefix)>768:raise ValueError('prefix budget')
    tokens=prefix.copy();generated=[];native_legal=[]
    with torch.inference_mode():
        for position in range(16):
            ids=torch.tensor([tokens],device='cuda')
            logits=study.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1].float()
            if not torch.isfinite(logits).all():raise ValueError('nonfinite logits')
            native=int(logits.argmax());native_legal.append(native in lookup)
            chosen=native if mode=='unconstrained' else legal[int(logits[legal].argmax())]
            generated.append(chosen);tokens.append(chosen)
    valid=all(t in lookup for t in generated)
    return dict(tokens=generated,decoded=study.tokenizer.decode(generated),format_valid=valid,
                output=[lookup[t] for t in generated] if valid else None,native_argmax_legal=native_legal)


def summarize(records):
    return dict(n=len(records),exact=sum(r['exact'] for r in records),valid_format=sum(r['format_valid'] for r in records),
                cell_accuracy=float(np.mean([r['correct_cells']/16 for r in records])),
                by_direction={d:dict(n=sum(r['direction']==d for r in records),exact=sum(r['exact'] for r in records if r['direction']==d)) for d in sorted({r['direction'] for r in records})})
