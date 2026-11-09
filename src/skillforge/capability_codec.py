"""Task-neutral serialization and direct native autoregressive generation."""
def prompt(row):
    size=row['size']
    if size not in (4,6) or len(row['input'])!=size*size or any(type(x)!=int or not 0<=x<=9 for x in row['input']):raise ValueError('grid codec')
    rule='Copy every cell unchanged.' if row['task']=='copy' else ('Move the connected monochrome object (color1 through8) exactly one cell toward marker9. '
        'Marker9 is outside the object bounding box and aligned with its rows or columns, indicating one cardinal direction. '
        'Preserve shape and color, keep marker9 fixed, and leave background0 elsewhere.')
    return f'Grid: {size} rows and {size} columns. {rule} Return exactly {size*size} digits in row-major order, no spaces or explanation.\nInput: '+''.join(map(str,row['input']))+'\nOutput:'


def digits(tok):
    result=[tok.encode(str(i),add_special_tokens=False) for i in range(10)]
    if any(len(x)!=1 for x in result):raise ValueError('single digit codec')
    return [x[0] for x in result]


def encode(tok,row):
    prefix=tok.encode(prompt(row),add_special_tokens=False);legal=digits(tok);suffix=[legal[x] for x in row['target']]
    if len(prefix)+len(suffix)>512:raise ValueError('sequence limit')
    return prefix+suffix,len(prefix)


def generate(study,row,cache=True):
    import torch
    legal=digits(study.tokenizer);lookup={t:i for i,t in enumerate(legal)}
    prefix=study.tokenizer.encode(prompt(row),add_special_tokens=False);tokens=[];past=None
    with torch.inference_mode():
        for _ in range(row['size']**2):
            ids=prefix+tokens if past is None else [tokens[-1]]
            output=study.model(input_ids=torch.tensor([ids],device='cuda'),past_key_values=past,use_cache=cache)
            logits=output.logits[0,-1].float()
            if not torch.isfinite(logits).all():raise ValueError('nonfinite generation')
            tokens.append(int(logits.argmax()));past=output.past_key_values if cache else None
    return dict(tokens=tokens,output=[lookup[t] for t in tokens] if all(t in lookup for t in tokens) else None,
                decoded=study.tokenizer.decode(tokens))


def evaluate(study,rows,check=lambda:None):
    records=[]
    for row in rows:
        check();answer=generate(study,row);record=dict(answer,id=row['id'],exact=answer['output']==row['target'])
        if row.get('second_target') is not None:
            if answer['output'] is None:record.update(second=None,exact=False)
            else:
                # Generic copy of generated cells; no movement calculation or correction.
                second=generate(study,dict(row,input=answer['output']))
                record.update(second=second,exact=second['output']==row['second_target'] and answer['output']==row['target'])
        records.append(record)
    return dict(n=len(records),exact=sum(r['exact'] for r in records),records=records)
