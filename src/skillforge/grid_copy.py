"""Copy-only foundation preparation. No movement labels or solver in training."""
import random
from pathlib import Path
import os
from .grid_io import render_prompt, palette_ids


def make_data(excluded, seed=193):
    rng=random.Random(seed);seen={tuple(x) for x in excluded};result={}
    for split,n in [('train',4096),('heldout',256)]:
        rows=[]
        while len(rows)<n:
            sparse=len(rows)%2==0
            if sparse:
                grid=[0]*16
                for i in rng.sample(range(16),rng.randint(2,6)):grid[i]=rng.randint(1,3)
            else:grid=[rng.randrange(4) for _ in range(16)]
            if tuple(grid) in seen:continue
            seen.add(tuple(grid));rows.append(dict(input=grid,kind='sparse' if sparse else 'dense'))
        result[split]=rows
    return result


def encode_copy(tokenizer,grid,demos):
    prefix=tokenizer.encode(render_prompt(grid,'copy',demos),add_special_tokens=False)
    suffix=[palette_ids(tokenizer)[x] for x in grid]
    if not prefix or len(prefix)+16>768:raise ValueError('copy sequence budget')
    return prefix+suffix,len(prefix)


def batch_tensors(encoded,pad,device):
    import torch
    size=max(len(x) for x,_ in encoded)
    ids=torch.full((len(encoded),size),pad,dtype=torch.long,device=device)
    mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
    for j,(tokens,start) in enumerate(encoded):
        ids[j,:len(tokens)]=torch.tensor(tokens,device=device);mask[j,:len(tokens)]=1
        labels[j,start:len(tokens)]=ids[j,start:len(tokens)]
    return ids,mask,labels


def copy_loss(model,batch):
    import torch
    ids,mask,labels=batch
    logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
    # Only the 16 answer positions contribute, including the first digit.
    active=labels[:,1:]!=-100
    return torch.nn.functional.cross_entropy(logits[:,:-1][active].float(),labels[:,1:][active])


def checkpoint(path,model,optimizer,step,source,order,rng):
    import torch
    path=Path(path);temp=path.with_suffix('.tmp')
    try:
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                        manifest_sha256=source,order=order,data_rng=rng.getstate(),
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp)
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)
