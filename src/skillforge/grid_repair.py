"""Native adapter repair contracts and acquisition-only diagnostics."""
import hashlib,random
import numpy as np
from .grid_domain import enumerate_domain,balanced_take
from .grid_movement import encode_task
from .grid_io import generate,palette_ids
from .relu_adapter import attached

EDITABLE=('read.weight','read.bias','write.weight','write.bias')


def larger_domain():
    shapes={'bar3h':((0,0),(0,1),(0,2)),'bar3v':((0,0),(1,0),(2,0)),'square':((0,0),(0,1),(1,0),(1,1))};rows=[]
    for name,shape in shapes.items():
        for top in range(4-max(r for r,c in shape)):
            for left in range(4-max(c for r,c in shape)):
                cells={(r+top,c+left) for r,c in shape};r0=min(r for r,c in cells);r1=max(r for r,c in cells);c0=min(c for r,c in cells);c1=max(c for r,c in cells)
                for mr in range(4):
                    for mc in range(4):
                        choices=[]
                        if c0<=mc<=c1 and mr<r0:choices.append(('up',-1,0))
                        if c0<=mc<=c1 and mr>r1:choices.append(('down',1,0))
                        if r0<=mr<=r1 and mc<c0:choices.append(('left',0,-1))
                        if r0<=mr<=r1 and mc>c1:choices.append(('right',0,1))
                        if len(choices)!=1:continue
                        direction,dr,dc=choices[0];moved={(r+dr,c+dc) for r,c in cells}
                        if (mr,mc) in moved or any(not(0<=r<4 and 0<=c<4) for r,c in moved):continue
                        for color in (1,2):
                            a=[0]*16;b=[0]*16;a[mr*4+mc]=b[mr*4+mc]=3
                            for r,c in cells:a[r*4+c]=color
                            for r,c in moved:b[r*4+c]=color
                            rows.append(dict(input=a,output=b,shape=name,direction=direction,color=color,id=hashlib.sha256(bytes(a)).hexdigest()))
    return rows


def prepare_data(previous,copy):
    seen={tuple(r['input']) for key,rows in previous.items() if key!='copy_train' for r in rows}
    seen_copy={tuple(r['input']) for rows in copy.values() for r in rows}
    fresh=[r for r in enumerate_domain() if r['shape'].startswith('L') and tuple(r['input']) not in seen|seen_copy]
    acquisition=[dict(r,task='movement') for r in balanced_take(previous['train'],64,401)]
    acquisition += [dict(r,task='copy') for r in copy['train'][:32]]
    reserved=dict(fresh_L=[dict(r,task='movement') for r in balanced_take(fresh,128,409)],
                  larger=[dict(r,task='movement') for r in balanced_take([r for r in larger_domain() if tuple(r['input']) not in seen|seen_copy],64,419)])
    # New copying cases are never in the copy foundation's pool or old evaluation.
    excluded=seen|{tuple(r['input']) for rows in copy.values() for r in rows}
    excluded|={tuple(r['input']) for rows in reserved.values() for r in rows}
    rng=random.Random(421);retention=[]
    while len(retention)<64:
        grid=[rng.randrange(4) for _ in range(16)]
        if tuple(grid) in excluded:continue
        excluded.add(tuple(grid));retention.append(dict(input=grid,task='copy'))
    reserved['copy']=retention
    return acquisition,reserved


def weights(adapter):return {k:v.detach().cpu().numpy().copy() for k,v in adapter.state_dict().items()}


def load_weights(adapter,bank):
    import torch
    adapter.load_state_dict({k:torch.from_numpy(bank[k]) for k in EDITABLE},strict=True)


def validate(bank,reference):
    if set(bank)!=set(reference):raise ValueError('tensor names')
    for k,v in bank.items():
        if v.shape!=reference[k].shape or v.dtype!=np.float32 or not np.isfinite(v).all():raise ValueError('shape/type/finite '+k)
        if k not in EDITABLE and v.tobytes()!=reference[k].tobytes():raise ValueError('read-only diagnostic '+k)
        if k in EDITABLE and float(np.linalg.norm(v.astype(np.float64)))>10*max(1.,float(np.linalg.norm(reference[k].astype(np.float64)))):raise ValueError('tensor norm guard '+k)


def evaluate(study,adapter,rows,demos,check=lambda:None):
    records=[]
    with attached(study.model,adapter):
        for row in rows:
            check();answer=generate(study,row['input'],row['task'],demos,'unconstrained');target=row['input'] if row['task']=='copy' else row['output']
            records.append(dict(answer,input=row['input'],target=target,task=row['task'],exact=answer['output']==target))
    return dict(records=records,metrics={task:dict(n=sum(r['task']==task for r in records),exact=sum(r['exact'] for r in records if r['task']==task)) for task in ('movement','copy')})


def diagnostics(study,adapter,rows,demos,check=lambda:None):
    import torch
    hidden=[];scores=[];targets=[];legal=palette_ids(study.tokenizer)
    for row in rows:
        check();tokens,start=encode_task(study.tokenizer,row,row['task'],demos);captured=[]
        def hook(module,args,output):captured.append(output[0] if isinstance(output,tuple) else output)
        handle=study.model.model.layers[27].register_forward_hook(hook)
        try:
            with attached(study.model,adapter),torch.no_grad():
                logits=study.model(input_ids=torch.tensor([tokens],device='cuda'),use_cache=False).logits
                hidden.append(captured[0][0,start-1:len(tokens)-1].float().cpu().numpy())
                scores.append(logits[0,start-1:len(tokens)-1,legal].float().cpu().numpy())
        finally:handle.remove()
        targets.append(row['input'] if row['task']=='copy' else row['output'])
    return dict(H=np.asarray(hidden,dtype=np.float32),digit_logits=np.asarray(scores,dtype=np.float32),
                gold=np.asarray(targets,dtype=np.float32),digit_head=study.model.get_output_embeddings().weight[legal].detach().float().cpu().numpy())
