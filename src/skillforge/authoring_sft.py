"""Small author-model LoRA training primitives; no benchmark entry point."""
import hashlib
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


TARGETS={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}


class LoRALinear(nn.Module):
    def __init__(self,base,rank=16,alpha=32):
        super().__init__();self.base=base;self.scale=alpha/rank
        self.base.requires_grad_(False)
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))

    def forward(self,x):
        result=self.base(x)
        update=F.linear(F.linear(x.float(),self.lora_A),self.lora_B)*self.scale
        return result+update.to(result.dtype)


def install(model,rank=16,alpha=32):
    model.requires_grad_(False);names=[]
    for name,module in list(model.named_modules()):
        if name.rsplit('.',1)[-1] in TARGETS and isinstance(module,nn.Linear):
            parent,leaf=name.rsplit('.',1)
            setattr(model.get_submodule(parent),leaf,LoRALinear(module,rank,alpha));names.append(name)
    if not names:raise ValueError('no target projections')
    return names


def adapter_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def restore_adapter(model,state):
    current={n:p for n,p in model.named_parameters() if p.requires_grad}
    if current.keys()!=state.keys():raise ValueError('adapter identity mismatch')
    with torch.no_grad():
        for n,p in current.items():
            if p.shape!=state[n].shape:raise ValueError('adapter shape mismatch')
            p.copy_(state[n])


def frozen_hash(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if p.requires_grad:continue
        h.update(n.encode());h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def encode(tokenizer,messages,max_length=2048):
    if len(messages)!=2 or [m['role'] for m in messages]!=['user','assistant']:
        raise ValueError('one user/assistant pair required')
    prompt=tokenizer.apply_chat_template(messages[:1],tokenize=False,add_generation_prompt=True)
    full=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    p=tokenizer(prompt,add_special_tokens=False)['input_ids']
    ids=tokenizer(full,add_special_tokens=False)['input_ids']
    if ids[:len(p)]!=p:raise ValueError('chat prefix tokenization mismatch')
    if len(ids)>max_length:raise ValueError('record exceeds context; no truncation')
    if len(p)<1 or len(p)>=len(ids):raise ValueError('empty completion or prompt')
    return dict(ids=ids,prefix=len(p),targets=len(ids)-len(p))


def loss_sum(model,record,device='cuda'):
    ids=torch.tensor([record['ids']],dtype=torch.long,device=device)
    # Last prompt position predicts first completion token; final logit is unused.
    count=record['targets']
    logits=model(input_ids=ids,use_cache=False,logits_to_keep=count+1).logits[:,:-1]
    targets=ids[:,record['prefix']:]
    if logits.shape[1]!=count or targets.numel()!=count:raise ValueError('completion mask alignment')
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]).float(),targets.reshape(-1),reduction='sum')


def checkpoint(path,model,optimizer,metadata):
    path=Path(path);temporary=path.with_suffix('.tmp')
    torch.save(dict(adapter=adapter_state(model),optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    metadata=metadata),temporary)
    temporary.replace(path)


class AuthoringSFTStudy:
    def __init__(self,model_path,seed=113,rank=16,alpha=32):
        from transformers import AutoModelForCausalLM,AutoTokenizer
        torch.set_num_threads(4)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
                    dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
        self.model.config.use_cache=False
        self.projections=install(self.model,rank,alpha)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.model.train()
