"""Fixed ARC answer-likelihood scoring; no hidden generation or fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np


def validate_row(row):
    choices=row['choices'];labels=choices['label'];texts=choices['text']
    if not isinstance(row['question'],str) or not row['question'].strip():raise ValueError('question required')
    if len(labels)!=len(texts) or len(labels)<2 or len(set(labels))!=len(labels) or row['answerKey'] not in labels:
        raise ValueError('invalid choice mapping')
    if any(not isinstance(s,str) or not s.strip() for s in texts):raise ValueError('empty answer text')
    return labels.index(row['answerKey'])


def encode_choices(tokenizer,row,max_length=1024):
    validate_row(row)
    # Freeze separate prefix/continuation tokenization. No chat template or few-shot examples.
    prefix=tokenizer.encode('Question: '+row['question']+'\nAnswer:',add_special_tokens=False)
    encoded=[]
    for text in row['choices']['text']:
        suffix=tokenizer.encode(' '+text,add_special_tokens=False)
        if not prefix or not suffix or len(prefix)+len(suffix)>max_length:raise ValueError('invalid or overlength example; no truncation')
        encoded.append((prefix+suffix,len(prefix)))
    return encoded


def score_encoded(model,encoded,pad_id,device):
    import torch
    n=len(encoded);length=max(len(ids) for ids,_ in encoded)
    ids=torch.full((n,length),pad_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
    for j,(tokens,_) in enumerate(encoded):ids[j,:len(tokens)]=torch.tensor(tokens,device=device);mask[j,:len(tokens)]=1
    with torch.inference_mode():
        logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits.float()
        lp=torch.log_softmax(logits[:,:-1,:],dim=-1).gather(-1,ids[:,1:,None]).squeeze(-1)
        totals=[];counts=[]
        for j,(tokens,start) in enumerate(encoded):
            scores=lp[j,start-1:len(tokens)-1]
            totals.append(float(scores.sum().item()));counts.append(len(tokens)-start)
    if not np.isfinite(totals).all():raise ValueError('nonfinite model score')
    return totals,counts


def score_row(model,tokenizer,row,device='cuda'):
    gold=validate_row(row);encoded=encode_choices(tokenizer,row)
    sums,counts=score_encoded(model,encoded,tokenizer.eos_token_id,device)
    normalized=[s/n for s,n in zip(sums,counts)]
    pred=int(np.argmax(normalized));raw=int(np.argmax(sums))
    return dict(id=row['id'],gold=gold,prediction=pred,raw_prediction=raw,correct=pred==gold,raw_correct=raw==gold,
        answer_logprobs=sums,answer_token_counts=counts,normalized_logprobs=normalized,chance=1/len(sums))


class ArcBaseline:
    def __init__(self,model_path):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        torch.manual_seed(731);torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
            dtype=torch.float32,attn_implementation='eager').to('cuda').eval()
        self.model.requires_grad_(False)
        if self.model.get_input_embeddings().weight.data_ptr()!=self.model.get_output_embeddings().weight.data_ptr():raise ValueError('expected tied embeddings')


def model_hash(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        array=t.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(array.tobytes())
    return h.hexdigest()
