"""Frozen copy foundation and a shared native-state movement adapter."""
from .arc_eval import ArcBaseline,model_hash
from .grid_io import render_prompt,palette_ids
from .grid_domain import enumerate_domain,balanced_take
from .go_replay import sha


class CopyFoundation(ArcBaseline):
    def __init__(self,model_path,checkpoint_path,checkpoint_sha256,model_sha256):
        import torch
        if sha(checkpoint_path)!=checkpoint_sha256:raise ValueError('parent checkpoint identity')
        super().__init__(model_path)
        state=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        self.model.load_state_dict(state['model'],strict=True)
        self.model.eval().requires_grad_(False)
        if model_hash(self.model)!=model_sha256:raise ValueError('foundation tensor identity')


def prepare_data(original,copy):
    excluded={r['id'] for rows in original.values() for r in rows}
    train=[r for r in enumerate_domain() if not r['shape'].startswith('L') and r['id'] not in excluded]
    assert len(train)==168
    return dict(demonstrations=original['demonstrations'],train=train,copy_train=copy['train'],
        training_diagnostic=balanced_take(train,32,317),familiar=original['movement_familiar'],
        novel=original['movement_novel'],copy_domain=original['copy'],copy_random=copy['heldout'][:64])


def encode_task(tokenizer,row,task,demos):
    prefix=tokenizer.encode(render_prompt(row['input'],task,demos),add_special_tokens=False)
    target=row['input'] if task=='copy' else row['output']
    suffix=[palette_ids(tokenizer)[c] for c in target]
    if not prefix or len(prefix)+len(suffix)>768:raise ValueError('sequence budget')
    return prefix+suffix,len(prefix)
