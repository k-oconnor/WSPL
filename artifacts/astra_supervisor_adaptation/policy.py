"""Qwen action-token executive, with a reversible public-interface codec.
No external diagnosis, goal selection, retries, or motor modifications.
P is a fixed bounded probe primitive, not a diagnostic controller.
"""
import sys
from pathlib import Path
sys.path.insert(0,'/inputs/code')
sys.path.insert(0,str(Path(__file__).resolve().parent))
from representation import render,decode_action

class Policy:
    def __init__(self,root,declaration):
        import torch
        from skillforge.supervisor_workbench import StudentLab
        torch.set_num_threads(4)
        self.root=Path(root)
        self.student=StudentLab('/inputs/model',str(self.root/'adapter.pt'),device='cuda' if torch.cuda.is_available() else 'cpu')
        self.student.model.eval()
        self.last_code=None
    def reset(self,public):
        self.student.calls.clear()
        self.last_code=None
    def act(self,observation):
        import torch
        t=self.student.tokenizer
        text=t.apply_chat_template([{'role':'user','content':render(observation)}],tokenize=False,add_generation_prompt=True)
        inputs=t(text,return_tensors='pt',add_special_tokens=False).to(self.student.device)
        with torch.inference_mode():
            logits=self.student.model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1]
            token=int(logits.argmax())
        self.last_code=t.decode([token])
        return decode_action(observation,self.last_code)
    def close(self):
        pass
