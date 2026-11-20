"""Development convenience APIs. Astra may inspect, bypass or extend these.

No benchmark entry point; caller owns resource accounting and data partitions.
"""
import json
from pathlib import Path


class RobotLab:
    def __init__(self,artifact,output):
        self.artifact=Path(artifact);self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.mission=None;self.counter=0
    def reset(self,seed=991,count=3,condition='ordinary',decisions=64):
        from .executive_teaching import TeachingMission
        self.close();self.counter+=1
        self.mission=TeachingMission(self.artifact,self.output/f'episode_{self.counter}',seed,count,condition,decisions)
        return self.mission.public()
    @property
    def env(self):
        return self.mission.backend.env
    def act(self,action):return self.mission.execute(json.dumps(action))
    def capture(self,camera='frontview'):
        return self.mission.backend.render(camera)
    def privileged(self):
        m=self.mission
        return dict(mapping=m.backend.mapping.tolist(),physics=m.backend.env.sim.get_state(),
                    true_trace=m.monitor.true_trace,result=m.result())
    def close(self):
        if self.mission is not None:self.mission.close();self.mission=None


class StudentLab:
    def __init__(self,model_path,parent=None,device='cuda',seed=113):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from .authoring_sft import install,restore_adapter
        torch.manual_seed(seed)
        self.model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
            dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32).to(device)
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        install(self.model)
        if parent:
            # Only load declared trusted starting artifacts here; untrusted submissions
            # must be loaded inside the isolated worker.
            saved=torch.load(parent,map_location='cpu',weights_only=False)
            restore_adapter(self.model,saved['adapter'])
        self.device=device;self.calls=[]
    def generate(self,text,max_new_tokens=96,**options):
        import torch
        rendered=self.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True)
        ids=self.tokenizer(rendered,return_tensors='pt',add_special_tokens=False).to(self.device)
        self.model.eval()
        with torch.inference_mode():
            result=self.model.generate(**ids,max_new_tokens=max_new_tokens,do_sample=False,**options)
        reply=self.tokenizer.decode(result[0,ids.input_ids.shape[1]:],skip_special_tokens=True)
        self.calls.append(dict(prompt=text,reply=reply,input_tokens=ids.input_ids.shape[1],output_tokens=result.shape[1]-ids.input_ids.shape[1]))
        return reply
    def save_adapter(self,path):
        import torch
        from .authoring_sft import adapter_state
        path=Path(path);tmp=path.with_suffix('.tmp')
        torch.save(dict(adapter=adapter_state(self.model)),tmp);tmp.replace(path)
    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]
    # model/tokenizer are intentionally public: arbitrary losses, optimizers,
    # parameter selection, hooks and KV code remain under Astra's control.
