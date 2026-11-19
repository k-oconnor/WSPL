"""Synthetic CPU orchestration fixture, with no learned-model predictions."""
import json
import gzip
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch

import run_closed_loop as runner
from skillforge.authoring_sft import adapter_state,frozen_hash
from skillforge.closed_loop_control import make_task,teacher
from skillforge.controller_author import KINDS,construct,teacher as block_teacher,make_spec
from skillforge.executive_study import teacher as old_teacher


class CPUTensor(torch.Tensor):
    def to(self,*args,**kwargs):
        if args and args[0]=='cuda':return self
        return super().to(*args,**kwargs)


class Tokenizer:
    eos_token_id=0
    def __init__(self):self.strings={};self.index=0
    def token(self,text):
        self.index+=1;self.strings[self.index]=text;return self.index
    def apply_chat_template(self,messages,**kwargs):return messages[0]['content']
    def __call__(self,text,**kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[self.token(text)]]).as_subclass(CPUTensor))
    def decode(self,tokens,**kwargs):return self.strings[tokens[0]]


class Model(torch.nn.Module):
    def __init__(self,tokenizer,spec):
        super().__init__();self.base=torch.nn.Linear(1,1);self.base.requires_grad_(False)
        self.lora_A=torch.nn.Parameter(torch.zeros(1));self.tokenizer=tokenizer;self.spec=spec
    def gradient_checkpointing_disable(self):pass
    def gradient_checkpointing_enable(self,**kwargs):pass
    def generate(self,ids,**kwargs):
        text=self.tokenizer.strings[int(ids[0,0])]
        if 'Observed state: ' in text:answer=teacher(json.loads(text.split('Observed state: ')[1]))
        elif 'Scene and task: ' in text:answer=old_teacher(json.loads(text.split('Scene and task: ')[1]))
        else:answer=block_teacher(self.spec)
        return torch.cat([ids,torch.tensor([[self.tokenizer.token(answer)]]).as_subclass(CPUTensor)],dim=1)


class RunnerTests(unittest.TestCase):
    def test_synthetic_training_loop_updates_and_preserves_final(self):
        with tempfile.TemporaryDirectory() as temp:
            tokenizer=Tokenizer();model=Model(tokenizer,{})
            data=dict(train=[dict(messages=[]) for _ in range(16)],development=[dict(messages=[])])
            reports=[]
            def save_checkpoint(path,model,opt,metadata):
                torch.save(dict(adapter=adapter_state(model),metadata=metadata),path)
            with patch.object(runner,'encode',return_value=dict(targets=1)),\
                 patch.object(runner,'loss_sum',side_effect=lambda model,row:(model.lora_A-1).square().sum()),\
                 patch.object(runner,'checkpoint',side_effect=save_checkpoint),\
                 patch.object(runner.torch.cuda,'synchronize'),\
                 patch.object(runner.torch.cuda,'empty_cache'):
                runner.fit_seed(model,tokenizer,data,Path(temp)/'seed',dict(epochs=2),7,
                                'engineering-fixture',frozen_hash(model),lambda:None,lambda **kw:reports.append(kw))
            result=json.loads((Path(temp)/'seed/training_summary.json').read_text())
            self.assertEqual(result['steps'],4)
            self.assertEqual(result['targets'],32)
            self.assertGreater(float(model.lora_A),0)
            self.assertTrue((Path(temp)/'seed/adapter_final.pt').exists())

    def test_all_phases_freeze_and_compare_on_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);specs={k:make_spec(k,20) for k in KINDS}
            blocks={k:construct(block_teacher(v),v) for k,v in specs.items()}
            changed=dict(specs['move'],gain=-specs['move']['gain'])
            tokenizer=Tokenizer();model=Model(tokenizer,changed)
            study=SimpleNamespace(model=model,tokenizer=tokenizer)
            for name,manifest in [('author','5de7ada4ce5d2c3c41614896e568cca4b91bc63991f5f4ce997a7e9404d1f3ae'),
                                  ('previous','43dabbb5b31820586f368de2037e2673767c71eafa89c55833d2d1fa6b60f5fc')]:
                torch.save(dict(adapter=adapter_state(model),base_hash=frozen_hash(model),manifest_sha256=manifest),p/(name+'.pt'))
            tasks=[make_task(21+i,2,d,panel=panel) for i,(panel,d) in enumerate(
                   [('long8','none'),('slip8','slip'),('blocked8','blocked'),('combined12','combined')])]
            (p/'source_receipt.json').write_text(json.dumps(dict(manifest_sha256='engineering-fixture')))
            def fit(model,tokenizer,data,out,*args):
                out.mkdir();torch.save(dict(adapter=adapter_state(model)),out/'adapter_final.pt')
            def proof(args,**kwargs):
                Path(args[-1]).write_text(json.dumps(dict(status='UNKNOWN',reason='engineering stub')))
                return SimpleNamespace(stdout='',stderr='')
            config=dict(resources={},library={},train_rows=1,dev_rows=1,seeds=[7],epochs=2,
                        wall_guard_seconds=60,author_adapter=str(p/'author.pt'),previous_adapter=str(p/'previous.pt'),
                        replacement_updates=1,proof_python='engineering-stub')
            with patch.object(runner,'load_library',return_value=(blocks,specs)),\
                 patch.object(runner,'training_data',return_value=dict(train=[{}],development=[{}])),\
                 patch.object(runner,'evaluation_tasks',return_value=tasks),\
                 patch.object(runner,'fit_seed',side_effect=fit),\
                 patch.object(runner,'fit_replacement',return_value=(construct(block_teacher(changed),changed),dict(synthetic=True))),\
                 patch.object(runner.subprocess,'run',side_effect=proof):
                runner.run(lambda:study,p,config)
            result=json.loads((p/'results.json').read_text())
            self.assertEqual(result['status'],'complete')
            self.assertTrue(result['reference_passed'])
            scores=json.loads((p/'scores.json').read_text())
            self.assertEqual(len(scores),25)
            trace=json.loads(gzip.decompress((p/scores[0]['trace_path']).read_bytes()))
            self.assertTrue(trace['trace'])
            self.assertTrue(trace['decisions'])
            self.assertTrue(all(r['task_success'] for r in scores if r['arm']=='closed'))
            self.assertFalse(next(r for r in scores if r['arm']=='inverted_unchanged')['task_success'])
            self.assertTrue(next(r for r in scores if r['arm']=='inverted_authored')['task_success'])
            self.assertTrue(all(r['reused'] for r in scores if r['arm']=='nominal_plan' and r['panel'] in ('long8','blocked8')))


if __name__=='__main__':unittest.main()
