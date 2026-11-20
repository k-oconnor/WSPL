import json
from pathlib import Path
import re
import tempfile
import unittest
import numpy as np
from skillforge.supervisor_persistence import run_gate
from skillforge.supervisor_workspace import Budget,digest


class Bridge:
    def __init__(self,work,mode='ok'):self.work=work;self.mode=mode;self.stage=0
    def collect(self,instruction,history,folder):
        self.nonce=re.search(r'\b[0-9a-f]{32}\b',instruction).group()
        return json.dumps(dict(tool='python',code='pass',gpu=self.mode=='gpu',seconds=60))
    def execute(self,request,folder):
        p=self.work/'persistence_gate'
        if self.mode=='missing':return dict(outcome='complete')
        if self.stage==0:
            p.mkdir();np.savez(p/'weights.npz',weights=np.array([[1,2],[3,5]],dtype=np.float32))
            (p/'receipt.json').write_text(json.dumps(dict(nonce=self.nonce)))
        else:
            (p/'reload.json').write_text(json.dumps(dict(nonce=self.nonce,
                output=[0,0] if self.mode=='bad' else [8,21],weights_sha256=digest(p/'weights.npz'))))
        self.stage+=1
        return dict(outcome='complete')


class PersistenceTests(unittest.TestCase):
    def check(self,mode):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);work=root/'work';work.mkdir()
            budget=Budget(root/'budget.json',60,30,2)
            result=run_gate(root/'gate',work,budget,Bridge(work,mode))
            self.assertEqual(len(result),6)
            self.assertEqual(json.loads((root/'budget.json').read_text())['calls_used'],2)
            self.assertEqual(json.loads((root/'gate/receipt.json').read_text())['status'],'passed')
    def test_export_reload_and_accounting(self):self.check('ok')
    def test_claim_without_file_fails(self):
        with self.assertRaises(FileNotFoundError):self.check('missing')
    def test_bad_reload_fails(self):
        with self.assertRaises(ValueError):self.check('bad')
    def test_gpu_request_fails(self):
        with self.assertRaises(ValueError):self.check('gpu')


if __name__=='__main__':unittest.main()
