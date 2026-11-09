import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from skillforge.arc_authoring import parse_reply,inspect_bank,read_bank,prompt,ORDER
from skillforge.smollm_adapter import initial_bank,LAYERS,tensor_hash
from run_arc_authoring import author_session,construct

ROOT=Path(__file__).resolve().parents[1]


class ArcAuthorTests(unittest.TestCase):
    def base(self):return {f'W{i}':np.full((576,576),.01,np.float32) for i in LAYERS}

    def test_reply_bounds_and_common_inspection(self):
        self.assertEqual(parse_reply('FINAL','lora'),('final',None))
        with self.assertRaises(ValueError):parse_reply('```python\npass\n```','wspl')
        bank={'P26':np.ones((576,576),np.float32)}
        req=dict(tensor='P26',row_start=0,row_count=1,col_start=0,col_count=2)
        self.assertEqual(inspect_bank(bank,[req])[0]['values'],[[1.,1.]])
        req['col_count']=576
        with self.assertRaises(ValueError):inspect_bank(bank,[req])
        self.assertEqual(len(set(ORDER)),9)

    def test_npz_shapes_and_extra_members_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.npz';expected={'A':np.zeros((2,3),np.float32)}
            np.savez_compressed(p,A=np.zeros((2,4),np.float32))
            with self.assertRaises(ValueError):read_bank(p,expected)
            np.savez_compressed(p,A=expected['A'],hidden=np.zeros(1))
            with self.assertRaises(ValueError):read_bank(p,expected)

    def test_inspection_and_rejection_preserve_parent_no_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);base=self.base();calls=[]
            def collector(folder,text,stop):
                folder.mkdir();calls.append(folder)
                answers=['INSPECT\n'+json.dumps([dict(tensor='P26',row_start=0,row_count=1,col_start=0,col_count=1)]),'```wspl\nbad\n```','FINAL']
                return answers[len(calls)-1],dict(thread_id=str(folder),usage={})
            def compiler(*args):return None,dict(exit_code=1,stderr='fixture',stdout='')
            def forbidden(*args):raise AssertionError('must not score')
            row=author_session('0_wspl','wspl',out,base,{},'guide',{},collector,compiler,forbidden,lambda:False,lambda row:None,set())
            self.assertEqual([t['status'] for t in row['turns']],['inspection','rejected','final'])
            bank=read_bank(out/row['path'],initial_bank(base,'direct'))
            self.assertEqual(tensor_hash(bank['W26']),tensor_hash(base['W26']))

    def test_failed_collection_does_not_freeze(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            def fail(*args):raise RuntimeError('timeout')
            with self.assertRaises(RuntimeError):author_session('0_lora','lora',out,self.base(),{},'guide',{},fail,fail,fail,lambda:False,lambda row:None,set())
            self.assertFalse((out/'0_lora/final.npz').exists())

    def test_four_accepted_steps_freeze_latest_not_validation_best(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);base=self.base();expected=initial_bank(base,'direct');evaluations=[]
            def collector(folder,*args):
                folder.mkdir();return '```wspl\nfixture\n```',dict(thread_id=str(folder),usage={})
            def compiler(arm,source,parent,bp,folder,cfg):
                bank=read_bank(parent,expected);bank['W26'][0,0]+=np.float32(1/1024)
                p=folder/'candidate.npz';np.savez_compressed(p,**bank)
                return p,dict(exit_code=0,stderr='',stdout='')
            def evaluate(weights,path):
                evaluations.append(float(weights['W26'][0,0]));return dict(accuracy=1-len(evaluations)/10)
            row=author_session('0_wspl','wspl',out,base,{},'guide',{},collector,compiler,evaluate,lambda:False,lambda row:None,set())
            self.assertEqual(len(evaluations),4);self.assertEqual(row['termination'],'response_budget_exhausted')
            self.assertEqual(float(read_bank(out/row['path'],expected)['W26'][0,0]),evaluations[-1])

    def test_full_size_native_and_wspl_match_with_readonly_base(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as d:
            out=Path(d);base=self.base();bank=initial_bank(base,'direct')
            state=out/'parent.npz';np.savez_compressed(state,**bank);bp=out/'base.npz';np.savez_compressed(bp,**base)
            config=dict(python_worker=str(ROOT/'tools/arc_python_worker.py'),wspl_worker=str(ROOT/'tools/arc_wspl_worker.py'),python_runtime='/home/kevin/venvs/skillforge-author')
            sources=dict(native="import os\nassert not os.path.exists('/mnt/c')\nw=load('W26');w[0,0]=np.float32(0.0102);save('W26',w)\nprint('not returned')",
                wspl='wspl 0.2;target b=artifact("parent",sha256="{{PARENT_SHA256}}");edit b as c mode=fixed numeric=exact_to_storage {W26[0,0]=51/5000;}')
            results={}
            for arm,source in sources.items():
                folder=out/arm;folder.mkdir();p,feedback=construct(arm,source,state,bp,folder,config)
                self.assertEqual(feedback['exit_code'],0,feedback);self.assertEqual(feedback['stdout'],'')
                results[arm]=read_bank(p,bank)
            for k in bank:np.testing.assert_array_equal(results['native'][k],results['wspl'][k])
            self.assertEqual(tensor_hash(results['wspl']['P26']),tensor_hash(bank['P26']))
