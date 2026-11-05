import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from skillforge.go_authoring import GoAuthoringStudy,reply_action,conversation_prompt,read_arrays,MAX_STEPS
from run_go_authoring import construct,author_sessions

ROOT=Path(__file__).resolve().parents[1]
CONFIG=dict(python_worker=str(ROOT/'tools/go_python_worker.py'),python_runtime='/home/kevin/venvs/skillforge-author',wspl_worker=str(ROOT/'tools/go_wspl_worker.py'))
PYTHON='''
import os,socket
assert not os.path.exists('/mnt/c') and not os.path.exists('/home/kevin/.codex')
assert not any(k.startswith(('CODEX_','OPENAI_')) for k in os.environ)
s=socket.socket();s.settimeout(.1);assert s.connect_ex(('1.1.1.1',443))!=0
w=load('W1');w[0,0]=1;w[1,1]=1;save('W1',w)
for layer in range(2,7):
 w=load('W'+str(layer))
 for j in range(512):w[j,j]=1
 save('W'+str(layer),w)
w=load('W7');w[0,0]=2;w[1,1]=-3;save('W7',w)
print('construction only',saved()['W1'])
'''
WSPL='''wspl 0.2;target b=artifact("parent",sha256="{{PARENT_SHA256}}");
edit b as c mode=fixed numeric=exact_to_storage {
 W1[0,0]=1; W1[1,1]=1;
 for j in range(512) {W2[j,j]=1; W3[j,j]=1; W4[j,j]=1; W5[j,j]=1; W6[j,j]=1;}
 W7[0,0]=2; W7[1,1]=-3;
}'''


class GoAuthorTests(unittest.TestCase):
    def test_parser_and_transcript(self):
        self.assertEqual(reply_action('FINAL\nfinished','python')[0],'final')
        with self.assertRaises(ValueError):reply_action('FINALish','wspl')
        with self.assertRaises(ValueError):reply_action('```python\npass\n```','wspl')
        history=[dict(assistant='code',construction_feedback=dict(error='bad'))]
        self.assertIn(json.dumps(history),conversation_prompt('guide','wspl',history,1))

    def test_both_full_size_constructions_and_weight_persistence(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as d:
            folder=Path(d);state=GoAuthoringStudy();parent=folder/'parent.npz';state.save(parent);results={}
            for arm,code in [('python',PYTHON),('wspl',WSPL)]:
                f=folder/arm;f.mkdir();path,receipt=construct(arm,code,parent,f,CONFIG)
                self.assertEqual(receipt['exit_code'],0,receipt)
                results[arm]=read_arrays(path)
            for k in results['python']:np.testing.assert_array_equal(results['python'][k],results['wspl'][k])
            self.assertEqual(int(np.count_nonzero(results['wspl']['W4'])),512)
            state.accept(folder/'wspl/constructed/candidate.npz');state.save(parent)
            f=folder/'continue';f.mkdir();path,receipt=construct('python',"b=load('b7');b[81]=-2;save('b7',b)",parent,f,CONFIG)
            self.assertEqual(receipt['exit_code'],0,receipt);arrays=read_arrays(path)
            self.assertEqual(arrays['W7'][0,0],2);self.assertEqual(arrays['b7'][81],-2)

    def test_transaction_rejection_retains_last_valid_state(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);calls=[]
            def collector(folder,prompt,stop):
                folder.mkdir();calls.append(folder)
                answer='```python\nraise ValueError()\n```' if folder.parent.name.endswith('python') else '```wspl\nbad\n```'
                if folder.name=='turn_01':answer='FINAL\nkeep zero'
                return answer,dict(thread_id=str(folder),usage={})
            def compiler(*args):return None,dict(exit_code=1,stdout='',stderr='fixture rejection')
            trials,frozen=author_sessions(GoAuthoringStudy,out,'guide',{},collector,compiler,lambda:False,lambda rows:None)
            self.assertEqual(len(calls),12);self.assertEqual(len(frozen),7)
            for row in trials:
                self.assertEqual(row['turns'][0]['status'],'rejected');self.assertEqual(row['termination'],'author_final')
            for entry in frozen.values():
                self.assertTrue(all(not np.count_nonzero(v) for v in read_arrays(out/entry['path']).values()))

    def test_collection_failure_does_not_freeze_or_evaluate(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            def fail(*args):raise RuntimeError('fixture')
            with self.assertRaises(RuntimeError):author_sessions(GoAuthoringStudy,out,'guide',{},fail,fail,lambda:False,lambda x:None)
            self.assertFalse((out/'frozen_candidates.json').exists())
