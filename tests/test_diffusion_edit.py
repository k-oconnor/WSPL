import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from skillforge.diffusion_edit import (DEST,shift,reference_edit,evaluate_arrays,extract,load_candidate,
    DiffusionEditStudy,MODEL_IDS,sha)
from skillforge.diffusion_replay import SHAPES

ORACLE_PY='''import numpy as np
with np.load('/parent.npz',allow_pickle=False) as z: p={k:z[k] for k in z.files}
a={k:v.copy() for k,v in p.items()}
for r in range(12):
 for c in range(12):
  i=12*r+c; j=12*r+(c+2)%12
  a['W1'][j,:]=p['W1'][i,:]
  a['W3'][:,j]=p['W3'][:,i]
  a['b3'][j]=p['b3'][i]
np.savez('/output/candidate.npz',**a)
'''
ORACLE_WSPL='''wspl 0.2;
target base=artifact("parent",sha256="{{PARENT_SHA256}}");
edit base as candidate mode=fixed numeric=exact_to_storage {
 for r in range(12) {
  for c in range(10) {
   W1[12*r+c+2,:]=parent.W1[12*r+c,:];
   W3[:,12*r+c+2]=parent.W3[:,12*r+c];
   b3[12*r+c+2]=parent.b3[12*r+c];
  }
  for c in range(10,12) {
   W1[12*r+c-10,:]=parent.W1[12*r+c,:];
   W3[:,12*r+c-10]=parent.W3[:,12*r+c];
   b3[12*r+c-10]=parent.b3[12*r+c];
  }
 }
}
'''


def synthetic():
    rng=np.random.default_rng(14)
    return {k:rng.uniform(-.01,.01,s).astype(np.float32) for k,s in SHAPES.items()}


class EditTests(unittest.TestCase):
    def test_shift_direction_and_wrap(self):
        x=np.arange(144).reshape(1,144);y=shift(x)
        self.assertEqual(y[0,2],0);self.assertEqual(y[0,0],10);self.assertEqual(y[0,14],12)
        np.testing.assert_array_equal(np.sort(DEST),np.arange(144))

    def test_real_identity_and_coupled_noise_synthetic(self):
        p=synthetic();a=reference_edit(p)
        e,s=evaluate_arrays(p,a,sample_count=2)
        self.assertTrue(e['numeric_pass']);self.assertEqual(e['structural_certificate'],'PROVED_REAL_GRAPH_PERMUTATION')
        np.testing.assert_allclose(s['candidate'],s['target'],atol=1e-6)
        for k in ('b1','W2','b2'):np.testing.assert_array_equal(a[k],p[k])
        np.testing.assert_array_equal(a['W1'][144:],p['W1'][144:])

    def test_wrong_output_only_edit_rejected_by_probes(self):
        p=synthetic();p['W1'][0,0]=1;p['W2'][0,0]=1;p['W3'][0,0]=1
        a=reference_edit(p);a['W1']=p['W1'].copy()
        e,_=evaluate_arrays(p,a,sample_count=1)
        self.assertFalse(e['numeric_pass']);self.assertEqual(e['structural_certificate'],'NOT_ESTABLISHED')

    def test_ws_python_construct_same_full_shape_synthetic(self):
        runtime='/home/kevin/venvs/skillforge-author'
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root.parent) as d:
            folder=Path(d);p=synthetic()
            for name in MODEL_IDS:np.savez(folder/(name+'.npz'),**p)
            study=DiffusionEditStudy(str(folder),{n:sha(folder/(n+'.npz')) for n in MODEL_IDS})
            for arm,source in [('python',ORACLE_PY),('wspl',ORACLE_WSPL)]:
                out=folder/arm;out.mkdir()
                candidates,receipts=study.construct(arm,'```'+arm+'\n'+source+'\n```',out,root/'tools/python_construction_worker.py',runtime)
                for candidate in candidates.values():
                    for k,v in reference_edit(p).items():self.assertEqual(candidate[k].tobytes(),v.tobytes())
                self.assertEqual(len(receipts),3)

    def test_parser_and_npz_fail_closed(self):
        with self.assertRaises(ValueError):extract('```python\npass\n```\n```python\npass\n```','python')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.npz';np.savez(path,unexpected=np.zeros(5))
            with self.assertRaises(ValueError):load_candidate(path)

    def test_sandbox_denies_host_and_network(self):
        import subprocess
        from skillforge.diffusion_edit import linux_path
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root.parent) as d:
            f=Path(d);np.savez(f/'parent.npz',**synthetic())
            source='''import os,socket,numpy as np
assert not os.path.exists('/mnt/c')
assert not os.path.exists('/home/kevin/.codex')
assert not os.path.exists('/dev/nvidia0')
assert not any(k.startswith(('OPENAI_','CODEX_')) for k in os.environ)
try:
 open('/parent.npz','wb')
 raise AssertionError('parent writable')
except OSError: pass
s=socket.socket();s.settimeout(.2)
assert s.connect_ex(('1.1.1.1',443))!=0
try:
 os.fork()
 raise AssertionError('fork allowed')
except OSError: pass
with np.load('/parent.npz') as z: np.savez('/output/candidate.npz',**{k:z[k] for k in z.files})
'''
            (f/'code.py').write_text(source)
            cmd=['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(root/'tools/python_construction_worker.py'),
                 '--source',linux_path(f/'code.py'),'--parent',linux_path(f/'parent.npz'),'--output',linux_path(f/'out'),
                 '--runtime','/home/kevin/venvs/skillforge-author','--receipt',linux_path(f/'receipt.json')]
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=35)
            self.assertEqual(r.returncode,0,r.stderr+(f/'receipt.json').read_text())
            self.assertEqual(json.loads((f/'receipt.json').read_text())['status'],'constructed')

    def test_runner_counts_invalid_and_stops_collection_error(self):
        from run_diffusion_edit import execute
        class Study:
            parents={}
            def construct(self,*args):raise ValueError('bad proposal fixture')
        with tempfile.TemporaryDirectory() as d:
            f=Path(d);prompt=f/'prompt.txt';prompt.write_text('fixture')
            config=dict(prompts={a:str(prompt) for a in ('python','wspl')},worker='unused',linux_runtime='unused')
            calls=[]
            def collector(folder,text,stop):
                calls.append(str(folder));folder.mkdir()
                return 'bad',dict(thread_id=str(folder),usage=dict(output_tokens=4))
            state=execute(Study,f,config,collector,lambda:False,lambda s:None)
            self.assertEqual(state['status'],'complete');self.assertEqual(len(calls),6)
            self.assertTrue(all(r['status']=='invalid_proposal' for r in state['trials']))
            with tempfile.TemporaryDirectory() as e:
                def fail(*args):raise RuntimeError('collector fixture')
                state=execute(Study,Path(e),config,fail,lambda:False,lambda s:None)
            self.assertEqual(state['status'],'incomplete')
            self.assertEqual(sum(r['status']=='not_attempted' for r in state['trials']),5)
