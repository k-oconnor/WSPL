import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from skillforge.wsp_kinetix import KinetixRepair,prompt
from skillforge.codex_one_shot import preflight

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'data/csail_kinetix_reference_v1/exp4_fix_car/runs/oneshot_repair/reply/prompt.txt'

class KinetixAuthorTests(unittest.TestCase):
    @unittest.skipUnless(REF.exists(),'reference prompt required')
    def test_native_wspl_same_fp32_edits(self):
        a=KinetixRepair(REF);b=KinetixRepair(REF)
        original={k:v.copy() for k,v in a.original.items()}
        aa,_,_=a.construct('native','```edits\nembed.b[0]=0.25\nactor_head.b[0:2] += [0.5,-0.5]\n```')
        text='wspl 0.2; target base=artifact("parent",sha256="'+b.parent.sha256+'"); edit base as candidate mode=fixed numeric=native_numpy_v1 {b1[0]=0.25;b7[0:2]+=[0.5,-0.5];}'
        bb,_,_=b.construct('wspl','```wspl\n'+text+'\n```')
        self.assertEqual(aa.sha256,bb.sha256)
        for k in original:np.testing.assert_array_equal(original[k],a.original[k])
        row=np.linspace(-1,1,679,dtype=np.float32)
        np.testing.assert_array_equal(a.logits(row),np.asarray(aa.trace(row,semantics='numpy_f32')['y'],np.float32))
        with self.assertRaises(ValueError):a.construct('native','```edits\nembed.b[0]=0\n```')

    @unittest.skipUnless(REF.exists(),'reference prompt required')
    def test_parent_and_prompt_identity(self):
        a=KinetixRepair(REF);reference=REF.read_text(encoding='utf-8')
        guide=(ROOT/'docs/wspl/KINETIX_REPAIR_INTERFACE_V1.md').read_text()
        x=prompt(reference,guide,'native',a.parent.sha256);y=prompt(reference,guide,'wspl',a.parent.sha256)
        self.assertEqual(x.rsplit('SELECTED ARM:',1)[0],y.rsplit('SELECTED ARM:',1)[0])
        self.assertIn(reference[reference.index('THE WEIGHTS (complete actor).'):],x)
        self.assertNotIn('Only the LAST edits block',x)
        self.assertIn(a.parent.sha256,x)

    @unittest.skipUnless(REF.exists(),'reference prompt required')
    def test_zero_and_rejection(self):
        a=KinetixRepair(REF);a.construct('native','```edits\nZERO actor_head.W\n```')
        self.assertTrue(np.all(a.weights['actor_head.W']==0))
        for answer in ('```edits\nembed.b[0]=__import__("os")\n```','```edits\nembed.b[0]=1\n```\n```edits\nembed.b[0]=2\n```'):
            with self.assertRaises(ValueError):KinetixRepair(REF).construct('native',answer)

    def test_large_preflight_never_uses_large_argv(self):
        text='1234567890'*60000
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/'empty').mkdir()
            def fake(args,**kwargs):
                self.assertLess(sum(len(str(a)) for a in args),16000)
                self.assertNotIn(text,args)
                return SimpleNamespace(stdout='[]',stderr='',returncode=0)
            with patch('skillforge.codex_one_shot.subprocess.run',side_effect=fake),patch('skillforge.codex_one_shot.audit_preview'):
                preflight('fake','catalog','instructions',folder,text)
            binding=json.loads((folder/'preflight_binding.json').read_text())
            self.assertEqual(binding['mode'],'context_only_digest_sentinel')
            self.assertEqual(binding['prompt_utf8_bytes'],600000)

    def test_decision_counts_invalid(self):
        sys.path.insert(0,str(ROOT/'experiments'))
        from run_wsp_kinetix import comparison
        rows=[dict(arm=a,status='complete',evaluation={'success_rate':1.}) for a in ('native','wspl') for _ in range(3)]
        self.assertEqual(comparison(rows)['decision'],'feasible')
        rows[-1]['status']='invalid'
        self.assertEqual(comparison(rows)['decision'],'no_go_this_configuration')
        self.assertEqual(comparison(rows[:-1])['decision'],'incomplete')

    def test_worker_synthetic_success_failure_and_stop(self):
        sys.path.insert(0,str(ROOT/'experiments'))
        from kinetix_author_eval import evaluate
        fakejax=SimpleNamespace(numpy=np,random=SimpleNamespace(PRNGKey=lambda x:x),tree=SimpleNamespace(leaves=lambda x:[x.last_distance]))
        policy=SimpleNamespace(logits=lambda x:np.zeros(16))
        for reward in (1.,-1.):
            class Env:
                def reset(self,*a,**kw):return np.zeros(679),SimpleNamespace(last_distance=-1.)
                def step(self,*a):return np.zeros(679),SimpleNamespace(last_distance=-1.),reward,True,{'GoalR':reward>0,'distance':0.}
            with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,{'jax':fakejax,'jax.numpy':np}):
                out=Path(d);evaluate(policy,Env(),None,None,out,lambda:False,episodes=3)
                result=json.loads((out/'evaluation.json').read_text())
                self.assertEqual(result['outcomes']['success' if reward>0 else 'red'],3)
                self.assertEqual(len(list(out.glob('trajectory_*.json'))),3)
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,{'jax':fakejax,'jax.numpy':np}):
            out=Path(d);evaluate(policy,Env(),None,None,out,lambda:True,episodes=3)
            self.assertEqual(json.loads((out/'evaluation.json').read_text())['status'],'incomplete')

    def test_large_stdin_delivery(self):
        from skillforge.codex_one_shot import collect
        text='long prompt bytes '*40000
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';answer=out/'answer.md'
            code='import sys,json,pathlib; s=sys.stdin.buffer.read(); assert len(s)=='+str(len(text.encode()))+';pathlib.Path('+repr(str(answer))+').write_text("ok");events=[{"type":"thread.started","thread_id":"fixture"},{"type":"turn.started"},{"type":"item.completed","item":{"type":"agent_message","text":"ok"}},{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,"output_tokens":1}}];[print(json.dumps(e),flush=True) for e in events]'
            with patch('skillforge.codex_one_shot.preflight'),patch('skillforge.codex_one_shot.command',return_value=[sys.executable,'-c',code]):
                _,audit=collect('fake','c','i',out,text,10,lambda:False)
            self.assertEqual(audit['prompt_transport']['bytes_written'],len(text.encode()))

    @unittest.skipUnless(os.name=='nt' and REF.exists(),'Windows orchestration fixture')
    def test_complete_host_orchestration_with_fake_services(self):
        sys.path.insert(0,str(ROOT/'experiments'))
        import run_wsp_kinetix as runner
        from skillforge.kinetix_replay import sha
        parent=KinetixRepair(REF)
        guide=ROOT/'docs/wspl/KINETIX_REPAIR_INTERFACE_V1.md'
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'approved_manifest.json').write_text('{}')
            (out/'source_receipt.json').write_text(json.dumps(dict(status='verified',manifest_sha256=sha(out/'approved_manifest.json'))))
            prompts={}
            for arm in ('native','wspl'):
                p=out/(arm+'.txt');p.write_text(prompt(REF.read_text(encoding='utf-8'),guide.read_text(encoding="utf-8"),arm,parent.parent.sha256),encoding='utf-8');prompts[arm]=str(p)
            cfg=dict(bound_files={},linux_python='fixture',prompts=prompts,reference_prompt=str(REF),guide=str(guide),cli='fake',catalog='fake',instructions='fake')
            def collect(cli,catalog,instructions,folder,text,seconds,stop):
                folder.mkdir();(folder/'empty').mkdir()
                if text.endswith('native\n'):answer='```edits\nembed.b[0]=0.25\n```'
                else:answer='```wspl\nwspl 0.2;target base=artifact("parent",sha256="'+parent.parent.sha256+'");edit base as candidate mode=fixed numeric=native_numpy_v1 {b1[0]=0.25;}\n```'
                (folder/'answer.md').write_text(answer)
                return answer,dict(thread_id=folder.name,usage=dict(input_tokens=1,output_tokens=1,cached_input_tokens=0))
            class Proc:
                def __init__(self,args,**kwargs):
                    path=args[args.index('--folder')+1];folder=Path('C:/'+path.removeprefix('/mnt/c/'))
                    (folder/'evaluation.json').write_text(json.dumps(dict(status='complete',success_rate=0. if folder.name=='parent_control' else 1.,completed=1024)))
                    self.returncode=0
                def poll(self):return 0
            with patch.object(runner,'collect',side_effect=collect),patch.object(runner.subprocess,'Popen',Proc):
                runner.run(lambda:KinetixRepair(REF),out,cfg)
            r=json.loads((out/'results.json').read_text())
            self.assertEqual(r['status'],'complete');self.assertEqual(len(r['trials']),6)
            self.assertEqual(r['comparison']['decision'],'feasible')

if __name__=='__main__':unittest.main()
