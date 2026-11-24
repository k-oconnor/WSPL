"""Synthetic data and fake orchestration only; no author calls or task scoring."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.robot_submissions import load_arrays,reconstruct,stage_submission,freeze_ledger,verify_frozen
from skillforge.wspl_lora import identity
from skillforge.robot_gradient import schedule
from skillforge.robot_trials import panel
from skillforge.robot_authoring import run_session
from skillforge.teaching_training import state_hash
import run_robot_authoring_v1 as runner


def bank():
    return {'p.lora_A':np.ones((2,3),np.float32),'p.lora_B':np.zeros((4,2),np.float32)}


class BoundaryTests(unittest.TestCase):
    def test_native_and_wspl_reconstruction(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=bank();candidate=bank();candidate['p.lora_B'][0,0]=.5
            np.savez(root/'native.npz',**candidate)
            (root/'native.json').write_text(json.dumps(dict(parent_id=identity(parent),arrays='native.npz',note='synthetic')))
            (root/'wspl.json').write_text(json.dumps(dict(parent_id=identity(parent),programs={'p':'B[0,0]=1/2;'},bindings=None,note='synthetic')))
            native,nr=reconstruct(parent,root/'native.json','native')
            wspl,wr=reconstruct(parent,root/'wspl.json','wspl')
            self.assertEqual(identity(native),identity(wspl));self.assertEqual(wr['effective_unscaled_delta_norm']['p'],np.sqrt(.75))

    def test_pickle_and_double_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.npz'
            for value in (np.array([{}],dtype=object),np.ones((2,2),np.float64)):
                np.savez(path,x=value)
                with self.assertRaises(ValueError):load_arrays(path)

    def test_duplicate_archive_and_capacity_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.npz';raw=io.BytesIO();np.save(raw,np.ones((2,2),np.float32))
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with zipfile.ZipFile(path,'w') as z:z.writestr('a.npy',raw.getvalue());z.writestr('a.npy',raw.getvalue())
            with self.assertRaises(ValueError):load_arrays(path)
            np.savez(path,a=np.ones((3,3),np.float32))
            with self.assertRaises(ValueError):load_arrays(path,8)

    def test_submission_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);work=root/'work';work.mkdir();(root/'private.json').write_text('{}')
            with self.assertRaises(ValueError):stage_submission(work,'../private.json',root/'stage','native','0'*64)

    def test_binding_and_no_parent_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=bank();before=identity(parent)
            np.savez(root/'const.npz',C=np.full((4,2),.25,np.float32))
            (root/'s.json').write_text(json.dumps(dict(parent_id=before,programs={'p':'B[:,:]=C;'},bindings='const.npz',note='fixture')))
            result,receipt=reconstruct(parent,root/'s.json','wspl')
            self.assertTrue((result['p.lora_B']==.25).all());self.assertEqual(identity(parent),before)
            self.assertIn('sha256=',receipt['expanded_programs']['p'])

    def test_freeze_requires_all_arms_and_detects_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'p.npz';np.savez(path,**bank())
            with self.assertRaises(ValueError):freeze_ledger({'parent':path},root/'f.json')
            arms={k:path for k in ('parent','gradient',*runner.ORDER)};arms['native0']=None
            ledger=freeze_ledger(arms,root/'f.json')
            self.assertEqual(ledger['native0']['status'],'authoring_failed')
            path.write_bytes(b'tampered')
            with self.assertRaises(ValueError):verify_frozen(ledger['parent'])

    def test_balanced_gradient_schedule_and_disjoint_panels(self):
        rows=[dict(condition=c) for c in runner.CONDITIONS]
        batches=schedule(rows,3,4)
        self.assertTrue(all(sorted(b)==[0,0,1,1,2,2,3,3] for b in batches))
        cfg=dict(acquisition_seed=73300000,development_seed=73400000,final_seed=73500000)
        self.assertEqual([len(x) for x in runner.validate_splits(cfg)],[16,16,64])
        cfg['final_seed']=cfg['acquisition_seed']
        with self.assertRaises(ValueError):runner.validate_splits(cfg)


class FakeWorkspace:
    def __init__(self,root,runtime):
        self.work=root/'work';self.inputs=root/'inputs'
        self.work.mkdir(parents=True);self.inputs.mkdir()
    def execute(self,code,timeout):return dict(exit_code=0,stdout=dict(text='fixture'))


class SessionTests(unittest.TestCase):
    def config(self):
        return dict(author_runtime='unused',native_runtime='unused',author_responses=3,proposal_attempts=1,author_call_seconds=1,
                    python_call_seconds=1,cli='unused',catalog='unused',instructions='unused')
    def test_bad_responses_consume_allowance_and_fail_without_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);common=root/'inputs';common.mkdir();count=[];charges=[]
            def collect(*a,**k):count.append(1);return 'bad JSON',{'usage':{}}
            result=run_session('native',common,{'adapter_id':'0'*64},self.config(),root/'author','unused','pin',
                lambda:None,lambda:False,lambda **k:None,lambda *a:None,charges.append,
                collect_fn=collect,workspace_factory=FakeWorkspace)
            self.assertIsNone(result);self.assertEqual(len(count),3);self.assertEqual(len(charges),3)
            frozen=json.loads((root/'author/frozen.json').read_text());self.assertEqual(frozen['status'],'authoring_failed')
    def test_transport_failure_does_not_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);common=root/'inputs';common.mkdir();count=[]
            def collect(*a,**k):count.append(1);raise RuntimeError('transport')
            with self.assertRaisesRegex(RuntimeError,'transport'):
                run_session('native',common,{'adapter_id':'0'*64},self.config(),root/'author','unused','pin',
                    lambda:None,lambda:False,lambda **k:None,lambda *a:None,lambda s:None,
                    collect_fn=collect,workspace_factory=FakeWorkspace)
            self.assertEqual(len(count),1)
    def test_evaluation_attempts_include_malformed_submissions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);common=root/'inputs';common.mkdir()
            def collect(*a,**k):return json.dumps({'tool':'evaluate','path':'missing.json'}),{'usage':{}}
            run_session('native',common,{'adapter_id':'0'*64},self.config(),root/'author','unused','pin',
                lambda:None,lambda:False,lambda **k:None,lambda *a:None,lambda s:None,
                collect_fn=collect,workspace_factory=FakeWorkspace)
            transcript=json.loads((root/'author/transcript.json').read_text())
            self.assertIn('cap exhausted',transcript[-1]['content']['error'])
            self.assertEqual(json.loads((root/'author/frozen.json').read_text())['proposal_attempts'],1)

    def test_final_candidate_is_last_accepted_not_best_score(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);common=root/'inputs';common.mkdir();parent=bank();pid=identity(parent)
            class Workspace(FakeWorkspace):
                def __init__(self,*a):
                    super().__init__(*a);np.savez(self.work/'candidate.npz',**parent)
                    (self.work/'submission.json').write_text(json.dumps(dict(parent_id=pid,arrays='candidate.npz',note='fixture')))
            def collect(*a,**k):return json.dumps(dict(tool='evaluate',path='submission.json')),{'usage':{}}
            def validate(submission,folder):
                np.savez(folder/'candidate.npz',**parent);return dict(status='accepted')
            scores=iter([16,8,0])
            def evaluate(candidate,folder):
                folder.mkdir();(folder/'episodes.json').write_text('[]');return dict(successes=next(scores))
            cfg=self.config();cfg['proposal_attempts']=3
            candidate=run_session('native',common,{'adapter_id':pid},cfg,root/'author','unused','pin',
                lambda:None,lambda:False,lambda **k:None,evaluate,lambda s:None,
                collect_fn=collect,workspace_factory=Workspace,validate_fn=validate)
            self.assertIn('turn_02',candidate)


class OrchestrationTests(unittest.TestCase):
    def test_actual_gradient_control_on_tiny_synthetic_model(self):
        from test_fresh_robot import TinyModel
        from skillforge.robot_gradient import fit
        from skillforge.authoring_sft import adapter_state,frozen_hash,restore_adapter
        class Tiny(TinyModel):
            def gradient_checkpointing_enable(self,**kwargs):pass
        model=Tiny();parent=adapter_state(model);base=frozen_hash(model)
        rows=[dict(condition=c,record={'ids':[1,2],'target':3}) for c in runner.CONDITIONS]
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'control'
            candidate=fit(model,parent,rows,dict(gradient_seed=4,gradient_updates=2,learning_rate=.001),
                          out,'synthetic-fixture',base,lambda:None,lambda **k:None)
            frozen=json.loads((out/'frozen.json').read_text())
            self.assertEqual(frozen['targets'],16);self.assertEqual(frozen['step'],2)
            self.assertEqual(frozen_hash(model),base);self.assertTrue(candidate.exists())
            saved=torch.load(out/'rolling.pt',map_location='cpu',weights_only=True)
            self.assertEqual(state_hash(saved['adapter']),state_hash(adapter_state(model)))
            self.assertEqual(saved['metadata']['manifest_sha256'],'synthetic-fixture')
            restore_adapter(model,parent);self.assertEqual(state_hash(adapter_state(model)),state_hash(parent))

    def test_all_authors_freeze_before_any_final_and_failures_stay_missing(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'source_receipt.json').write_text(json.dumps(dict(manifest_sha256='fixture')))
            codec=out/'codec.fixture';codec.write_text('fixture')
            state={k:torch.from_numpy(v) for k,v in bank().items()};events=[]
            cfg=dict(resources={},acquisition_seed=73300000,development_seed=73400000,final_seed=73500000,
                     development_wall_seconds=100,final_wall_seconds=100,compute_seconds=100,
                     parent_checkpoint='unused',base_hash='base',parent_factor_hash=state_hash(state),codec_source=str(codec))
            def evaluate(model,tokenizer,config,folder,manifest,pin,missions,guard,report):
                folder.mkdir();final=len(missions)==64
                if final:self.assertTrue((out/'all_frozen.json').exists());self.assertEqual(len([e for e in events if e.startswith('author')]),6)
                events.append('final' if final else 'dev')
                result=[dict(**m,success=True,start_sha256=str(m['seed']),model_calls=1,invalid_tokens=0) for m in missions]
                (folder/'episodes.json').write_text(json.dumps(result));return result
            def fit(model,parent,rows,config,folder,*a):
                folder.mkdir();np.savez(folder/'candidate.npz',**bank());return folder/'candidate.npz'
            def session(arm,common,packet,config,folder,*a):
                folder.mkdir();events.append('author:'+folder.name)
                if folder.name=='native0':return None
                np.savez(folder/'candidate.npz',**bank());return folder/'candidate.npz'
            from types import SimpleNamespace
            patches=dict(acquire=lambda *a:[dict(condition=c) for c in runner.CONDITIONS],
                features=lambda *a:({'X':np.zeros((4,3),np.float32)},[],'fixture'),
                evaluate=evaluate,fit=fit,run_session=session,restore_adapter=lambda *a:None,
                frozen_hash=lambda m:'base',adapter_state=lambda m:state)
            with patch.multiple(runner,**patches),patch.object(runner.torch,'load',return_value={'adapter':state}):
                runner.run(lambda:SimpleNamespace(model=object(),tokenizer=object()),out,cfg)
            result=json.loads((out/'results.json').read_text())
            self.assertEqual(result['status'],'complete');self.assertEqual(events.count('final'),7)
            self.assertNotIn('native0',result['comparisons']);self.assertEqual(result['arms']['native0']['status'],'authoring_failed')


if __name__=='__main__':unittest.main()
