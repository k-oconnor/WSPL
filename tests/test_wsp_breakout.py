"""Known-answer editor/transport checks, with no live calls or scored gameplay."""
import json
import re
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.wsp_breakout import ReworkedBreakoutPolicy, HEADER, native_source, prompt, frozen_checks, stored_equations
from skillforge.wsp_editor import apply, EditPlan
from skillforge.wsp_artifact import TensorArtifact
from skillforge.native_weight_edits import NativeEditedPolicy
from run_wsp_breakout import execute_trials, comparisons, ORDER, TraceEnvironment


def wspl(body,profile='exact_to_storage'):
    return '```wspl\n'+HEADER+'edit base as candidate mode=fixed numeric='+profile+' {\n'+body+'\n}\n```'


class ConstructionTests(unittest.TestCase):
    def test_shared_guide_examples_compile_in_declared_interfaces(self):
        guide=(ROOT/'docs/wspl/BREAKOUT_REWORKED_INTERFACE_V1.md').read_text(encoding='utf-8')
        examples=re.findall(r'```(edits|wspl)\n(.*?)```',guide,re.S)
        self.assertEqual(len(examples),3)
        for tag,body in examples:
            p=ReworkedBreakoutPolicy();a,_,_=p.construct('native' if tag=='edits' else 'wspl','```'+tag+'\n'+body+'```')
            self.assertEqual(frozen_checks(p,a)['max_absolute_error'],0)

    def test_full_parameter_capacity_and_native_parity(self):
        body='\n'.join(f'{name}[{selection}]=0.25' for name,selection in
                       [('W1',':,:'),('b1',':'),('W2',':,:'),('b2',':'),('W3',':,:'),('b3',':')])
        p=ReworkedBreakoutPolicy(); a,_,r=p.construct('native','```edits\n'+body+'\n```')
        q=ReworkedBreakoutPolicy(); b,_,s=q.construct('wspl',wspl(';\n'.join(body.splitlines())+';', 'native_numpy_v1'))
        self.assertEqual(a.sha256,b.sha256); self.assertEqual(r['nonzero'],20486)
        self.assertEqual(r['parameters'],s['parameters']); self.assertEqual(p.tensor_identity(),q.tensor_identity())
        self.assertEqual(frozen_checks(p,a)['max_absolute_error'],0)

    def test_archived_three_native_controllers_reconstruct_bitwise(self):
        for path in sorted((ROOT/'artifacts/astra_breakout_native_v2').glob('*.answer.md')):
            answer=path.read_text(encoding='utf-8'); old=NativeEditedPolicy(); old.apply_answer(answer)
            p=ReworkedBreakoutPolicy(); a,plan,r=p.construct('native',answer)
            self.assertEqual(old.tensor_identity(),p.tensor_identity())
            self.assertEqual(apply(plan,parent=p.parent,expected_parent_id=p.parent.sha256).sha256,a.sha256)
            self.assertEqual(frozen_checks(p,a)['status'],'EMPIRICAL_PARITY')
        self.assertEqual(len(list((ROOT/'artifacts/astra_breakout_native_v2').glob('*.answer.md'))),3)

    def test_construction_loops_constants_copy_and_actual_equations(self):
        source=HEADER+'const gain=1/4; edit base as candidate mode=fixed { for j in range(4) { W1[j,j]=gain; } W2[0:4,0:4]=copy(W1[0:4,0:4]); b3[0]=1/3; }'
        p=ReworkedBreakoutPolicy(); a,plan,r=p.construct('wspl','```wspl\n'+source+'\n```')
        np.testing.assert_array_equal(a.arrays()['W2'][:4,:4],np.eye(4,dtype=np.float32)/4)
        self.assertEqual(r['resources']['loop_iterations'],4)
        self.assertGreater(plan.to_dict()['rounding']['changed'],0)
        eq=stored_equations(a); self.assertEqual(len(eq['layers'][0]['equations']),32)
        self.assertEqual(eq['formal_status'],'UNSUPPORTED_TANH')

    def test_invalid_native_statement_rejects_whole_proposal(self):
        p=ReworkedBreakoutPolicy(); before=p.parent.sha256
        with self.assertRaises(ValueError):
            p.construct('native','```edits\nb3[0]=2\nb3[9]=1\n```')
        self.assertEqual(p.parent.sha256,before)

    def test_invalid_wspla_statement_rejects_whole_proposal(self):
        p=ReworkedBreakoutPolicy(); before=p.parent.sha256
        with self.assertRaises(ValueError):p.construct('wspl',wspl('b3[0]=2;b3[9]=1;'))
        self.assertEqual(p.parent.sha256,before)

    def test_architecture_cannot_change(self):
        for source in (wspl('b3[0]=1;').replace('tanh,tanh','relu,relu'),
                       wspl('b3[0]=1;').replace('600,32,32,6','600,33,32,6')):
            with self.assertRaises(ValueError):ReworkedBreakoutPolicy().construct('wspl',source)

    def test_graph_modes_descriptions_and_circuit_declarations_rejected(self):
        for source in (wspl('').replace('mode=fixed','mode=graph budget="other"'),
                       wspl('').replace('\n```','\ndescribe candidate { hypothesis h="test"; }\n```')):
            with self.assertRaises(ValueError):ReworkedBreakoutPolicy().construct('wspl',source)

    def test_no_python_execution_or_unbounded_coefficients(self):
        for body in ('b3[0]=__import__("os");','b3[0]=1000001;','b3[0]=1/0;'):
            with self.assertRaises(ValueError):ReworkedBreakoutPolicy().construct('wspl',wspl(body))

    def test_exactly_one_matching_fence_both_arms(self):
        for arm,answer in [('native','```edits\nb3[:]=0\n```'),('wspl',wspl('b3[:]=0;'))]:
            for bad in (answer+answer,answer.replace('```','~~~'),answer.replace('edits','python').replace('```wspl','```python')):
                with self.assertRaises(ValueError):ReworkedBreakoutPolicy().construct(arm,bad)

    def test_one_proposal_per_constructor(self):
        p=ReworkedBreakoutPolicy();p.construct('wspl',wspl('b3[:]=0;'))
        with self.assertRaises(ValueError):p.construct('wspl',wspl('b3[:]=1;'))

    def test_saved_plan_artifact_replay_and_tampering(self):
        p=ReworkedBreakoutPolicy();a,plan,_=p.construct('wspl',wspl('b3[0]=1;'))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);plan.save(path/'plan.json');a.save(path/'a.json')
            self.assertEqual(TensorArtifact.load(path/'a.json').sha256,a.sha256)
            self.assertEqual(apply(EditPlan.load(path/'plan.json'),parent=p.parent,expected_parent_id=p.parent.sha256).sha256,a.sha256)
            d=json.loads((path/'plan.json').read_text());d['body']['source']+='bad'
            (path/'plan.json').write_text(json.dumps(d))
            with self.assertRaises(ValueError):EditPlan.load(path/'plan.json')

    def test_common_prompt_excludes_original_skipping_instruction(self):
        original=(ROOT/'data/csail_breakout_reference_v1/exp5_write_from_scratch/runs/breakout/message.txt').read_text(encoding='utf-8')
        a=prompt(original,'shared guide','native');b=prompt(original,'shared guide','wspl')
        self.assertEqual(a.rsplit('SELECTED ARM:',1)[0],b.rsplit('SELECTED ARM:',1)[0])
        self.assertIn('THE INPUT',a);self.assertNotIn('a statement that cannot be read is skipped',a)


class RunnerTests(unittest.TestCase):
    def setup_fake(self,temp):
        out=Path(temp);guide=out/'guide';guide.write_text('guide',encoding='utf-8')
        ref=out/'ref';ref.write_text('THE GAME: game\nTHE NETWORK (network)',encoding='utf-8')
        prompts={}
        for a in ('native','wspl'):
            p=out/(a+'.txt');p.write_text(prompt(ref.read_text(),'guide',a),encoding='utf-8');prompts[a]=str(p)
        return out,dict(guide=str(guide),reference_prompt=str(ref),prompts=prompts,episodes=2)

    def collector(self):
        seen=[]
        def call(folder,text,stop):
            folder.mkdir();draw,arm=ORDER[len(seen)];seen.append(arm)
            answer='```edits\nb3[:]=0\n```' if arm=='native' else wspl('b3[:]=0;')
            (folder/'answer.md').write_text(answer,encoding='utf-8')
            return answer,dict(thread_id=str(len(seen)),usage=dict(input_tokens=10,cached_input_tokens=0,output_tokens=20))
        return call

    def evaluator(self,policy,trial,index,folder,stop):
        return dict(status='complete',reward=200 if trial['arm']=='native' else 190,
                    steps=1,truncated=False,terminal=True,env_seed=1110000+index,action_seed=1210000+index,action_counts=[1,0,0,0,0,0])

    def test_full_fake_screen_saves_replayable_artifacts_and_paired_records(self):
        with tempfile.TemporaryDirectory() as temp:
            out,c=self.setup_fake(temp)
            r=execute_trials(ReworkedBreakoutPolicy,out,c,self.collector(),lambda:False,lambda s:None,self.evaluator)
            self.assertEqual(r['status'],'complete');self.assertEqual(r['comparisons']['decision'],'feasible_for_next_domain')
            for t in r['trials']:
                self.assertEqual(t['completed_episodes'],2)
                folder=out/f"draw_{t['draw']}_{t['arm']}"
                index=json.loads((folder/'artifact_index.json').read_text())
                self.assertTrue({'answer.md','policy.npz','candidate.json','edit.plan.json','stored_equations.json','deployment_check.json','episodes.jsonl'}<=set(index))
                self.assertEqual(json.loads((folder/'episodes.jsonl').read_text().splitlines()[0])['env_seed'],1110000)
            self.assertEqual(r['comparisons']['arms']['wspl']['failure_adjusted_mean'],190)

    def test_all_invalid_count_zero_keep_denominator_and_raw_answers(self):
        with tempfile.TemporaryDirectory() as temp:
            out,c=self.setup_fake(temp);counter=[]
            def invalid(folder,*args):
                folder.mkdir();(folder/'answer.md').write_text('bad');counter.append(1)
                return 'bad',dict(thread_id=str(len(counter)),usage={})
            r=execute_trials(ReworkedBreakoutPolicy,out,c,invalid,lambda:False,lambda s:None,self.evaluator)
            self.assertEqual(r['attempted_invocations'],6);self.assertTrue(r['comparisons']['complete'])
            self.assertEqual(r['comparisons']['arms']['wspl']['failure_adjusted_mean'],0)

    def test_collection_failure_does_not_retry_or_impute_remaining_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            out,c=self.setup_fake(temp)
            def fail(*args):raise RuntimeError('fake quota')
            r=execute_trials(ReworkedBreakoutPolicy,out,c,fail,lambda:False,lambda s:None,self.evaluator)
            self.assertEqual(r['status'],'incomplete');self.assertEqual(r['attempted_invocations'],1)
            self.assertIsNone(r['comparisons']['arms']['wspl']['failure_adjusted_mean'])

    def test_duplicate_threads_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            out,c=self.setup_fake(temp);collector=self.collector()
            def duplicate(*args):
                answer,audit=collector(*args);audit['thread_id']='same';return answer,audit
            r=execute_trials(ReworkedBreakoutPolicy,out,c,duplicate,lambda:False,lambda s:None,self.evaluator)
            self.assertEqual(r['status'],'incomplete');self.assertEqual(r['attempted_invocations'],2)

    def test_stop_and_partial_episode_are_not_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            out,c=self.setup_fake(temp)
            r=execute_trials(ReworkedBreakoutPolicy,out,c,self.collector(),lambda:True,lambda s:None,self.evaluator)
            self.assertEqual(r['status'],'paused');self.assertEqual(r['attempted_invocations'],0)
        with tempfile.TemporaryDirectory() as temp:
            out,c=self.setup_fake(temp)
            r=execute_trials(ReworkedBreakoutPolicy,out,c,self.collector(),lambda:False,lambda s:None,
                             lambda *args:dict(status='incomplete',steps=0))
            self.assertEqual(r['status'],'paused');self.assertIsNone(r['trials'][0]['summary'])
            self.assertFalse(r['comparisons']['complete'])

    def test_trace_wrapper_adds_no_environment_queries(self):
        class Fake:
            def __init__(self):self.queries=0;self.acts=0
            def state(self):self.queries+=1;return np.zeros((10,10,4),np.bool_)
            def act(self,a):self.acts+=1;return 1,True
        fake=Fake();wrapped=TraceEnvironment(fake);wrapped.state();wrapped.act(3)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'trace.npz';wrapped.save(p)
            with np.load(p,allow_pickle=False) as d:
                self.assertEqual(d['states'].shape,(1,10,10,4));self.assertEqual(d['requested_actions'].tolist(),[3])
        self.assertEqual((fake.queries,fake.acts),(1,1))


if __name__=='__main__':unittest.main()
