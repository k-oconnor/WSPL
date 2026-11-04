"""Known-answer harness validation, never live authoring."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch as mock_patch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
READY=all(importlib.util.find_spec(x) for x in ('numpy','z3'))
if READY:
    from skillforge.wsp_repair import RepairStudy,task,public_task,prompt,ARMS,TASKS,module
    from skillforge.wsp_graph import node
    from run_wsp_repair import execute_trials,comparisons,ORDER


def fence(d):return '```json\n'+json.dumps(d)+'\n```'
def value(shape,values):return dict(shape=shape,values=values)
def patch(task_id):
    w=[1,1] if task_id=='clipped_drive' else [1,-1]
    bias=['1/2','-1/2'] if task_id=='clipped_drive' else ['-1/4','-1/4']
    return dict(graph=dict(kind='dag',inputs={'x':1},nodes=[node('f0','affine',{'x':'input:x'},params={'weight':'A','bias':'bA'},attrs={'layout':'in_out'}),node('r','relu',{'x':'node:f0'}),node('f1','affine',{'x':'node:r'},params={'weight':'B','bias':'bB'},attrs={'layout':'in_out'})],outputs={'y':'node:f1'}),
        tensors={'A':value([1,2],w),'bA':value([2],bias),'B':value([2,1],[1,-1]),'bB':value([1],['-1/2' if task_id=='clipped_drive' else 0])})


def answer(task_id,arm):
    if arm=='direct':
        if task_id=='shared_gain':return fence({'raw':{'put_tensors':{'G':value([1,1],[3])},'put_nodes':[node('n0','affine',{'x':'node:s0'},params={'weight':'G','bias':'b'},attrs={'layout':'in_out'})]}})
        p=patch(task_id);nodes=copy.deepcopy(p['graph']['nodes'])
        nodes[0]['inputs']['x']='node:n0' if task_id=='clipped_drive' else 'node:s0'
        return fence({'raw':{'put_tensors':p['tensors'],'put_nodes':nodes,'outputs':{'drive':'node:f1'}}})
    header='wspl 0.2; target base=artifact("base",sha256="{{base_sha256}}");'
    if task_id=='shared_gain':return fence({'wspl':header+'edit base as candidate mode=graph budget="repair" { untie parameter("n0.weight") as G; G[0,0]=3; }'})
    if task_id=='clipped_drive':
        # Saturate the existing drive through a new module, leaving the original graph intact.
        source=header+'''edit base as candidate mode=graph budget="repair" {
          transplant artifact("m",sha256="{{module_sha256:m}}") as m { inputs={x:signal("node:n0")}; }
          redirect edge("output.drive")=signal("node:m__f1"); }'''
    else:
        source=header+'''region r in base { nodes=[node("n0"),node("n2")];
          inputs={signal:edge("n0.x"),cross:edge("n2.b")};outputs={drive:edge("output.drive")}; shared_parameters=["W","b"]; }
          edit base as candidate mode=graph budget="repair" {
            replace r with artifact("m",sha256="{{module_sha256:m}}") as m {
              inputs={x:r.inputs.signal}; outputs={r.outputs.drive:y}; drop_inputs=[cross]; } }'''
    return fence({'wspl':source,'modules':{'m':patch(task_id)}})


@unittest.skipUnless(READY,'optional NumPy/Z3 unavailable')
class RepairTests(unittest.TestCase):
    def test_known_answers_both_arms_pass_all_external_requirements(self):
        for t in TASKS:
            for a in ARMS:
                with self.subTest(task=t,arm=a):
                    c,_=RepairStudy().construct(t,a,answer(t,a));r=RepairStudy().evaluate(t,c)
                    self.assertTrue(r['success'],r);self.assertEqual(r['formal']['status'],'PROVED')

    def test_unchanged_parents_fail_repair_but_pass_preservation(self):
        for t in TASKS:
            r=RepairStudy().evaluate(t,task(t)['parent']);self.assertFalse(r['success'])
            statuses={x['name']:x['status'] for x in r['formal']['results']}
            self.assertEqual(statuses,{'repair':'REFUTED','preservation':'PROVED'})

    def test_shared_weight_naive_repair_fails_preservation(self):
        c,_=RepairStudy().construct('shared_gain','direct',fence({'raw':{'tensor_source':'W[0,0]=3;'}}))
        r=RepairStudy().evaluate('shared_gain',c);self.assertFalse(r['success'])
        self.assertEqual({x['name']:x['status'] for x in r['formal']['results']},{'repair':'PROVED','preservation':'REFUTED'})

    def test_raw_path_available_to_wspla_without_feature_requirement(self):
        for t in TASKS:
            a,_=RepairStudy().construct(t,'direct',answer(t,'direct'));b,_=RepairStudy().construct(t,'wspl',answer(t,'direct'))
            self.assertEqual(a.sha256,b.sha256)

    def test_interface_and_budget_cannot_be_changed(self):
        for raw in ({'outputs':{'new':'input:x'}},{'put_tensors':{'large':value([129],[0]*129)}}):
            with self.assertRaises(ValueError):RepairStudy().construct('shared_gain','direct',fence({'raw':raw}))

    def test_direct_cannot_use_structured_syntax(self):
        with self.assertRaises(ValueError):RepairStudy().construct('shared_gain','direct',answer('shared_gain','wspl'))

    def test_no_proposer_claims_or_ambiguous_reply(self):
        for text in (fence({'requirements':'true'}),fence({})+fence({}),'```json\n{"raw":{},"raw":{}}\n```'):
            with self.assertRaises(ValueError):RepairStudy().construct('shared_gain','wspl',text)

    def test_module_cannot_supply_nonfinite_or_wrong_shape(self):
        for bad in (value([1],['NaN']),value([1],[True]),dict(shape=[1],bits=['7f800000'])):
            p=patch('clipped_drive');p['tensors']['A']=bad
            with self.assertRaises(ValueError):module(p)

    def test_packet_common_except_arm_selector(self):
        for t in TASKS:
            a=prompt(t,'direct','guide');b=prompt(t,'wspl','guide')
            self.assertEqual(a.rsplit('SELECTED ARM:',1)[0],b.rsplit('SELECTED ARM:',1)[0])
            self.assertIn(task(t)['parent'].sha256,a)

    def test_late_raw_failure_leaves_parent_identity(self):
        before=task('shared_gain')['parent'].sha256
        with self.assertRaises(ValueError):RepairStudy().construct('shared_gain','direct',fence({'raw':{'tensor_source':'W[0,0]=3;b[1000]=1;'}}))
        self.assertEqual(task('shared_gain')['parent'].sha256,before)

    def test_exact_rational_and_bit_storage_paths(self):
        p=patch('clipped_drive');p['tensors']['bB']=dict(shape=[1],bits=['bf000000'])
        self.assertEqual(module(p).sha256,module(patch('clipped_drive')).sha256)

    def test_unknown_is_failure_even_with_correct_numerical_outputs(self):
        c,_=RepairStudy().construct('shared_gain','direct',answer('shared_gain','direct'))
        with mock_patch('skillforge.wsp_repair.check',return_value={'status':'UNKNOWN'}):
            r=RepairStudy().evaluate('shared_gain',c)
        self.assertEqual(r['numerical']['status'],'PASS');self.assertFalse(r['success'])

    def test_numerical_failure_cannot_be_overridden_by_proof_status(self):
        with mock_patch('skillforge.wsp_repair.check',return_value={'status':'PROVED'}):
            r=RepairStudy().evaluate('shared_gain',task('shared_gain')['parent'])
        self.assertEqual(r['numerical']['status'],'FAIL');self.assertFalse(r['success'])

    def test_duplicate_thread_stops_before_second_proposal_grading(self):
        with tempfile.TemporaryDirectory() as temp:
            out,config=self.setup_runner(temp)
            def collect(folder,text,stop):
                folder.mkdir();return 'invalid',dict(thread_id='same',usage={})
            result=execute_trials(RepairStudy,out,config,collect,lambda:False,lambda s:None)
            self.assertEqual(result['status'],'incomplete');self.assertEqual(result['attempted_invocations'],2)
            self.assertEqual(result['trials'][1]['status'],'collection_failed')
            self.assertIn('collection',result['trials'][1])

    def setup_runner(self,temp):
        out=Path(temp);guide=out/'guide.md';guide.write_text('guide',encoding='utf-8');prompts={}
        for t in TASKS:
            prompts[t]={}
            for a in ARMS:
                p=out/f'{t}_{a}.txt';p.write_text(prompt(t,a,'guide'),encoding='utf-8');prompts[t][a]=str(p)
        return out,dict(guide=str(guide),prompts=prompts)

    def test_fake_collection_full_queue_accounting(self):
        with tempfile.TemporaryDirectory() as temp:
            out,config=self.setup_runner(temp);seen=[]
            def collect(folder,text,stop):
                folder.mkdir();t,draw,a=ORDER[len(seen)];seen.append(folder)
                return answer(t,a),dict(thread_id=str(len(seen)),usage=dict(input_tokens=10,cached_input_tokens=0,output_tokens=20))
            result=execute_trials(RepairStudy,out,config,collect,lambda:False,lambda s:None)
            self.assertEqual(result['status'],'complete');self.assertEqual(len(seen),12)
            self.assertEqual(result['comparisons']['decision'],'feasible_but_ceiling_no_advantage_established')
            for a in ARMS:self.assertEqual(result['comparisons']['arms'][a]['successes'],6)

    def test_invalid_responses_count_and_are_never_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            out,config=self.setup_runner(temp);calls=[]
            def collect(folder,text,stop):
                folder.mkdir();calls.append(1);return 'bad',dict(thread_id=str(len(calls)),usage={})
            result=execute_trials(RepairStudy,out,config,collect,lambda:False,lambda s:None)
            self.assertEqual(len(calls),12);self.assertTrue(result['comparisons']['complete'])
            self.assertEqual(result['comparisons']['arms']['wspl']['invalid'],6)

    def test_collection_failure_preserves_unattempted_denominators(self):
        with tempfile.TemporaryDirectory() as temp:
            out,config=self.setup_runner(temp)
            def collect(*args):raise RuntimeError('fake failure')
            result=execute_trials(RepairStudy,out,config,collect,lambda:False,lambda s:None)
            self.assertEqual(result['status'],'incomplete');self.assertEqual(result['attempted_invocations'],1)
            self.assertFalse(result['comparisons']['complete']);self.assertEqual(len(result['trials']),12)
            self.assertTrue(all(p['wspl_minus_direct'] is None for p in result['comparisons']['paired_success_differences']))

    def test_stop_makes_zero_model_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            out,config=self.setup_runner(temp)
            def collect(*args):raise AssertionError('unexpected collection')
            result=execute_trials(RepairStudy,out,config,collect,lambda:True,lambda s:None)
            self.assertEqual(result['status'],'paused');self.assertEqual(result['attempted_invocations'],0)


if __name__=='__main__':unittest.main()
