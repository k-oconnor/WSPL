"""Stateless graph surgery engineering checks; no training or scored episodes."""
import copy
from dataclasses import asdict, replace as change_budget
from fractions import Fraction as Q
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
HAS_NUMPY=importlib.util.find_spec('numpy') is not None
HAS_Z3=importlib.util.find_spec('z3') is not None
if HAS_NUMPY:
    import numpy as np
    from skillforge.wsp_artifact import TensorArtifact,inspect,diff,canonical
    from skillforge.wsp_editor import plan,apply,EditPlan,EditError
    from skillforge.wsp_evidence import describe,check,probe,equations
    from skillforge.wsp_graph import (node,to_dag,validate_graph,GraphBudget,region,declared_region,
                                     replace_region,transplant,redirect,untie,add_residual)


def model(nodes,arrays,*,inputs=None,outputs=None):
    return TensorArtifact.from_arrays({k:np.array(v,dtype=np.float32) for k,v in arrays.items()},
        graph=dict(kind='dag',inputs=inputs or {'x':1},nodes=nodes,outputs=outputs or {'y':'node:sum'}))


def branched():
    return model([node('left','affine',{'x':'input:x'},params={'weight':'W','bias':'lb'},attrs={'layout':'in_out'}),
                  node('right','affine',{'x':'input:x'},params={'weight':'W','bias':'rb'},attrs={'layout':'in_out'}),
                  node('sum','add',{'a':'node:left','b':'node:right'})],
                 {'W':[[2]],'lb':[1],'rb':[0]},outputs={'y':'node:sum','keep':'node:right'})


def linear(gain=3,bias=0,width=1):
    dense=TensorArtifact.dense([width,width],['linear'])
    arrays={'W1':np.eye(width,dtype=np.float32)*np.float32(gain),'b1':np.full((width,),bias,dtype=np.float32)}
    return to_dag(TensorArtifact.from_arrays(arrays,graph=dense.to_dict()['graph']))


def left_region(parent):
    return declared_region(parent,['left'],{'obs':'left.x'},{'contribution':'sum.a'},['W'])


def source(parent,body,regions=''):
    return f'wspl 0.2; target base=artifact("base",sha256="{parent.sha256}"); {regions} edit base as candidate mode=graph budget="repair" {{ {body} }}'


REGION='''region steering in base {
 nodes=[node("left")]; inputs={obs:edge("left.x")};
 outputs={contribution:edge("sum.a")}; shared_parameters=["W"];
}'''


def replace_body(patch):
    return f'''replace steering with artifact("patch",sha256="{patch.sha256}") as fixed {{
 inputs={{x:steering.inputs.obs}}; outputs={{steering.outputs.contribution:y}};
}}'''


@unittest.skipUnless(HAS_NUMPY,'optional NumPy unavailable')
class DAGTests(unittest.TestCase):
    def test_graphify_keeps_every_tensor_bit_and_outputs(self):
        rng=np.random.default_rng(7)
        d=TensorArtifact.dense([3,5,2],['relu','linear'])
        a=TensorArtifact.from_arrays({k:rng.normal(size=v.shape).astype(np.float32) for k,v in d.arrays().items()},graph=d.to_dict()['graph'])
        g=to_dag(a)
        self.assertNotEqual(a.sha256,g.sha256)
        self.assertEqual(a.to_dict()['tensors'],g.to_dict()['tensors'])
        for x in rng.normal(size=(32,3)).astype(np.float32):
            np.testing.assert_array_equal(np.array(a.trace(x,semantics='numpy_f32')['y']).view(np.uint32),np.array(g.trace({'x':list(x)},semantics='numpy_f32')['y']).view(np.uint32))
        self.assertEqual(a.trace([Q(1),Q(-2),Q(3)])['y'],g.trace({'x':[Q(1),Q(-2),Q(3)]})['y'])

    def test_topology_is_validated_independent_of_declaration_order(self):
        a=branched();d=a.to_dict();d['graph']['nodes'].reverse();b=TensorArtifact(d)
        self.assertEqual(a.trace({'x':[2]}),b.trace({'x':[2]}))
        self.assertEqual(a.trace({'x':[2]})['y'],[Q(9)])

    def test_dangling_cycles_and_unknown_ops_rejected(self):
        for change in ('dangling','cycle','unknown'):
            d=branched().to_dict()
            if change=='dangling':d['graph']['nodes'][0]['inputs']['x']='node:absent'
            elif change=='cycle':d['graph']['nodes'][0]['inputs']['x']='node:sum'
            else:d['graph']['nodes'][0]['op']='python'
            with self.subTest(change=change),self.assertRaises(ValueError):TensorArtifact(d)

    def test_width_bias_and_port_schema_rejected(self):
        for change in ('bias','width','port','output'):
            d=branched().to_dict()
            if change=='bias':d['graph']['nodes'][0]['params']['bias']='W'
            elif change=='width':d['graph']['inputs']['x']=2
            elif change=='port':d['graph']['nodes'][2]['inputs']['c']='input:x'
            else:d['graph']['outputs']['y']='node:absent'
            with self.subTest(change=change),self.assertRaises(ValueError):TensorArtifact(d)

    def test_shared_storage_is_explicit_in_inspection(self):
        a=branched();r=inspect(a)
        self.assertEqual(r['aliases'],[dict(tensor='W',references=['left.weight','right.weight'])])
        self.assertTrue(r['capabilities']['graph_surgery'])

    def test_add_has_no_implicit_broadcast(self):
        d=branched().to_dict();d['graph']['nodes'][2]['inputs']['b']='input:other';d['graph']['inputs']['other']=2
        with self.assertRaisesRegex(ValueError,'widths differ'):TensorArtifact(d)

    def test_multi_input_order_and_concat_slice(self):
        a=model([node('join','concat',{'a':'input:a','b':'input:b'}),node('tail','slice',{'x':'node:join'},attrs={'start':1,'stop':4,'step':1})],{},inputs={'a':2,'b':2},outputs={'y':'node:tail'})
        self.assertEqual(a.trace({'a':[1,2],'b':[3,4]})['y'],[Q(2),Q(3),Q(4)])
        np.testing.assert_array_equal(a.trace({'a':[1,2],'b':[3,4]},semantics='numpy_f32')['y'],[2,3,4])

    def test_input_only_dag_supported(self):
        a=model([],{},inputs={'x':2},outputs={'y':'input:x'})
        self.assertEqual(a.trace({'x':[1,2]})['y'],[Q(1),Q(2)])

    def test_immutable_import_and_roundtrip(self):
        a=branched();d=a.to_dict();d['graph']['nodes'][0]['inputs']['x']='node:sum'
        self.assertEqual(a.trace({'x':[1]})['y'],[Q(5)])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.json';a.save(p);self.assertEqual(TensorArtifact.load(p).sha256,a.sha256)


@unittest.skipUnless(HAS_NUMPY,'optional NumPy unavailable')
class BoundaryTests(unittest.TestCase):
    def test_shared_parameter_and_every_consumer_in_cut(self):
        a=branched();cut=region(a,['right'])
        self.assertEqual(cut['outputs'],{'output.keep':'node:right','sum.b':'node:right'})
        self.assertEqual(cut['shared_parameters'],{'W':['left.weight','right.weight']})

    def test_missing_public_output_edge_rejected(self):
        with self.assertRaisesRegex(ValueError,'outputs'):
            declared_region(branched(),['right'],{'obs':'right.x'},{'out':'sum.b'},['W'])

    def test_missing_shared_parameter_rejected(self):
        with self.assertRaisesRegex(ValueError,'shared parameter'):
            declared_region(branched(),['left'],{'obs':'left.x'},{'out':'sum.a'},[])

    def test_duplicate_edge_aliases_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            declared_region(branched(),['left'],{'a':'left.x','b':'left.x'},{'out':'sum.a'},['W'])

    def test_skip_edge_cannot_be_omitted(self):
        d=branched().to_dict();d['graph']['nodes'][2]['inputs']['b']='input:x';a=TensorArtifact(d)
        with self.assertRaisesRegex(ValueError,'inputs'):
            declared_region(a,['left','sum'],{'obs':'left.x'},{'out':'output.y'},['W'])

    def test_empty_unknown_and_duplicate_region_nodes_rejected(self):
        for ids in ([],['missing'],['left','left']):
            with self.subTest(ids=ids),self.assertRaises(ValueError):region(branched(),ids)

    def test_forged_region_aliases_are_revalidated_at_edit(self):
        a=branched();cut=left_region(a);cut['inputs']={}
        with self.assertRaisesRegex(ValueError,'region inputs'):
            replace_region(a,cut,linear(),prefix='r',inputs={},outputs={'contribution':'y'},budget=GraphBudget())


@unittest.skipUnless(HAS_NUMPY,'optional NumPy unavailable')
class SurgeryTests(unittest.TestCase):
    def test_replacement_preserves_shared_storage_and_unrelated_output(self):
        a=branched();h=a.sha256
        b,r=replace_region(a,left_region(a),linear(),prefix='repair',inputs={'x':'obs'},outputs={'contribution':'y'},budget=GraphBudget())
        self.assertEqual(a.sha256,h);self.assertEqual(b.trace({'x':[2]})['y'],[Q(10)])
        self.assertEqual(b.trace({'x':[2]})['keep'],a.trace({'x':[2]})['keep'])
        self.assertEqual(b.to_dict()['tensors']['W'],a.to_dict()['tensors']['W'])
        self.assertEqual(r['removed_tensors'],['lb'])
        delta=diff(a,b)['graph'];self.assertIn('right',delta['unaffected_dependency_nodes'])
        self.assertNotIn('keep',delta['potentially_affected_outputs'])

    def test_complete_output_fanout_replacement(self):
        a=branched();cut=declared_region(a,['right'],{'obs':'right.x'},{'keep':'output.keep','sum':'sum.b'},['W'])
        b,_=replace_region(a,cut,linear(4),prefix='repair',inputs={'x':'obs'},outputs={'keep':'y','sum':'y'},budget=GraphBudget())
        self.assertEqual(b.trace({'x':[1]})['keep'],[Q(4)]);self.assertEqual(b.trace({'x':[1]})['y'],[Q(7)])

    def test_unmapped_or_unknown_boundary_rejected(self):
        a=branched()
        for ins,outs in (({}, {'contribution':'y'}),({'x':'absent'},{'contribution':'y'}),({'x':'obs'},{})):
            with self.assertRaises(ValueError):replace_region(a,left_region(a),linear(),prefix='r',inputs=ins,outputs=outs,budget=GraphBudget())

    def test_declared_input_dropping_is_explicit(self):
        d=branched().to_dict();d['graph']['nodes'][2]['inputs']['b']='input:x';a=TensorArtifact(d)
        cut=declared_region(a,['left','sum'],{'obs':'left.x','skip':'sum.b'},{'out':'output.y'},['W'])
        with self.assertRaisesRegex(ValueError,'explicitly dropped'):
            replace_region(a,cut,linear(),prefix='r',inputs={'x':'obs'},outputs={'out':'y'},budget=GraphBudget())
        b,_=replace_region(a,cut,linear(),prefix='r',inputs={'x':'obs'},outputs={'out':'y'},drop_inputs=['skip'],budget=GraphBudget())
        self.assertEqual(b.trace({'x':[2]})['y'],[Q(6)])

    def test_stale_region_detects_added_consumer(self):
        a=branched();cut=left_region(a)
        b,_=redirect(a,'output.keep','node:left',budget=GraphBudget())
        with self.assertRaisesRegex(ValueError,'stale region'):
            replace_region(b,cut,linear(),prefix='r',inputs={'x':'obs'},outputs={'contribution':'y'},budget=GraphBudget())

    def test_patch_declared_input_width_cannot_be_ignored(self):
        with self.assertRaisesRegex(ValueError,'declared width'):
            transplant(branched(),linear(width=2),prefix='r',inputs={'x':'input:x'},budget=GraphBudget())

    def test_redirect_cycle_and_output_shape_fail_atomically(self):
        a=branched();h=a.sha256
        with self.assertRaisesRegex(ValueError,'cycle'):redirect(a,'left.x','node:sum',budget=GraphBudget())
        with self.assertRaises(ValueError):redirect(a,'output.keep','node:missing',budget=GraphBudget())
        self.assertEqual(a.sha256,h)

    def test_output_only_redirect_is_visible_in_diff(self):
        a=branched();b,_=redirect(a,'output.keep','node:left',budget=GraphBudget())
        delta=diff(a,b)['graph'];self.assertEqual(delta['potentially_affected_nodes'],[])
        self.assertEqual(delta['potentially_affected_outputs'],['keep'])

    def test_transplant_copies_by_default_and_preserves_internal_ties(self):
        a=branched();b,r=transplant(a,branched(),prefix='copy',inputs={'x':'input:x'},budget=GraphBudget())
        self.assertEqual(b.trace({'x':[2]})['y'],a.trace({'x':[2]})['y'])
        self.assertEqual(inspect(b)['graph_analysis']['parameter_uses']['copy__W'],['copy__left.weight','copy__right.weight'])
        self.assertIn('copy__W',r['imported']['copied_tensors'])

    def test_explicit_sharing_requires_identical_tensors(self):
        a=branched()
        b,r=transplant(a,linear(2),prefix='copy',inputs={'x':'input:x'},share={'W1':'W','b1':'rb'},budget=GraphBudget())
        self.assertEqual(r['imported']['copied_tensors'],[])
        self.assertIn('copy__a1.weight',inspect(b)['graph_analysis']['shared_parameters']['W'])
        with self.assertRaisesRegex(ValueError,'identical'):
            transplant(a,linear(3),prefix='copy',inputs={'x':'input:x'},share={'W1':'W'},budget=GraphBudget())

    def test_untie_clones_only_selected_parameter_reference(self):
        a=branched();b,_=untie(a,'left.weight','privateW',budget=GraphBudget())
        arrays=b.arrays(bits=True);arrays['privateW'][0,0]=np.float32(5).view(np.uint32);c=b.with_bits(arrays)
        self.assertEqual(c.trace({'x':[1]})['y'],[Q(8)])
        self.assertEqual(c.trace({'x':[1]})['keep'],[Q(2)])
        self.assertEqual(a.to_dict()['tensors']['W'],c.to_dict()['tensors']['W'])

    def test_residual_order_and_stored_scale(self):
        a=branched();b,r=add_residual(a,linear(3),prefix='boost',inputs={'x':'input:x'},output='y',into='output.y',scale=Q(1,2),budget=GraphBudget())
        self.assertEqual(b.trace({'x':[2]})['y'],[Q(12)])
        self.assertEqual(b.trace({'x':[2]})['keep'],[Q(4)])
        self.assertTrue(r['augmentation']);self.assertEqual(r['scale_stored'],'1/2')

    def test_residual_auxiliary_names_do_not_block_valid_modules(self):
        patch=model([node('scaled','identity',{'x':'input:x'}),node('sum','identity',{'x':'node:scaled'})],{'gain':[1]},outputs={'y':'node:sum'})
        b,r=add_residual(branched(),patch,prefix='r',inputs={'x':'input:x'},output='y',into='output.y',scale=Q(1),budget=GraphBudget())
        self.assertEqual(r['auxiliary_nodes'],['r__scaled_1','r__sum_1'])
        self.assertEqual(r['gain_tensor'],'r__gain_1');self.assertEqual(b.trace({'x':[1]})['y'],[Q(6)])

    def test_node_splitting_by_replacement_and_concat(self):
        base=to_dag(TensorArtifact.from_arrays({'W1':np.array([[1,2],[3,4]],np.float32),'b1':np.array([5,6],np.float32)},graph=TensorArtifact.dense([2,2],['linear']).to_dict()['graph']))
        patch=model([node('one','affine',{'x':'input:x'},params={'weight':'wa','bias':'ba'},attrs={'layout':'in_out'}),
                     node('two','affine',{'x':'input:x'},params={'weight':'wb','bias':'bb'},attrs={'layout':'in_out'}),
                     node('joined','concat',{'a':'node:one','b':'node:two'})],
                    {'wa':[[1],[3]],'ba':[5],'wb':[[2],[4]],'bb':[6]},inputs={'x':2},outputs={'y':'node:joined'})
        cut=declared_region(base,['a1'],{'obs':'a1.x'},{'out':'y.x'},[])
        b,_=replace_region(base,cut,patch,prefix='split',inputs={'x':'obs'},outputs={'out':'y'},budget=GraphBudget())
        for x in ([1,2],[-1,Q(1,2)],[0,0]):self.assertEqual(base.trace({'x':x})['y'],b.trace({'x':x})['y'])
        self.assertEqual(inspect(b)['graph_analysis']['parameters'],6)

    def test_topology_parameter_depth_and_operator_budgets(self):
        a=branched()
        for budget in (GraphBudget(max_nodes=3),GraphBudget(max_parameters=3),GraphBudget(max_depth=2),GraphBudget(allowed_ops=('affine','add'))):
            with self.subTest(budget=budget),self.assertRaises(ValueError):
                add_residual(a,linear(),prefix='r',inputs={'x':'input:x'},output='y',into='output.y',scale=Q(1),budget=budget)


@unittest.skipUnless(HAS_NUMPY,'optional NumPy unavailable')
class GraphEditorTests(unittest.TestCase):
    def run_plan(self,text,parent,patch=None):
        p=plan(text,parent=parent,artifacts={'patch':patch} if patch else {},budgets={'repair':GraphBudget()})
        return apply(p,parent=parent,expected_parent_id=parent.sha256),p

    def test_text_region_replacement_and_exact_replay(self):
        a=branched();q=linear();b,p=self.run_plan(source(a,replace_body(q),REGION),a,q)
        self.assertEqual(b.trace({'x':[2]})['y'],[Q(10)])
        self.assertEqual(p.to_dict()['graph_edits']['operations'][0]['operation'],'replace')

    def test_text_transplant_redirect_and_raw_edit(self):
        a=branched();q=linear()
        body=f'''transplant artifact("patch",sha256="{q.sha256}") as branch {{ inputs={{x:signal("input:x")}}; }}
        branch__W1[0,0]=4; redirect edge("output.keep")=signal("node:branch__y");'''
        b,_=self.run_plan(source(a,body),a,q)
        self.assertEqual(b.trace({'x':[2]})['keep'],[Q(8)])
        self.assertEqual(b.trace({'x':[2]})['y'],[Q(9)])

    def test_text_untie_followed_by_weight_edit(self):
        a=branched();b,p=self.run_plan(source(a,'untie parameter("left.weight") as privateW; privateW[:,:]=5;'),a)
        self.assertEqual(b.trace({'x':[1]})['keep'],[Q(2)])
        self.assertEqual(p.to_dict()['writes'][0]['parameter_users'],['left.weight'])

    def test_fixed_mode_raw_edits_report_all_shared_users(self):
        a=branched();s=source(a,'W[:,:]=3;').replace('mode=graph budget="repair"','mode=fixed')
        b,p=self.run_plan(s,a)
        self.assertEqual(p.to_dict()['writes'][0]['parameter_users'],['left.weight','right.weight'])
        self.assertEqual(diff(a,b)['graph']['potentially_affected_outputs'],['keep','y'])

    def test_late_graph_failure_does_not_publish_earlier_writes(self):
        a=branched();h=a.sha256
        with self.assertRaises(EditError):self.run_plan(source(a,'W[:,:]=9; redirect edge("left.x")=signal("node:sum");'),a)
        self.assertEqual(a.sha256,h)

    def test_budget_and_pinned_module_are_mandatory(self):
        a=branched();q=linear();s=source(a,replace_body(q),REGION)
        with self.assertRaisesRegex(ValueError,'budget'):plan(s,parent=a,artifacts={'patch':q})
        with self.assertRaisesRegex(ValueError,'identity'):plan(s,parent=a,artifacts={'patch':linear(7)},budgets={'repair':GraphBudget()})

    def test_graph_operations_rejected_in_fixed_mode(self):
        a=branched();s=source(a,'untie parameter("left.weight") as privateW;').replace('mode=graph budget="repair"','mode=fixed')
        with self.assertRaisesRegex(ValueError,'graph mode'):self.run_plan(s,a)

    def test_tampered_budget_receipt_and_candidate_rejected(self):
        a=branched();q=linear();_,p=self.run_plan(source(a,replace_body(q),REGION),a,q)
        for item in ('budget','candidate'):
            d=p.to_dict()
            if item=='budget':d['graph_edits']['budget_id']='0'*64
            else:d['candidate']['graph']['outputs']['y']='node:right'
            with self.assertRaisesRegex(ValueError,'replay'):apply(EditPlan(canonical(d)),parent=a,expected_parent_id=a.sha256)

    def test_text_residual(self):
        a=branched();q=linear();body=f'''add_residual artifact("patch",sha256="{q.sha256}") as boost {{
        inputs={{x:signal("input:x")}}; output=y; into=edge("output.y"); scale=1/2;
        }}'''
        b,_=self.run_plan(source(a,body),a,q);self.assertEqual(b.trace({'x':[2]})['y'],[Q(12)])

    def test_repeated_namespace_is_not_silently_overwritten(self):
        a=branched();q=linear();op=f'transplant artifact("patch",sha256="{q.sha256}") as copy {{ inputs={{x:signal("input:x")}}; }}'
        with self.assertRaisesRegex(ValueError,'collision'):self.run_plan(source(a,op+op),a,q)


@unittest.skipUnless(HAS_NUMPY,'optional NumPy unavailable')
class GraphEvidenceTests(unittest.TestCase):
    @unittest.skipUnless(HAS_Z3,'optional Z3 unavailable')
    def test_dag_claims_prove_and_false_claims_replay(self):
        a=branched();d=describe(a,'describe a { view left=tap("left")[0]; claim c semantics=real_graph { ensure y[0]==4*x[0]+1; ensure keep[0]==2*x[0]; ensure left==2*x[0]+1; } }')
        self.assertEqual(check(a,d)['status'],'PROVED')
        bad=describe(a,'describe a { claim c semantics=real_graph { ensure y[0]==0; } }')
        r=check(a,bad);self.assertEqual(r['status'],'REFUTED');self.assertTrue(r['results'][0]['witness_replayed'])
        self.assertIn('x',r['results'][0]['inputs'])

    @unittest.skipUnless(HAS_Z3,'optional Z3 unavailable')
    def test_multi_input_witness_and_slice_concat_encoder(self):
        a=model([node('joined','concat',{'a':'input:a','b':'input:b'}),node('part','slice',{'x':'node:joined'},attrs={'start':1,'stop':3,'step':1})],{},inputs={'a':2,'b':1},outputs={'y':'node:part'})
        d=describe(a,'describe a { claim c semantics=real_graph { ensure y[0]==a[1]; ensure y[1]==b[0]; } }')
        self.assertEqual(check(a,d)['status'],'PROVED')

    def test_tanh_graph_executes_but_formal_check_is_unsupported(self):
        a=to_dag(TensorArtifact.dense([1,1],['tanh']))
        d=describe(a,'describe a { claim c semantics=real_graph { ensure y[0]==0; } }')
        self.assertEqual(check(a,d)['status'],'UNSUPPORTED')
        self.assertEqual(a.trace({'x':[1]},semantics='numpy_f32')['y'],[0])

    def test_probe_hash_binds_values_not_just_named_input_keys(self):
        a=branched();one=probe(a,[{'x':[1]}]);two=probe(a,[{'x':[2]}])
        self.assertNotEqual(one['input_fixture_id'],two['input_fixture_id'])
        self.assertEqual(one['status'],'EMPIRICAL')

    def test_description_stales_after_surgery(self):
        a=branched();d=describe(a,'describe a { claim c semantics=real_graph { ensure y[0]==4*x[0]+1; } }')
        b,_=redirect(a,'output.y','node:right',budget=GraphBudget())
        self.assertEqual(check(b,d)['status'],'INVALID')

    def test_node_description_uses_actual_parameter_identities(self):
        a=branched();r=equations(a,node_id='left')
        self.assertEqual(r['node']['params']['weight'],'W');self.assertEqual(r['output_width'],1)

    @unittest.skipUnless(HAS_Z3,'optional Z3 unavailable')
    def test_claims_on_residual_candidate_check_full_edited_graph(self):
        a=branched();b,_=add_residual(a,linear(3),prefix='r',inputs={'x':'input:x'},output='y',into='output.y',scale=Q(1,2),budget=GraphBudget())
        d=describe(b,'describe a { claim c semantics=real_graph { ensure y[0]==11/2*x[0]+1; ensure keep[0]==2*x[0]; } }')
        self.assertEqual(check(b,d)['status'],'PROVED')


@unittest.skipUnless(HAS_NUMPY,'optional NumPy unavailable')
class GraphCLITests(unittest.TestCase):
    @unittest.skipUnless(HAS_Z3,'optional Z3 unavailable')
    def test_committed_graph_examples_and_demo(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'demo'
            r=subprocess.run([sys.executable,str(ROOT/'tools/demo_wsp_graph.py'),'--output',str(output)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['parent_unchanged'])
            self.assertEqual(set(summary['edits']),{'replace','residual','untie'})
            for result in summary['edits'].values():
                self.assertEqual(result['claims']['status'],'PROVED')
                self.assertTrue(result['false_claim']['results'][0]['witness_replayed'])

    def test_complete_cli_graph_workflow(self):
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);a=branched();q=linear();a.save(temp/'parent.json');q.save(temp/'patch.json')
            (temp/'edit.wspl').write_text(source(a,replace_body(q),REGION),encoding='utf-8')
            (temp/'budget.json').write_text(json.dumps(asdict(GraphBudget())),encoding='utf-8')
            env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
            def call(*args):
                r=subprocess.run([sys.executable,'-m','skillforge.wsp_editor',*map(str,args)],capture_output=True,text=True,env=env)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr);return json.loads(r.stdout)
            cut=call('boundary',temp/'parent.json','--nodes','left')
            self.assertIn('W',cut['shared_parameters'])
            call('plan',temp/'edit.wspl','--parent',temp/'parent.json','--module',f'patch={temp/"patch.json"}','--budget',f'repair={temp/"budget.json"}','--output',temp/'plan.json')
            call('apply',temp/'plan.json','--parent',temp/'parent.json','--output',temp/'candidate.json')
            info=call('describe',temp/'candidate.json','--node','fixed__a1')
            self.assertEqual(info['node']['op'],'affine')
            self.assertTrue(call('diff',temp/'parent.json',temp/'candidate.json')['graph']['changed'])
            dense=TensorArtifact.dense([1,1],['linear']);dense.save(temp/'dense.json')
            converted=call('graphify',temp/'dense.json','--output',temp/'dag.json')
            self.assertTrue(converted['tensor_storage_unchanged'])


if __name__=='__main__':unittest.main()
