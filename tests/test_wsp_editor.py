"""Editor conformance checks. No model requests, game episodes or training."""
import copy
from dataclasses import replace
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
HAS_NUMPY = importlib.util.find_spec('numpy') is not None
HAS_Z3 = importlib.util.find_spec('z3') is not None
if HAS_NUMPY:
    import numpy as np
    from skillforge.wsp_artifact import TensorArtifact, Bits, fraction, round_f32, inspect, diff, canonical, digest
    from skillforge.wsp_editor import plan, apply, EditPlan, EditError, import_legacy
    from skillforge.wsp_values import Limits, pack
    from skillforge.wsp_evidence import describe, Description, check, probe, equations


DEFINITION = '''
circuit Deadband(e:Real)->(u:Real) {
    p=relu(e-0.25); n=relu(-e-0.25); u=p-n;
}
'''
PLACEMENT = '''place Deadband(e=x[0]-x[1]) at {
 p=h1[0]; n=h1[1]; u=y[0];
} writes=replace_producers;'''
RAW = '''W1[:,:]=[[1,-1,0,0],[-1,1,0,0]];
b1[:]=[-0.25,-0.25,0,0]; W2[:,:]=[[1],[-1],[0],[0]]; b2[:]=0;'''
RESPONSE = '''describe candidate {
view positive = h1[0]; view raw_positive = pre(h1[0]);
hypothesis "positive represents error above threshold";
claim response semantics=real_graph {
 assume x[0] in [-1,1] and x[1] in [-1,1];
 ensure y[0]==relu(x[0]-x[1]-0.25)-relu(x[1]-x[0]-0.25);
}
}'''


def source(body='', *, widths='[2,4,1]', acts='[relu,linear]', layout='in_out', profile='exact_to_storage', declarations='', after='', parent=None):
    target = f'artifact("fixture",sha256="{parent.sha256}")' if parent else f'dense(widths={widths},activations={acts},layout={layout},storage=f32,init=zeros)'
    return f'wspl 0.2; target control={target}; {declarations}\nedit control as candidate mode=fixed numeric={profile} {{ {body} }}\n{after}'


def run(text, parent=None, **kwargs):
    p=plan(text,parent=parent,**kwargs)
    original=TensorArtifact(p.to_dict()['parent'])
    return apply(p,parent=original,expected_parent_id=original.sha256),p


@unittest.skipUnless(HAS_NUMPY,'optional editor NumPy unavailable')
class StorageTests(unittest.TestCase):
    def test_direct_rounding_avoids_double_rounding(self):
        q=Q(1)+Q(1,2**24)+Q(1,2**80)
        self.assertEqual(round_f32(q),0x3f800001)
        self.assertEqual(struct.unpack('<I',struct.pack('<f',float(q)))[0],0x3f800000)

    def test_subnormal_ties_and_overflow(self):
        self.assertEqual(round_f32(Q(1,2**150)),0)
        self.assertEqual(round_f32(Q(1,2**150)+Q(1,2**200)),1)
        self.assertEqual(round_f32(Q(3,2**150)),2)
        self.assertEqual(round_f32(-Q(1,2**200)),0x80000000)
        maximum=(Q(2)-Q(1,2**23))*2**127
        self.assertEqual(round_f32(maximum),0x7f7fffff)
        with self.assertRaisesRegex(ValueError,'overflow'): round_f32(maximum+2**103)

    def test_random_finite_fp32_values_roundtrip(self):
        rng=random.Random(3107)
        for _ in range(1500):
            b=rng.getrandbits(32)
            if b & 0x7f800000 == 0x7f800000 or b & 0x7fffffff == 0: continue
            self.assertEqual(round_f32(fraction(b)),b)

    def test_storage_copies_cannot_mutate_parent(self):
        a=TensorArtifact.dense([2,4,1],['relu','linear']); h=a.sha256
        a.arrays()['W1'][:]=9
        d=a.to_dict();d['tensors']['W1']['bits'][0]='3f800000'
        self.assertEqual(a.sha256,h)
        with self.assertRaises(Exception): a._text='changed'

    def test_opaque_named_rank_three_tensor_is_editable(self):
        a=TensorArtifact.from_arrays({'bank':np.zeros((2,3,4),dtype=np.float32)})
        b,p=run(source('bank[-1,:,::-1]=[[1,2,3,4],[5,6,7,8],[9,10,11,12]];',parent=a),a)
        np.testing.assert_array_equal(b.arrays()['bank'][1,0],[4,3,2,1])
        self.assertFalse(inspect(b)['capabilities']['numpy_execution'])

    def test_aliasing_and_implicit_dtype_conversion_rejected(self):
        a=np.zeros((4,),dtype=np.float32)
        with self.assertRaisesRegex(ValueError,'shared'): TensorArtifact.from_arrays({'a':a,'b':a[1:]})
        with self.assertRaisesRegex(ValueError,'float32'): TensorArtifact.from_arrays({'a':a.astype(np.float64)})

    def test_no_hidden_magnitude_cap_in_exact_profile(self):
        b,_=run(source('W1[0,0]=1e10;'))
        self.assertEqual(b.arrays()['W1'][0,0],np.float32(1e10))

    def test_bits_signed_zero_and_copy_are_preserved(self):
        b,_=run(source('W1[0,0]=f32bits(0x80000000); W1[1,0]=copy(W1[0,0]);'))
        self.assertEqual(list(b.arrays(bits=True)['W1'][:,0]),[0x80000000,0x80000000])
        self.assertEqual(diff(TensorArtifact.dense([2,4,1],['relu','linear']),b)['changed_elements'],2)

    def test_nonfinite_bit_patterns_and_overflow_rejected(self):
        for expr in ('f32bits(0x7f800000)','f32bits(0x7fc00001)','1e100'):
            with self.subTest(expr=expr), self.assertRaises(ValueError): plan(source(f'W1[0,0]={expr};'))

    def test_out_in_layout_and_serialization(self):
        a,_=run(source('W1[:,:]=[[1,-1],[-1,1],[0,0],[0,0]]; b1[:]=[-0.25,-0.25,0,0]; W2[:,:]=[[1,-1,0,0]];',layout='out_in'))
        self.assertEqual(a.trace([Q(1),Q(0)])['y'],[Q(3,4)])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.json';a.save(p)
            self.assertEqual(TensorArtifact.load(p).sha256,a.sha256)
            with self.assertRaises(FileExistsError):a.save(p)


@unittest.skipUnless(HAS_NUMPY,'optional editor NumPy unavailable')
class EditingTests(unittest.TestCase):
    def test_arbitrary_complete_target_assignment(self):
        b,p=run(source('W1[:,:]=[[1,2,3,4],[5,6,7,8]]; b1[:]=[9,10,11,12]; W2[:,:]=[[13],[14],[15],[16]]; b2[:]=17;'))
        self.assertEqual(diff(TensorArtifact(p.to_dict()['parent']),b)['changed_elements'],17)
        self.assertEqual(b.trace([Q(1),Q(1)])['y'],[Q(1163)])

    def test_slices_broadcast_and_augmented_arithmetic(self):
        b,_=run(source('W1[:,::2]=[1,2]; W1[-1,:]+=1; W1[:,0]*=2; W1[0,2]-=1;'))
        np.testing.assert_array_equal(b.arrays()['W1'],[[2,0,1,0],[4,1,3,1]])

    def test_overlapping_copy_reads_statement_snapshot(self):
        b,_=run(source('b1[:]=[1,2,3,4]; b1[1:]=copy(b1[:-1]);'))
        np.testing.assert_array_equal(b.arrays()['b1'],[1,1,2,3])

    def test_parent_reads_immutable_original(self):
        a,_=run(source('b1[:]=[1,2,3,4];'))
        b,_=run(source('b1[:]=9; b1[:]=parent.b1;',parent=a),a)
        self.assertEqual(a.sha256,b.sha256)

    def test_construction_constants_loops_and_matrix_operations(self):
        b,p=run(source('for i in range(4) { b1[i]=i/2; } W1[:,:]=transpose(reshape(v,[4,2])); W2[:,:]=full([4,1],2);',declarations='const v=[1,2,3,4,5,6,7,8];'))
        np.testing.assert_array_equal(b.arrays()['W1'],[[1,3,5,7],[2,4,6,8]])
        np.testing.assert_array_equal(b.arrays()['b1'],[0,.5,1,1.5])
        self.assertEqual(p.to_dict()['resources']['loop_iterations'],4)

    def test_low_rank_bound_update_is_identity_pinned(self):
        u=np.array([[Q(1)],[Q(2)]],dtype=object);v=np.array([[Q(3),Q(4),Q(5),Q(6)]],dtype=object)
        declarations=f'const U=tensor("u",sha256="{digest(pack(u))}"); const V=tensor("v",sha256="{digest(pack(v))}");'
        b,_=run(source('W1[:,:]+=U@V;',declarations=declarations),bindings={'u':u,'v':v})
        np.testing.assert_array_equal(b.arrays()['W1'],[[3,4,5,6],[6,8,10,12]])
        with self.assertRaisesRegex(ValueError,'identity'):plan(source('W1[:,:]+=U@V;',declarations=declarations),bindings={'u':u+1,'v':v})

    def test_rounding_report_and_numeric_scientific_constants(self):
        b,p=run(source('W1[0,0]=1/10; W1[1,0]=2.5e-1;'))
        self.assertEqual(p.to_dict()['rounding']['changed'],1)
        self.assertEqual(p.to_dict()['rounding']['examples'][0]['requested'],'1/10')
        self.assertEqual(b.arrays()['W1'][1,0],.25)

    def test_late_error_is_atomic(self):
        a=TensorArtifact.dense([2,4,1],['relu','linear']);h=a.sha256
        with self.assertRaises(EditError) as raised:plan(source('W1[:,:]=1; b1[99]=0;',parent=a),parent=a)
        self.assertEqual(a.sha256,h)
        self.assertTrue(raised.exception.diagnostic['parent_unchanged'])
        self.assertIsNotNone(raised.exception.diagnostic['source_span'])

    def test_empty_and_clipped_slices_visible(self):
        _,p=run(source('b1[9:20]=3; b1[-99:99]=0;'))
        self.assertTrue(p.to_dict()['writes'][0]['empty'])
        self.assertTrue(p.to_dict()['writes'][1]['clipped_slice'])
        self.assertEqual(p.to_dict()['status'],'NO_CHANGE')

    def test_stale_parent_rejected(self):
        p=plan(source('W1[0,0]=1;'));other,_=run(source('b1[0]=2;'))
        with self.assertRaisesRegex(ValueError,'parent'):apply(p,parent=other,expected_parent_id=other.sha256)

    def test_tampered_plan_is_replayed(self):
        p=plan(source('W1[0,0]=1;'));d=p.to_dict();a=TensorArtifact(d['parent'])
        d['candidate']['tensors']['W1']['bits'][0]='40000000'
        with self.assertRaisesRegex(ValueError,'replay'):apply(EditPlan(canonical(d)),parent=a,expected_parent_id=a.sha256)

    def test_plan_save_load_and_refuse_overwrite(self):
        p=plan(source('W1[0,0]=1;'))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'p.json';p.save(path)
            self.assertEqual(EditPlan.load(path).sha256,p.sha256)
            with self.assertRaises(FileExistsError):p.save(path)
            d=json.loads(path.read_text());d['body']['numeric_profile']='changed';path.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'identity'):EditPlan.load(path)

    def test_resource_guards_before_expansion(self):
        for body in ('for i in range(1000000000) { b1[0]=0; }','W1[:,:]=zeros([1000000000,1000000000]);','b1[0]=1e999999;'):
            with self.subTest(body=body),self.assertRaises(ValueError):plan(source(body))
        with self.assertRaisesRegex(ValueError,'statement'):plan(source('for i in range(4) { b1[i]=0; }'),limits=Limits(max_statements=2))

    def test_injection_unknown_calls_and_shape_mismatch(self):
        for body in ('W1[0,0]=__import__("os");','parent.W1[0,0]=1;','W1[:,:]=[[1,2],[3]];','b1[::0]=1;','b1[:]=[1,2,3];'):
            with self.subTest(body=body),self.assertRaises(ValueError):plan(source(body))

    def test_comments_do_not_execute_and_declarations_are_unique(self):
        b,_=run(source('W1[0,0]=1; // comment\n # comment\n b1[0]=2;'))
        self.assertEqual(b.arrays()['b1'][0],2)
        with self.assertRaisesRegex(ValueError,'duplicate'):plan(source('',declarations='const k=1; const k=2;'))


@unittest.skipUnless(HAS_NUMPY,'optional editor NumPy unavailable')
class CircuitTests(unittest.TestCase):
    def test_placed_and_raw_networks_have_identical_all_bits(self):
        a,_=run(source(PLACEMENT,declarations=DEFINITION));b,_=run(source(RAW))
        self.assertEqual(a.sha256,b.sha256)
        for x in (Q(-1),Q(-1,4),Q(0),Q(1,4),Q(1)):
            for y in (Q(-1),Q(0),Q(1)):
                expected=max(Q(0),x-y-Q(1,4))-max(Q(0),y-x-Q(1,4))
                self.assertEqual(a.trace([x,y])['y'],[expected])

    def test_replaces_fanin_and_bias_preserves_other_storage(self):
        parent,_=run(source('W1[:,:]=7; b1[:]=8; W2[:,:]=9; b2[:]=10;'))
        child,_=run(source(PLACEMENT,declarations=DEFINITION,parent=parent),parent)
        np.testing.assert_array_equal(child.arrays()['W1'][:,2:],parent.arrays()['W1'][:,2:])
        np.testing.assert_array_equal(child.arrays()['b1'][2:],parent.arrays()['b1'][2:])
        np.testing.assert_array_equal(child.arrays()['W2'][:,0],[1,-1,0,0])
        self.assertEqual(child.arrays()['b2'][0],0)

    def test_external_consumers_reported_without_deleting_edges(self):
        parent,_=run(source('W2[:,1]=3;',widths='[2,4,2]'))
        child,p=run(source(PLACEMENT,declarations=DEFINITION,parent=parent),parent)
        self.assertEqual(len(p.to_dict()['placements'][0]['external_consumers']),2)
        np.testing.assert_array_equal(child.arrays()['W2'][:,1],[3,3,3,3])

    def test_activation_mismatch_does_not_insert_clamp(self):
        with self.assertRaisesRegex(ValueError,'needs linear'):plan(source(PLACEMENT.replace('u=y[0]','u=h1[2]'),declarations=DEFINITION))

    def test_skip_dependencies_do_not_allocate_hidden_relays(self):
        with self.assertRaisesRegex(ValueError,'unavailable edges'):plan(source(PLACEMENT,declarations=DEFINITION.replace('u=p-n','u=p-n+e')))

    def test_circuits_cannot_capture_undeclared_target_signals(self):
        with self.assertRaisesRegex(ValueError,'declared ports'):
            plan(source(PLACEMENT,declarations=DEFINITION.replace('e-0.25','x[0]-0.25')))

    def test_duplicate_and_missing_placement_rejected(self):
        for body in (PLACEMENT+PLACEMENT,PLACEMENT.replace('n=h1[1];',''),PLACEMENT.replace('n=h1[1]','n=h1[0]')):
            with self.subTest(body=body),self.assertRaises(ValueError):plan(source(body,declarations=DEFINITION))

    def test_raw_edit_after_placement_marks_description_stale(self):
        _,p=run(source(PLACEMENT+'W1[0,0]=2;',declarations=DEFINITION))
        self.assertTrue(p.to_dict()['placements'][0]['later_edits_may_invalidate'])

    def test_refinements_recorded_as_unverified_not_assumed(self):
        _,p=run(source(PLACEMENT,declarations=DEFINITION.replace('e:Real','e:Real[0,1]')))
        obligation=p.to_dict()['placements'][0]['refinement_obligations'][0]
        self.assertEqual(obligation['status'],'UNVERIFIED');self.assertFalse(obligation['used_for_lowering'])


@unittest.skipUnless(HAS_NUMPY,'optional editor NumPy unavailable')
class LegacyTests(unittest.TestCase):
    def test_valid_legacy_transcripts_are_bitwise_compatible(self):
        from skillforge.native_weight_edits import NativeEditedPolicy,NAMES,statements
        import re
        for path in sorted((ROOT/'artifacts/astra_breakout_native_v2').glob('draw_*.answer.md')):
            before=hashlib.sha256(path.read_bytes()).hexdigest();answer=path.read_text(encoding='utf-8')
            historical=NativeEditedPolicy();receipt=historical.apply_answer(answer)
            self.assertEqual(receipt['skipped'],0)
            block=re.findall(r'```edits[^\S\r\n]*\r?\n(.*?)```',answer,re.S)[-1]
            body=';\n'.join(statements(block))+';'
            new,_=run(source(body,widths='[600,32,32,6]',acts='[tanh,tanh,linear]',profile='native_numpy_v1'))
            for name,key in NAMES.items():
                np.testing.assert_array_equal(new.arrays(bits=True)[name],historical.weights[key].view(np.uint32))
            self.assertEqual(before,hashlib.sha256(path.read_bytes()).hexdigest())

    def test_legacy_skip_policy_explicit_new_transactions_strict(self):
        artifact,receipt=import_legacy('```edits\nW1[0,0]=1\nW1[9999,0]=2\nb1[0]=3\n```')
        self.assertEqual(receipt['skipped'],1)
        self.assertEqual(artifact.arrays()['b1'][0],3)
        with self.assertRaises(ValueError):plan(source('W1[0,0]=1; W1[9999,0]=2; b1[0]=3;',profile='native_numpy_v1'))

    def test_native_profile_keeps_numeric_limits(self):
        for body in ('W1[0,0]=1e2;','W1[0,0]=1000001;'):
            with self.assertRaises(ValueError):plan(source(body,profile='native_numpy_v1'))


@unittest.skipUnless(HAS_NUMPY,'optional editor NumPy unavailable')
class EvidenceTests(unittest.TestCase):
    def test_description_is_separate_identity_and_immutable(self):
        artifact,_=run(source(RAW));identity=artifact.sha256
        d=describe(artifact,RESPONSE)
        e=describe(artifact,RESPONSE.replace('positive represents','hypothetically represents'))
        self.assertEqual(artifact.sha256,identity);self.assertNotEqual(d.sha256,e.sha256)
        self.assertEqual(d.to_dict()['executable_changes'],0)

    def test_description_only_plan_changes_zero_weights(self):
        parent,_=run(source(RAW))
        s=f'wspl 0.2; target candidate=artifact("fixture",sha256="{parent.sha256}"); '+RESPONSE
        child,p=run(s,parent)
        self.assertEqual(child.sha256,parent.sha256);self.assertEqual(p.to_dict()['writes'],[])

    @unittest.skipUnless(HAS_Z3,'optional Z3 unavailable')
    def test_frozen_equation_proves_and_false_claim_replays(self):
        artifact,_=run(source(RAW))
        self.assertEqual(check(artifact,describe(artifact,RESPONSE))['status'],'PROVED')
        bad=RESPONSE.replace('ensure y[0]==relu(x[0]-x[1]-0.25)-relu(x[1]-x[0]-0.25);','ensure y[0]==0;')
        r=check(artifact,describe(artifact,bad))
        self.assertEqual(r['status'],'REFUTED');self.assertTrue(r['results'][0]['witness_replayed'])

    @unittest.skipUnless(HAS_Z3,'optional Z3 unavailable')
    def test_preactivation_taps_and_views(self):
        artifact,_=run(source(RAW))
        d=describe(artifact,'''describe a { view p=pre(h1[0]); claim c semantics=real_graph {
        ensure p==x[0]-x[1]-0.25; ensure h1[0]>=0; } }''')
        self.assertEqual(check(artifact,d)['status'],'PROVED')

    @unittest.skipUnless(HAS_Z3,'optional Z3 unavailable')
    def test_vacuity_and_solver_budget_are_not_proofs(self):
        artifact,_=run(source(RAW))
        d=describe(artifact,'describe a { claim c semantics=real_graph { assume x[0]>1 and x[0]<0; ensure y[0]==0; } }')
        self.assertEqual(check(artifact,d)['status'],'INVALID')
        self.assertEqual(check(artifact,describe(artifact,RESPONSE),max_parameters=1)['status'],'UNKNOWN')
        self.assertEqual(check(artifact,d,timeout_ms=0)['status'],'INVALID')

    def test_stale_artifact_description_rejected(self):
        a,_=run(source(RAW));b,_=run(source(RAW+'b2[0]=1;'))
        self.assertEqual(check(b,describe(a,RESPONSE))['status'],'INVALID')

    def test_tanh_editing_and_execution_do_not_require_formal_support(self):
        artifact,_=run(source('W1[:,:]=1;',acts='[tanh,linear]'))
        d=describe(artifact,'describe a { claim c semantics=real_graph { ensure y[0]==0; } }')
        self.assertEqual(check(artifact,d)['status'],'UNSUPPORTED')
        self.assertEqual(artifact.trace([1,1],semantics='numpy_f32')['y'],[0])

    def test_equations_come_from_actual_coefficients(self):
        artifact,_=run(source(RAW+'W1[0,0]=2;'))
        first=equations(artifact,layer=1,limit=1)['equations'][0]
        self.assertEqual(first['terms'][0]['coefficient'],'2')

    def test_probe_is_empirical_not_proof(self):
        artifact,_=run(source(RAW));r=probe(artifact,[[Q(0),Q(0)],[Q(1),Q(0)]])
        self.assertEqual(r['status'],'EMPIRICAL');self.assertFalse(r['universal_claim'])
        self.assertEqual(r['taps'][1]['y'],['3/4'])

    def test_description_cannot_write_storage_or_reference_bad_tap(self):
        a,_=run(source(RAW))
        for text in ('describe a { W1[0,0]=1; }','describe a { view nope=h1[99]; }'):
            with self.assertRaises(ValueError):describe(a,text)

    def test_description_file_identity_and_origin(self):
        a,_=run(source(RAW));d=describe(a,RESPONSE,origin='external_requirement')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'d.json';d.save(p)
            self.assertEqual(Description.load(p).sha256,d.sha256)
            self.assertEqual(Description.load(p).to_dict()['origin'],'external_requirement')


@unittest.skipUnless(HAS_NUMPY,'optional editor NumPy unavailable')
class CLITests(unittest.TestCase):
    def test_cli_plan_apply_inspect_and_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);s=tmp/'source.wspl';p=tmp/'plan.json';a=tmp/'artifact.json'
            s.write_text(source(PLACEMENT,declarations=DEFINITION),encoding='utf-8')
            import os
            env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
            def invoke(*args):
                result=subprocess.run([sys.executable,'-m','skillforge.wsp_editor',*map(str,args)],capture_output=True,text=True,env=env)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                return json.loads(result.stdout)
            invoke('plan',s,'--output',p);invoke('apply',p,'--output',a)
            data=invoke('inspect',a,'--tensor','W1','--limit',2)
            self.assertEqual(data['page']['total'],8)
            self.assertEqual(invoke('diff',a,a)['status'],'NO_CHANGE')
            eq=invoke('describe',a,'--layer',1,'--limit',1)
            self.assertEqual(eq['equations'][0]['bias'],'-1/4')
            d=tmp/'description.json';r=tmp/'checked.json';probed=tmp/'probe.json'
            invoke('describe',a,'--source',ROOT/'examples/wsp_editor/response.description.wspl','--output',d)
            result=invoke('check',a,d,'--output',r)
            self.assertEqual(result['status'],'PROVED' if HAS_Z3 else 'UNSUPPORTED')
            finite=invoke('probe',a,ROOT/'examples/wsp_editor/inputs.json','--output',probed)
            self.assertEqual(finite['status'],'EMPIRICAL')
            self.assertEqual(finite['samples'],5)

    def test_committed_raw_and_placed_examples_match(self):
        examples=ROOT/'examples/wsp_editor'
        placed,_=run((examples/'deadband.wspl').read_text(encoding='utf-8'))
        direct,_=run((examples/'deadband_raw.wspl').read_text(encoding='utf-8'))
        self.assertEqual(placed.sha256,direct.sha256)


if __name__=='__main__':unittest.main()
