import copy
import unittest
import json
from pathlib import Path
import tempfile

from skillforge.controller_author import KINDS, construct, make_spec, teacher as block_teacher
from skillforge.executive_study import make_expanded_task, teacher, score
from skillforge.control_dsl import compile_program
from skillforge.wspl_language import compile_source, library_document, digest, tensor_from_module
from skillforge import unified_executive


class UnifiedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs={k:make_spec(k,20) for k in KINDS}
        cls.blocks={k:construct(block_teacher(s),s) for k,s in cls.specs.items()}
        cls.library=library_document(cls.blocks,cls.specs)

    def test_module_identity_preserved_for_all_families(self):
        for kind,spec in self.specs.items():
            compiled=compile_source('wspl 0.3;\n'+block_teacher(spec))
            self.assertEqual(compiled.modules['Block'].sha256,self.blocks[kind].sha256)

    def test_definition_edit_bridge_and_rounding_receipt(self):
        source='wspl 0.3;\n'+block_teacher(self.specs['move'])
        compiled=compile_source(source+'\ntarget base=compiled(Block);\nedit base as changed mode=fixed { b1[0]+=1/7; }')
        parent,receipt=tensor_from_module(compiled.modules['Block'])
        self.assertNotEqual(parent.sha256,compiled.candidate.sha256)
        self.assertEqual(receipt,compiled.provenance['conversion'])
        self.assertEqual(compiled.provenance['editor_plan_sha256'],compiled.edit_plan.sha256)
        self.assertEqual(receipt['contracts'],'not_transferred')
        self.assertGreater(receipt['rounded_coefficients'],0)

    def test_structural_typed_calls_match_legacy_without_text_translation(self):
        for n in (2,4,8):
            task=make_expanded_task(120+n,n)
            src=unified_executive.teacher(task,self.library)
            compiled=compile_source(src,libraries={'planar':self.library},objects=task['objects'],points=task['points'],limit=64)
            self.assertEqual(compiled.calls,compile_program(teacher(task),task['objects'],task['points'],64))

    def test_runtime_trace_and_scores_match_including_failure(self):
        for jam in (False,True):
            task=make_expanded_task(123,3,jam=jam)
            old=score(teacher(task),task,self.blocks,self.specs)
            new=unified_executive.score(unified_executive.teacher(task,self.library),task,self.blocks,self.specs)
            new.pop('compilation')
            self.assertEqual(old,new)

    def test_tampered_library_and_wrong_argument_types_rejected(self):
        task=make_expanded_task(333,2)
        src=unified_executive.teacher(task,self.library)
        bad=copy.deepcopy(self.library);bad['modules']['move']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'identity'):
            compile_source(src,libraries={'planar':bad},objects=task['objects'],points=task['points'])
        bad_src=src.replace('robot.move_to(home)','robot.grasp(home)')
        self.assertFalse(unified_executive.score(bad_src,task,self.blocks,self.specs)['constructed'])

    def test_unknown_calls_excess_calls_and_imports_rejected(self):
        task=make_expanded_task(1,2)
        src=unified_executive.teacher(task,self.library)
        for modified in (src.replace('transfer','teleport'),src.replace('robot.transfer','missing.transfer'),
                         src+' task Again {}',src.replace('wspl 0.3','wspl 99.0')):
            with self.assertRaises(ValueError):
                compile_source(modified,libraries={'planar':self.library},objects=task['objects'],points=task['points'])
        with self.assertRaisesRegex(ValueError,'budget'):
            compile_source(src,libraries={'planar':self.library},objects=task['objects'],points=task['points'],limit=1)

    def test_no_executable_python_or_silent_unlinked_module(self):
        for src in ['wspl 0.3; import os;', 'wspl 0.3; task T { robot.move_to(home); }',
                    'wspl 0.3;', 'wspl 0.3; target t=compiled(Missing);']:
            with self.assertRaises(ValueError):compile_source(src,points={'home'})

    def test_invalid_library_ports(self):
        bad=dict(self.blocks);bad['holding']=bad['move']
        with self.assertRaisesRegex(ValueError,'port'):library_document(bad,self.specs)

    def test_unified_namespace_rejects_shadowing(self):
        source='wspl 0.3;\n'+block_teacher(self.specs['move'])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            compile_source(source+'\ntarget Block=compiled(Block);')

    def test_contracts_retained_as_separate_unproved_source_claims(self):
        source='''wspl 0.3;
module Positive {
 input x: Real[-1,1] normalized_x;
 output y: Real[0,1] normalized_y;
 y=relu(x);
 ensures Bounded: forall inputs: y>=0;
}
target base=compiled(Positive);
edit base as changed mode=fixed { b1[0]=-10; }
'''
        compiled=compile_source(source)
        self.assertEqual(len(compiled.module_contracts['Positive']),1)
        self.assertEqual(compiled.edit_plan.to_dict()['descriptions'],[])
        self.assertEqual(compiled.provenance['conversion']['contracts'],'not_transferred')

    def test_reuse_checks_full_tasks_and_unique_replies(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);rows=[dict(id='a',task=dict(objects=['box']))]
            reply=dict(id='a',arm='trained',source='move_to home')
            (p/'tasks').write_text(json.dumps(rows[0])+'\n')
            (p/'replies').write_text(json.dumps(reply)+'\n')
            config=dict(previous_tasks=str(p/'tasks'),previous_replies=str(p/'replies'))
            self.assertTrue(unified_executive.reused_replies(config,rows)[0]['reused'])
            with self.assertRaisesRegex(ValueError,'content'):
                unified_executive.reused_replies(config,[dict(id='a',task={})])
            (p/'replies').write_text((json.dumps(reply)+'\n')*2)
            with self.assertRaisesRegex(ValueError,'identities'):
                unified_executive.reused_replies(config,rows)


if __name__=='__main__':unittest.main()
