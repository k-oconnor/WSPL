"""Reusable given decompositions and independent macro-world engineering cases."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from skillforge.hierarchy import Hierarchy
from skillforge.tabletop import build_case,tabletop_model
from skillforge.automata import compile_dfa
from skillforge.planning import FiniteModel,synthesize,execute
from skillforge.ltlf import evaluate_closed_trace


def suite():return json.loads((ROOT/'examples/hierarchical_tasks.json').read_text())


class HierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document=suite();cls.results={}
        for case in cls.document['cases']:
            fixture=build_case(cls.document,case['case_id']);dfa=compile_dfa(fixture['task']['formula'])
            model=FiniteModel.from_dict(fixture['model']);plan=synthesize(dfa,model,fixture['initial'])
            result=execute(plan,model,lambda s,a,o:o[-1]) if plan.realizable else None
            cls.results[case['case_id']]=(fixture,model,plan,result)

    def test_all_fixture_expectations_are_computed_independently(self):
        for case in self.document['cases']:
            fixture,model,plan,result=self.results[case['case_id']]
            self.assertEqual(plan.realizable,case['expected_realizable'],case['case_id'])
            if result:
                self.assertTrue(result['accepted']);self.assertTrue(evaluate_closed_trace(plan.dfa.formula,result['trace']))

    def test_named_subtasks_and_primitives_are_reused(self):
        fixture=self.results['load_two'][0];stats=fixture['statistics']
        self.assertEqual(stats['template_calls'],{'load_pair':1,'transfer':2,'pickup':2,'put_down':2,'park':1})
        self.assertEqual(stats['primitive_calls'],15);self.assertEqual(len(stats['primitive_skill_names']),8)
        self.assertEqual(stats['maximum_template_depth'],3)

    def test_repeated_binding_needs_three_distinct_pickup_witnesses(self):
        fixture,model,plan,result=self.results['round_trip']
        self.assertEqual(result['actions'].count('pickup_a'),3)
        self.assertEqual(fixture['statistics']['template_calls']['transfer'],3)
        self.assertEqual(fixture['statistics']['primitive_calls'],22)
        bad=copy.deepcopy(result['trace']);first=next(v for v in bad if v['picked_a']);first['picked_a']=False
        self.assertFalse(plan.dfa.accepts(bad))

    def test_dependency_is_in_world_not_an_ordered_root_instruction(self):
        case=next(c for c in self.document['cases'] if c['case_id']=='clear_then_retrieve')
        self.assertEqual(case['root']['op'],'all')
        fixture,model,plan,result=self.results[case['case_id']]
        self.assertLess(result['actions'].index('place_a_buffer'),result['actions'].index('pickup_b'))
        self.assertNotIn('pickup_b',model.transitions[fixture['initial']])

    def test_mid_task_failure_reuses_subtasks_without_restarting_finished_object(self):
        fixture,model,plan,result=self.results['second_object_recovery']
        self.assertTrue(any(v['grasp_failed'] for v in result['trace']))
        self.assertEqual(result['actions'].count('pickup_a'),1)
        self.assertEqual(result['actions'].count('pickup_b'),2)
        self.assertEqual(result['actions'][-1],'return_home')

    def test_terminal_state_matters_after_milestone_witnesses(self):
        fixture,model,plan,result=self.results['load_two'];bad=copy.deepcopy(result['trace'])
        bad[-1]['b_at_tray']=False
        self.assertFalse(plan.dfa.accepts(bad))

    def test_world_dynamics_do_not_depend_on_requested_task(self):
        a=self.results['transfer_one'][1];b=self.results['round_trip'][1]
        self.assertEqual(a.identity,b.identity)

    def test_cycles_unknown_calls_and_bad_bindings_are_rejected(self):
        document=copy.deepcopy(self.document)
        document['templates']['pickup']['body']={'op':'call','template':'transfer','bindings':{'object':'{object}','target':'tray'}}
        with self.assertRaises(ValueError):build_case(document,'transfer_one')
        h=Hierarchy(self.document['templates'])
        for node in [dict(op='call',template='missing',bindings={}),dict(op='call',template='pickup',bindings={})]:
            with self.assertRaises(ValueError):h.expand(node)

    def test_unsupported_nested_unordered_sequence_is_rejected(self):
        h=Hierarchy(self.document['templates']);root=dict(op='sequence',children=[dict(op='all',children=[dict(op='call',template='park',bindings={})])])
        with self.assertRaises(ValueError):h.outcome_words(root)

    def test_schema_and_case_ids(self):
        from jsonschema import Draft202012Validator
        schema=json.loads((ROOT/'schemas/hierarchy.schema.json').read_text());Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(self.document)
        self.assertEqual(len(self.results),7)

    def test_hierarchy_cli_exposes_scope_and_execution(self):
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
        p=subprocess.run([sys.executable,'-m','skillforge','hierarchy','examples/hierarchical_tasks.json','--case','load_two','--execute'],cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);r=json.loads(p.stdout)
        self.assertTrue(r['execution']['accepted']);self.assertEqual(r['hierarchy_statistics']['primitive_calls'],15)
        self.assertIn('No primitive hierarchical executor',r['fixture_metadata']['evaluation_scope'])


if __name__=='__main__':unittest.main()
