"""Engineering semantics checks; no physical experiment or model fitting."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from skillforge import Expr,parse,evaluate_closed_trace,RMVReference


def ap(name):return dict(op='ap',name=name)
def op(name,*args):return dict(op=name,args=list(args))
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))


class FiniteTraceTests(unittest.TestCase):
    def test_next_and_weak_next_at_terminal_position(self):
        trace=[dict(p=True)]
        self.assertFalse(evaluate_closed_trace(op('X',ap('p')),trace))
        self.assertTrue(evaluate_closed_trace(op('Xw',ap('p')),trace))

    def test_next_observes_the_following_position(self):
        self.assertTrue(evaluate_closed_trace(op('X',ap('p')),[dict(p=False),dict(p=True)]))

    def test_eventual_false_prefix_is_not_a_definitive_failure(self):
        f=op('F',ap('p'))
        self.assertFalse(evaluate_closed_trace(f,[dict(p=False)]))
        self.assertTrue(evaluate_closed_trace(f,[dict(p=False),dict(p=True)]))

    def test_global_and_boolean_connectives(self):
        f=op('G',op('implies',ap('p'),op('not',ap('q'))))
        self.assertTrue(evaluate_closed_trace(f,[dict(p=True,q=False),dict(p=False,q=True)]))
        self.assertFalse(evaluate_closed_trace(f,[dict(p=True,q=True)]))

    def test_strong_until_requires_a_witness(self):
        f=op('U',ap('p'),ap('q'))
        self.assertFalse(evaluate_closed_trace(f,[dict(p=True,q=False)]*3))
        self.assertTrue(evaluate_closed_trace(f,[dict(p=True,q=False),dict(p=False,q=True)]))
        self.assertFalse(evaluate_closed_trace(f,[dict(p=False,q=False),dict(p=True,q=True)]))

    def test_until_can_be_satisfied_immediately(self):
        self.assertTrue(evaluate_closed_trace(op('U',ap('p'),ap('q')),[dict(p=False,q=True)]))

    def test_order_requires_a_later_witness(self):
        f=op('F',op('and',ap('grasp'),op('X',op('F',ap('place')))))
        self.assertFalse(evaluate_closed_trace(f,[dict(grasp=True,place=True)]))
        self.assertTrue(evaluate_closed_trace(f,[dict(grasp=True,place=False),dict(grasp=False,place=True)]))

    def test_task_example_and_independent_counterexamples(self):
        task=read('examples/pick_place.task.json');trace=read('examples/pick_place.trace.json')
        self.assertTrue(evaluate_closed_trace(task['formula'],trace))
        bad=copy.deepcopy(trace);bad[2]['at_target']=False
        self.assertFalse(evaluate_closed_trace(task['formula'],bad))
        bad=copy.deepcopy(trace);bad[0]['collision']=True
        self.assertFalse(evaluate_closed_trace(task['formula'],bad))

    def test_missing_or_fuzzy_propositions_are_rejected(self):
        for trace in ([{}],[dict(p=.7)]):
            with self.assertRaises(ValueError):evaluate_closed_trace(ap('p'),trace)
        with self.assertRaises(ValueError):evaluate_closed_trace(dict(op='true'),[])

    def test_invalid_ast_fields_arity_and_operator_rejected(self):
        for node in (dict(op='and',args=[]),dict(op='X',args=[]),dict(op='magic'),dict(op='ap',name='p',weight=1)):
            with self.subTest(node=node),self.assertRaises(ValueError):parse(node)
        with self.assertRaises(ValueError):Expr('X')

    def test_roundtrip_atoms_and_limits(self):
        node=op('and',ap('p'),op('F',ap('q')));f=parse(node)
        self.assertEqual(f.to_dict(),node);self.assertEqual(f.atoms,{'p','q'})
        with self.assertRaises(ValueError):parse(node,max_nodes=2)
        with self.assertRaises(ValueError):parse(node,max_depth=0)

    def test_temporal_dual_on_small_closed_traces(self):
        for values in ((False,),(True,),(False,True),(True,False,True)):
            trace=[dict(p=x) for x in values]
            self.assertEqual(evaluate_closed_trace(op('not',op('G',ap('p'))),trace),evaluate_closed_trace(op('F',op('not',ap('p'))),trace))


class RMVTests(unittest.TestCase):
    def setUp(self):self.graph=read('examples/lift_gate.rmv.json')

    def test_known_grounded_calculation(self):
        r=RMVReference(self.graph).evaluate(dict(height_error_m=.05,contact=1.))
        self.assertEqual(r['predicates'],[.5,1.]);self.assertEqual(r['clauses'],[.5]);self.assertAlmostEqual(r['commands'][0],.1)

    def test_contact_gate_changes_output(self):
        r=RMVReference(self.graph)
        self.assertEqual(r.evaluate(dict(height_error_m=.1,contact=0.))['commands'],[0.])
        self.assertEqual(r.evaluate(dict(height_error_m=.1,contact=1.))['commands'],[.2])

    def test_normalization_and_final_readout_clipping_included(self):
        graph=copy.deepcopy(self.graph);graph['readout_weights']=[[-3.]];graph['readout_bias']=[1.]
        r=RMVReference(graph).evaluate(dict(height_error_m=100.,contact=1.))
        self.assertEqual(r['predicates'],[1,1]);self.assertEqual(r['commands'],[-1.])

    def test_bad_shapes_and_logical_signs_rejected(self):
        for key,value in [('and_weights',[[-1.,1.]]),('literal_sources',[dict(input=5,negated=False)]),('readout_weights',[[1.,2.]])]:
            graph=copy.deepcopy(self.graph);graph[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):RMVReference(graph)

    def test_invalid_ranges_duplicate_channels_and_nonfinite_weights_rejected(self):
        for change in ('range','duplicate','nonfinite'):
            g=copy.deepcopy(self.graph)
            if change=='range':g['inputs']['upper'][0]=0
            elif change=='duplicate':g['inputs']['names'][1]=g['inputs']['names'][0]
            else:g['readout_bias'][0]=float('nan')
            with self.subTest(change=change),self.assertRaises(ValueError):RMVReference(g)

    def test_observation_names_and_numeric_types_are_strict(self):
        r=RMVReference(self.graph)
        for obs in (dict(height_error_m=.1),dict(height_error_m=.1,contact=True),dict(height_error_m=float('inf'),contact=1.)):
            with self.assertRaises(ValueError):r.evaluate(obs)

    def test_positive_and_negative_literals_of_one_channel_can_coexist(self):
        g=copy.deepcopy(self.graph)
        g['literal_sources']=[dict(input=0,negated=False),dict(input=0,negated=True)]
        g['and_weights']=[[1.,0.],[0.,1.]];g['or_weights']=[[1.,0.],[0.,1.]]
        g['readout_weights']=[[1.,-1.]];g['readout_bias']=[0.]
        r=RMVReference(g)
        self.assertEqual(r.evaluate(dict(height_error_m=.1,contact=0.))['commands'],[1.])
        self.assertEqual(r.evaluate(dict(height_error_m=0.,contact=0.))['commands'],[-1.])


class ArtifactTests(unittest.TestCase):
    def test_schema_definitions_and_example_instances(self):
        from jsonschema import Draft202012Validator
        for name in ('ltlf','task','skill','rmv'):Draft202012Validator.check_schema(read('schemas/'+name+'.schema.json'))
        for schema,example in [('task','pick_place.task'),('skill','lift.skill'),('rmv','lift_gate.rmv')]:
            Draft202012Validator(read('schemas/'+schema+'.schema.json')).validate(read('examples/'+example+'.json'))

    def test_task_vocabulary_and_skill_normalization_are_consistent(self):
        task=read('examples/pick_place.task.json');observed=set(task['observable']);controlled=set(task['controllable'])
        self.assertTrue(observed.isdisjoint(controlled));self.assertLessEqual(parse(task['formula']).atoms,observed|controlled)
        skill=read('examples/lift.skill.json')
        self.assertEqual(len({x['name'] for x in skill['input_channels']}),12)
        self.assertTrue(all(c['lower']<c['upper'] for c in skill['input_channels']+skill['command_channels']))
        self.assertLessEqual(parse(skill['applicability']).atoms,set(skill['predicate_vocabulary']))

    def test_provenance_copies_match_recorded_hashes(self):
        for entry in read('provenance/index.json')['items']:
            data=(ROOT/entry['destination']).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(),entry['sha256'])

    def test_registry_entries_require_user_approval_provenance(self):
        registry=read('experiments/source_pins/approved.json');self.assertEqual(registry['schema'],1)
        for entry in registry['models'].values():
            self.assertEqual(entry['approved_by'],'user')
            self.assertEqual(hashlib.sha256((ROOT/entry['manifest']).read_bytes()).hexdigest(),entry['sha256'])


if __name__=='__main__':unittest.main()
