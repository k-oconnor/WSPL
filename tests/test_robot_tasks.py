"""Inspected specifications and labelled trace checks, never robot execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from skillforge.robot_tasks import RobotTaskSuite, read_bddl, audit_assets


def document(): return json.loads((ROOT/'examples/robot_task_suite_v1.json').read_text())


def good_trace(task):
    props = [e['id'] for e in task['milestones']] + task['terminal'] + task['invariants'] + [d['state'] for d in task['dwell']]
    stride = max([d['ticks'] for d in task['dwell']] + [0])+2
    trace = [{p: p not in {e['id'] for e in task['milestones']} for p in props}
             for _ in range(stride*len(task['milestones'])+2)]
    for i, row in enumerate(trace): row['time_s'] = i*.05
    for i, event in enumerate(task['milestones']): trace[1+i*stride][event['id']] = True
    return trace


class RobotTaskTests(unittest.TestCase):
    def setUp(self): self.doc=document(); self.suite=RobotTaskSuite(self.doc)

    def test_diversity_counts_readiness_and_snapshot(self):
        self.assertEqual(len(self.suite.task_ids),16)
        self.assertEqual({t['expected_grasp_occurrences'] for t in self.doc['tasks']},{1,2,3})
        for key in self.suite.task_ids:
            self.assertFalse(self.suite.readiness(key)['physical_runnable'])
            self.assertTrue(self.suite.readiness(key)['missing'])
        self.doc['tasks'][0]['bindings']['a']='made_up'
        self.assertNotEqual(self.doc['tasks'][0],self.suite.task('single_box'))

    def test_all_supplied_positive_label_fixtures(self):
        for key in self.suite.task_ids:
            with self.subTest(key=key):
                t=self.suite.task(key)
                self.assertTrue(self.suite.check_trace(key,good_trace(t))['accepted'])

    def test_wrong_order_and_coincident_milestones_fail(self):
        key='ordered_load';t=self.suite.task(key);trace=good_trace(t)
        a=next(i for i,r in enumerate(trace) if r['placed_a'])
        b=next(i for i,r in enumerate(trace) if r['picked_b'])
        trace[a]['placed_a']=False;trace[b]['picked_b']=False
        trace[a]['picked_b']=True;trace[b]['placed_a']=True
        self.assertFalse(self.suite.check_trace(key,trace)['accepted'])
        trace=good_trace(t);trace[a]['picked_b']=True;trace[b]['picked_b']=False
        self.assertFalse(self.suite.check_trace(key,trace)['accepted'])

    def test_repeated_pickup_needs_distinct_pulses_and_rejects_sticky_event(self):
        key='round_trip';trace=good_trace(self.suite.task(key))
        for row in trace: row['picked_a_middle']=False
        self.assertFalse(self.suite.check_trace(key,trace)['accepted'])
        trace=good_trace(self.suite.task(key));trace[2]['picked_a_first']=True
        self.assertFalse(self.suite.check_trace(key,trace)['accepted'])

    def test_undo_final_placement_and_forbidden_contact_fail(self):
        key='load_two';t=self.suite.task(key);trace=good_trace(t)
        trace[-1]['a_at_dst']=False
        self.assertIn('terminal:a_at_dst',self.suite.check_trace(key,trace)['violations'])
        trace=good_trace(t);trace[3]['no_forbidden_contact']=False
        self.assertFalse(self.suite.check_trace(key,trace)['accepted'])

    def test_wait_requires_twenty_consecutive_true_samples(self):
        key='hold_then_place';t=self.suite.task(key);trace=good_trace(t)
        self.assertTrue(self.suite.check_trace(key,trace)['accepted'])
        trace[8]['a_held']=False
        self.assertIn('dwell:a_held',self.suite.check_trace(key,trace)['violations'])
        trace=good_trace(t);place=next(i for i,r in enumerate(trace) if r['placed_a'])
        trace[place]['placed_a']=False;trace[3]['placed_a']=True
        self.assertIn('dwell:a_held',self.suite.check_trace(key,trace)['violations'])

    def test_missing_sensor_tick_unknown_value_and_budget(self):
        key='single_box';t=self.suite.task(key);trace=good_trace(t)
        trace[3]['time_s']+=.05
        with self.assertRaises(ValueError): self.suite.check_trace(key,trace)
        trace=good_trace(t);trace[0]['no_forbidden_contact']=None
        with self.assertRaises(ValueError): self.suite.check_trace(key,trace)
        trace=good_trace(t)
        while len(trace)<=t['max_motor_ticks']+1:
            row=copy.deepcopy(trace[-1]);row['time_s']=len(trace)*.05;trace.append(row)
        self.assertIn('motor_budget',self.suite.check_trace(key,trace)['violations'])

    def test_blocker_dependency_is_not_a_supplied_order_edge(self):
        t=self.suite.task('clear_then_fetch')
        self.assertNotIn(['placed_blocker','picked_a'],t['precedence'])
        self.assertIn('target_accessible_at_pickup',t['invariants'])
        self.assertEqual(t['scene']['mode'],'custom_scene_planned')

    def test_recovery_retains_first_delivery_and_requires_new_grasp(self):
        key='recover_second_item';t=self.suite.task(key);trace=good_trace(t)
        trace[7]['first_delivery_preserved']=False
        self.assertFalse(self.suite.check_trace(key,trace)['accepted'])
        trace=good_trace(t)
        for row in trace:row['picked_b_retry']=False
        self.assertFalse(self.suite.check_trace(key,trace)['accepted'])

    def test_malformed_bindings_cycles_dwell_and_false_native_claim(self):
        for mutate in (
            lambda t:t['milestones'][0]['bindings'].update(object='unknown'),
            lambda t:t['precedence'].append(['placed_a','picked_a']),
            lambda t:t['scene'].update(planned_entities={'fake_barrier':'unbuilt'}),
            lambda t:t.update(expected_grasp_occurrences=99),
            lambda t:t['scene'].update(asset='../escape.bddl')):
            d=document();mutate(d['tasks'][0])
            with self.assertRaises(ValueError):RobotTaskSuite(d)
        d=document();t=next(t for t in d['tasks'] if t['id']=='hold_then_place')
        t['precedence']=[['placed_a','picked_a']]
        with self.assertRaises(ValueError):RobotTaskSuite(d)

    def test_bddl_inspector_grouped_objects_regions_and_native_goal(self):
        text='''(define (problem test) (:fixtures table - table)
          (:objects a b - box) (:regions (slot (:target table)))
          (:goal (And (On a table_slot) (On b table_slot))))'''
        info=read_bddl(text)
        self.assertEqual(set(info['entities']),{'a','b','table','table_slot'})
        self.assertEqual(info['native_goal'][0],'And')
        with self.assertRaises(ValueError):read_bddl(text+')')
        with self.assertRaises(ValueError):read_bddl(text.replace('a b - box','a b'))

    def test_asset_audit_rejects_unbound_native_entity(self):
        d=document();d['tasks']=d['tasks'][:1];t=d['tasks'][0]
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/t['scene']['asset'];p.parent.mkdir(parents=True)
            p.write_text('(define (problem test) (:objects cream_cheese_1 akita_black_bowl_1 - box) (:goal (On cream_cheese_1 akita_black_bowl_1)))')
            self.assertEqual(len(audit_assets(RobotTaskSuite(d),directory)['tasks']),1)
            t['bindings']['a']='missing'
            with self.assertRaises(ValueError):audit_assets(RobotTaskSuite(d),directory)


if __name__=='__main__':unittest.main()
