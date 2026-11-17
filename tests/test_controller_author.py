"""Regression checks for compiled blocks, physical scoring and temporal composition."""
import unittest
from skillforge.controller_author import *


class ControllerAuthorTests(unittest.TestCase):
    def library(self, index=20):
        specs = {kind: make_spec(kind, index) for kind in KINDS}
        return specs, {kind: construct(teacher(spec), spec) for kind, spec in specs.items()}

    def test_teacher_blocks_match_independent_functions(self):
        for index in [0, 20, 160, 161, 319]:
            specs, blocks = self.library(index)
            for kind in KINDS:
                self.assertTrue(check_block(blocks[kind], specs[kind])['passes'])

    def test_single_markdown_fence_is_accepted(self):
        spec = make_spec('move', 20)
        source = teacher(spec)
        self.assertEqual(construct(source, spec).sha256,
                         construct('```wspl\n' + source + '```', spec).sha256)

    def test_split_specifications_are_disjoint(self):
        data = dataset()
        self.assertEqual([len(v) for v in data.values()], [1536, 96, 96])
        ids = [r['id'] for rows in data.values() for r in rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_formula_changes_selected_skills(self):
        specs, blocks = self.library()
        direct = rollout(blocks, specs)
        via = rollout(blocks, specs, True)
        self.assertTrue(direct['success'])
        self.assertTrue(via['success'])
        self.assertNotIn('waypoint', direct['actions'])
        self.assertIn('waypoint', via['actions'])

    def test_false_positive_arrival_does_not_fake_success(self):
        specs, blocks = self.library()
        source = teacher(specs['near'])
        source = source[:source.index('  score =')] + '  score = 1;\n}\n'
        blocks['near'] = construct(source, specs['near'])
        self.assertFalse(rollout(blocks, specs)['success'])

    def test_wrong_gripper_does_not_fake_success(self):
        specs, blocks = self.library()
        blocks['close'] = blocks['open']
        self.assertFalse(rollout(blocks, specs)['success'])

    def test_unbounded_command_rejected(self):
        specs, blocks = self.library()
        source = teacher(specs['move'])
        source = source[:source.index('  vx =')] + '  vx = 1;\n  vy = 1;\n  grip = 0;\n}\n'
        blocks['move'] = construct(source, specs['move'])
        result = rollout(blocks, specs)
        self.assertFalse(result['success'])
        self.assertEqual(result['status'], 'actuator_violation')

    def test_swapped_ports_rejected(self):
        spec = make_spec('move', 20)
        source = teacher(spec).replace('output vx:', 'output temporary:').replace('output vy:', 'output vx:').replace('output temporary:', 'output vy:')
        with self.assertRaises(ValueError):
            construct(source, spec)

    def test_task_order_is_checked_on_physical_trace(self):
        trace = [dict(safe=True, holding=False, visited=False, delivered=True)]
        self.assertFalse(evaluate_closed_trace(mission(False), trace))
        self.assertFalse(evaluate_closed_trace(mission(True), trace))

    def test_negative_coordinates_and_l1_predicates(self):
        specs, blocks = self.library(319)
        result = rollout(blocks, specs, True, initial=(-.7, .6), target=(.6, -.55))
        self.assertTrue(result['success'])


if __name__ == '__main__':
    unittest.main()
