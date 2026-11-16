"""Deterministic engineering fixtures, not a held-out robot benchmark."""
import unittest
from skillforge.control_dsl import Executor, Workcell, compile_program
from skillforge.controller_author import KINDS, make_spec, construct, teacher
from skillforge.ltlf import Expr, evaluate_closed_trace


class ControlDSLTests(unittest.TestCase):
    def setup_case(self, source='transfer a to tray\ntransfer b to buffer'):
        world = Workcell((-.7, -.6), {'a': (-.25, .1), 'b': (.3, -.3)}, radius=.04)
        points = {'tray': (.6, .55), 'buffer': (-.5, .55), 'home': (-.7, -.6)}
        specs = {k: make_spec(k, 20) for k in KINDS}
        blocks = {k: construct(teacher(s), s) for k, s in specs.items()}
        executor = Executor(compile_program(source, world.objects, points), points, blocks)
        return world, executor

    def test_two_objects_reuse_library_and_satisfy_temporal_order(self):
        world, executor = self.setup_case()
        identities = {k: b.sha256 for k, b in executor.blocks.items()}
        result = executor.run(world)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['completed_instructions'], 8)
        self.assertIsNone(world.held)
        for name, target in [('a', (.6, .55)), ('b', (-.5, .55))]:
            self.assertLessEqual(max(abs(a-b) for a,b in zip(world.objects[name], target)), .04000001)
        trace = [dict(a=t['after']['held']=='a', b=t['after']['held']=='b', safe=t['after']['safe']) for t in executor.trace]
        atom = lambda n: Expr('ap', name=n)
        formula = Expr('and', (Expr('G', (atom('safe'),)),
            Expr('F', (Expr('and', (atom('a'), Expr('X', (Expr('F', (atom('b'),)),)))),))))
        self.assertTrue(evaluate_closed_trace(formula, trace))
        self.assertEqual(identities, {k: b.sha256 for k,b in executor.blocks.items()})

    def test_macro_and_explicit_sequence_identical(self):
        w1, e1 = self.setup_case('transfer a to tray')
        w2, e2 = self.setup_case('move_to a\ngrasp a\nmove_to tray\nrelease a')
        self.assertEqual(e1.run(w1), e2.run(w2))

    def test_type_errors(self):
        for source in ['grasp tray', 'transfer tray to a', 'release home', 'move_to a to tray', 'python eval']:
            with self.assertRaises(ValueError):
                self.setup_case(source)

    def test_no_implicit_repair_of_invalid_order(self):
        w, e = self.setup_case('release a')
        self.assertEqual(e.run(w)['reason'], 'release_precondition')
        self.assertEqual(w.objects['a'], (-.25, .1))

    def test_jammed_grasp_times_out(self):
        w, e = self.setup_case('transfer a to tray')
        w.jammed = frozenset({'a'})
        e.skill_budget = 80
        self.assertEqual(e.run(w)['reason'], 'budget_exhausted')
        self.assertEqual(e.pc, 1)
        self.assertIsNone(w.held)

    def test_terminal_tick_is_noop(self):
        w, e = self.setup_case('move_to home')
        e.run(w)
        before = w.observe(), len(e.trace), e.ticks
        self.assertEqual(e.tick(w), 'success')
        self.assertEqual(before, (w.observe(), len(e.trace), e.ticks))

    def test_total_budget(self):
        w, e = self.setup_case()
        e.total_budget = 1
        self.assertEqual(e.run(w)['reason'], 'budget_exhausted')

    def test_false_arrival_does_not_advance(self):
        w, e = self.setup_case('move_to tray')
        spec = make_spec('near', 20)
        source = teacher(spec)
        start = source.index('  score =')
        e.blocks['near'] = construct(source[:start] + '  score = 1;\n}\n', spec)
        self.assertEqual(e.run(w)['reason'], 'false_completion_predicate')
        self.assertEqual(e.pc, 0)

    def test_collision_of_names_rejected(self):
        with self.assertRaises(ValueError):
            compile_program('move_to a', {'a'}, {'a'})

    def test_motion_limits_and_nonfinite_commands(self):
        for command in [dict(vx=.3,vy=0,grip=0),dict(vx=float('nan'),vy=0,grip=0)]:
            w, _ = self.setup_case()
            with self.assertRaises(ValueError):
                w.apply(command)

    def test_expansion_budget(self):
        with self.assertRaises(ValueError):
            compile_program('transfer a to tray', {'a'}, {'tray'}, limit=3)


if __name__ == '__main__':
    unittest.main()
