"""plan_and_apply must return exactly what plan() followed by apply() returns."""
import unittest

import numpy as np

from skillforge.wsp_artifact import TensorArtifact
from skillforge.wsp_editor import plan, apply, plan_and_apply, EditPlan, EditError
from skillforge.wsp_values import Limits


class PlanAndApplyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        self.parent = TensorArtifact.from_arrays({'A': rng.standard_normal((6, 6)).astype(np.float32), 'c': np.zeros(6, np.float32)})

    def source(self, body):
        return 'wspl 0.2;target b=artifact("p",sha256="'+self.parent.sha256+'");edit b as c mode=fixed numeric=exact_to_storage {'+body+'}'

    def test_matches_plan_then_apply(self):
        for body in ('A[:,:]=0;', 'for i in range(6) { A[i,:]+=parent.A[5-i,:]*(1/3); c[i]=i; }', 'A[0,0]=parent.A[0,0];'):
            with self.subTest(body=body):
                staged, candidate = plan_and_apply(self.source(body), parent=self.parent)
                self.assertEqual(staged.sha256, plan(self.source(body), parent=self.parent).sha256)
                replayed = apply(staged, parent=self.parent, expected_parent_id=self.parent.sha256)
                self.assertEqual(candidate._text, replayed._text)
                self.assertEqual(candidate.sha256, staged.to_dict()['candidate_id'])

    def test_declared_dense_target_and_errors_unchanged(self):
        source = 'wspl 0.2;target base=dense(widths=[3,2],activations=[relu],layout=in_out,storage=f32,init=zeros);edit base as c mode=fixed numeric=exact_to_storage { W1[0,0]=1; }'
        staged, candidate = plan_and_apply(source)
        self.assertEqual(candidate._text, apply(staged, parent=TensorArtifact(staged.to_dict()['parent']), expected_parent_id=staged.to_dict()['parent_id'])._text)
        for body in ('A[9,0]=1;', 'A[0,0]=1e39;'):
            with self.assertRaises(EditError) as direct:
                plan(self.source(body), parent=self.parent)
            with self.assertRaises(EditError) as combined:
                plan_and_apply(self.source(body), parent=self.parent)
            self.assertEqual(str(direct.exception), str(combined.exception))
        with self.assertRaises(ValueError):
            plan_and_apply(self.source('A[:,:]=0;'), parent=self.parent, limits=Limits(max_operations=10))

    def test_saved_plans_still_replay(self):
        staged, _ = plan_and_apply(self.source('A[1,:]=2;'), parent=self.parent)
        reloaded = EditPlan(staged._text)  # As EditPlan.load returns: no cached candidate.
        self.assertFalse(hasattr(reloaded, '_candidate'))
        self.assertEqual(apply(reloaded, parent=self.parent, expected_parent_id=self.parent.sha256).sha256, staged.to_dict()['candidate_id'])


if __name__ == '__main__':
    unittest.main()
