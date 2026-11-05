import unittest
import numpy as np
from skillforge.wsp_editor import selected_flat_indices,plan,apply
from skillforge.wsp_artifact import TensorArtifact
from skillforge.wsp_values import Limits


class SliceScalingTests(unittest.TestCase):
    def test_indices_match_numpy_random_basic_slices(self):
        rng=np.random.default_rng(17)
        for shape in ((7,11),(3,4,5),(2,),()):
            for _ in range(150):
                sel=[]
                for width in shape:
                    if rng.random()<.35:sel.append(int(rng.integers(-width,width)))
                    else:
                        lo=None if rng.random()<.3 else int(rng.integers(-2*width,2*width+1))
                        hi=None if rng.random()<.3 else int(rng.integers(-2*width,2*width+1))
                        step=int(rng.choice([-3,-1,1,2]));sel.append(slice(lo,hi,step))
                actual=selected_flat_indices(shape,tuple(sel))
                expected=np.arange(int(np.prod(shape))).reshape(shape)[tuple(sel)]
                np.testing.assert_array_equal(actual,expected)

    def test_overlap_parent_signed_zero_and_augmented_snapshot(self):
        a=np.arange(36,dtype=np.float32).reshape(6,6);a[0,0]=-0.
        parent=TensorArtifact.from_arrays({'A':a})
        body='A[1:5,:]=A[0:4,:]; A[::-2,1:6:2]=parent.A[::-2,1:6:2]; A[2:4,0:2]+=A[3:5,1:3]; A[0,0]=parent.A[0,0];'
        source='wspl 0.2;target b=artifact("p",sha256="'+parent.sha256+'");edit b as c mode=fixed numeric=exact_to_storage {'+body+'}'
        staged=plan(source,parent=parent);result=apply(staged,parent=parent,expected_parent_id=parent.sha256).arrays()['A']
        expected=a.copy();expected[1:5]=expected[0:4].copy();expected[::-2,1:6:2]=a[::-2,1:6:2];expected[2:4,:2]+=expected[3:5,1:3].copy();expected[0,0]=a[0,0]
        np.testing.assert_array_equal(result.view(np.uint32),expected.view(np.uint32))

    def test_scalar_cost_independent_of_tensor_size_and_budget_enforced(self):
        parent=TensorArtifact.from_arrays({'A':np.zeros((512,512),np.float32)})
        source='wspl 0.2;target b=artifact("p",sha256="'+parent.sha256+'");edit b as c mode=fixed numeric=exact_to_storage {A[1,2]=parent.A[3,4];}'
        staged=plan(source,parent=parent,limits=Limits(max_operations=100))
        self.assertLess(staged.to_dict()['resources']['operations'],100)
        with self.assertRaises(ValueError):plan(source,parent=parent,limits=Limits(max_operations=1))
