import unittest
import numpy as np
from skillforge.wsp_artifact import TensorArtifact
from skillforge.wsp_editor import plan,apply


class WholeTensorTests(unittest.TestCase):
    def execute(self,parent,body):
        source='wspl 0.2;target p=artifact("parent",sha256="'+parent.sha256+'");edit p as c mode=fixed numeric=exact_to_storage {'+body+'}'
        staged=plan(source,parent=parent)
        return apply(staged,parent=parent,expected_parent_id=parent.sha256),staged.to_dict()

    def test_whole_matches_full_slice_bits_and_write_accounting(self):
        for shape in ((4,),(2,3),(2,2,3)):
            a=np.arange(np.prod(shape),dtype=np.float32).reshape(shape);a.flat[0]=-0.
            parent=TensorArtifact.from_arrays({'A':a});sel='['+','.join(':' for _ in shape)+']'
            for expression in ('parent.A','parent.A * (17/16)','0'):
                x,xp=self.execute(parent,'A = '+expression+';')
                y,yp=self.execute(parent,'A'+sel+' = '+expression+';')
                self.assertEqual(x.sha256,y.sha256)
                self.assertEqual(xp['touched'],yp['touched']);self.assertEqual(xp['resources'],yp['resources'])
                self.assertEqual(xp['writes'][0]['flat_indices'],yp['writes'][0]['flat_indices'])

    def test_augmented_snapshot_and_parent_unchanged(self):
        parent=TensorArtifact.from_arrays({'A':np.arange(6,dtype=np.float32).reshape(2,3)})
        before=parent.sha256
        whole,_=self.execute(parent,'A += A; A *= 2; A -= parent.A;')
        sliced,_=self.execute(parent,'A[:,:] += A; A[:,:] *= 2; A[:,:] -= parent.A;')
        self.assertEqual(whole.sha256,sliced.sha256);self.assertEqual(parent.sha256,before)

    def test_unknown_names_parent_writes_and_shape_change_rejected(self):
        parent=TensorArtifact.from_arrays({'A':np.zeros((2,3),np.float32)})
        for body in ('B=0;','parent.A=0;','A=zeros([4,4]);'):
            with self.assertRaises(ValueError):self.execute(parent,body)
