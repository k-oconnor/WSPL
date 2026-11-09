import json
from pathlib import Path
import tempfile
import unittest
from run_capability_arc import paired,load_rows


class CapabilityArcTests(unittest.TestCase):
    def test_paired_corrections_regressions_and_raw(self):
        a=[dict(id=str(i),gold=0,correct=v,raw_correct=not v) for i,v in enumerate([True,False,False,True])]
        b=[dict(id=str(i),gold=0,correct=v,raw_correct=v) for i,v in enumerate([False,True,True,True])]
        p=paired(a,b)
        self.assertEqual((p['corrections'],p['regressions'],p['net_correct'],p['accuracy_delta']),(2,1,1,.25))
        self.assertEqual(paired(a,b,'raw_correct')['net_correct'],1)
        self.assertEqual(paired(a,a)['net_correct'],0)

    def test_misalignment_rejected(self):
        row=dict(id='x',gold=0,correct=True)
        for a,b in [([],[]),([row],[]),([row],[dict(row,id='y')]),([row],[dict(row,gold=1)])]:
            with self.assertRaises(ValueError):paired(a,b)

    def test_rows_count_unique_and_labels(self):
        row=dict(id='x',question='Synthetic?',choices=dict(label=['A','B'],text=['one','two']),answerKey='B')
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'rows.jsonl';p.write_text(json.dumps(row)+'\n')
            self.assertEqual(load_rows(p,1),[row])
            with self.assertRaises(ValueError):load_rows(p,2)
            p.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaises(ValueError):load_rows(p,2)
