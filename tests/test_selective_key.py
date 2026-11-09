import unittest
import numpy as np
from skillforge.selective_key import validate_key,metrics,descriptive_gate
from test_attention_probe import bank


class SelectivityTests(unittest.TestCase):
    def test_only_active_key_writable(self):
        ref=bank();ref['alpha'][2,0]=.75;ref['V'][2,0,0]=1;ref['train_Q']=np.zeros((13,3,64),np.float32)
        candidate={k:v.copy() for k,v in ref.items()};candidate['K'][2,0,0]=2;validate_key(candidate,ref)
        candidate['K'][1,0,0]=1
        with self.assertRaises(ValueError):validate_key(candidate,ref)
        candidate['K'][1,0,0]=0;candidate['train_Q'][0,0,0]=1
        with self.assertRaises(ValueError):validate_key(candidate,ref)

    def test_selectivity_and_retention_are_separate(self):
        choices=dict(text=['convection','radiation','conduction','evaporation'],label=['A','B','C','D'])
        rows=[dict(id=str(i),choices=choices,answerKey='C' if i<2 else 'D',category='conduction' if i<2 else 'evaporation',setting=str(i//2)) for i in range(6)]
        base=[dict(id=str(i),correct=i>=2,virtual_mass=[]) for i in range(6)]
        records=[dict(id=str(i),correct=True,virtual_mass=[dict(mass=[[.8]*3 if i<2 else [.05]*3])]) for i in range(6)]
        m=metrics(rows,records,base);b=metrics(rows,base,base)
        self.assertEqual(m['routing_auc'],1);self.assertEqual(m['initially_correct_negative_n'],4)
        self.assertTrue(all(v for k,v in descriptive_gate(dict(metrics=m),dict(metrics=b)).items() if k!='note'))
        for r in records:r['virtual_mass']=[dict(mass=[[.75]*3])]
        broad=metrics(rows,records,base)
        self.assertEqual(broad['routing_auc'],.5)
        self.assertFalse(descriptive_gate(dict(metrics=broad),dict(metrics=b))['selective'])
        records[-1]['correct']=False
        broken=metrics(rows,records,base);self.assertEqual(broken['broken'],['5'])
        with self.assertRaises(ValueError):metrics(rows,list(reversed(records)),base)

    def test_panel_is_balanced_and_disjoint_wording(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
        from prepare_selective_key import final_panel
        choices=dict(text=['convection','radiation','conduction','evaporation'],label=['A','B','C','D'])
        panel=final_panel(choices)
        self.assertEqual(len({r['question'] for r in panel}),32)
        for c in choices['text']:self.assertEqual(sum(r['category']==c for r in panel),8)
        for setting in {r['setting'] for r in panel}:self.assertEqual(sum(r['setting']==setting for r in panel),4)
