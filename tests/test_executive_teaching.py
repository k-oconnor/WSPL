import json
import unittest
from skillforge.executive_teaching import eligible_rows,parse_teacher,ordinary_action,teacher_prompt,CONDITIONS
from skillforge.teaching_training import draw_schedule
from run_executive_teaching import summarize


class TeachingTests(unittest.TestCase):
    def test_no_success_no_imitation(self):
        episodes=[dict(mission=c,condition=c,result=dict(success=True),records=[dict(accepted=True,student_prompt='observation',action='{"op":"stop"}')]) for c in CONDITIONS]
        rows,counts=eligible_rows(episodes);self.assertEqual(len(rows),4)
        episodes[0]['result']['success']=False
        with self.assertRaises(ValueError):eligible_rows(episodes)
    def test_invalid_teacher_rows_excluded(self):
        episodes=[dict(mission=c,condition=c,result=dict(success=True),records=[dict(accepted=False)]) for c in CONDITIONS]
        with self.assertRaises(ValueError):eligible_rows(episodes)
    def test_prediction_not_student_label(self):
        prediction,action=parse_teacher('{"prediction":"I expect movement","action":{"op":"stop"}}')
        self.assertEqual(prediction,'I expect movement');self.assertEqual(json.loads(action),dict(op='stop'))
        with self.assertRaises(ValueError):parse_teacher('{"action":{"op":"stop"}}')
    def test_matched_schedules(self):
        a=draw_schedule(512,40,'rehearsal');b=draw_schedule(512,40,'taught')
        self.assertEqual(len(a),128);self.assertEqual(sum(len(x) for x in b),1024)
        self.assertTrue(all(x[:4]==y[:4] for x,y in zip(a,b)))
        self.assertEqual(sum(k=='teacher' for batch in b for k,i in batch),512)
        self.assertEqual(sum(k=='teacher' for batch in a for k,i in batch),0)
    def test_ordinary_reference_uses_only_documented_role(self):
        p=dict(documentation={'opaque':'Original motor block'},observation=dict(selected_module=None))
        self.assertEqual(ordinary_action(p),dict(op='select',module='opaque'))
    def test_teacher_gets_student_fields_only(self):
        p=dict(observation={'last':None},documentation={},private_mapping='SECRET')
        self.assertNotIn('SECRET',teacher_prompt(p));self.assertIn('one JSON object',teacher_prompt(p))
    def test_learning_gate_requires_control_and_retention(self):
        rows=[]
        for arm in ('unchanged','rehearsal','taught'):
            for c in CONDITIONS:
                for i in range(8):
                    rows.append(dict(arm=arm,condition=c,mission=i,success=c=='ordinary' or arm=='taught',probes=1,physical_ticks=10))
        self.assertTrue(summarize(rows)['learning_signal_passed'])
        next(r for r in rows if r['arm']=='taught' and r['condition']=='ordinary')['success']=False
        self.assertFalse(summarize(rows)['learning_signal_passed'])


if __name__=='__main__':unittest.main()
