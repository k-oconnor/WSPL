import json,tempfile,time,unittest
from pathlib import Path
from skillforge.supervisor_recovery import restore
from skillforge.supervisor_session import TOOL_INSTRUCTIONS,SupervisorSession
from skillforge.supervisor_workspace import save,Budget


class RecoveryTests(unittest.TestCase):
    def fixture(self,root):
        p=root/'parent';out=root/'out';out.mkdir()
        for d in ('supervisor/turn_002/collection','persistence_check','development/candidate'):
            (p/d).mkdir(parents=True,exist_ok=True)
        limits=dict(wall_seconds=72000,gpu_seconds=28800,calls=64)
        history=[dict(role='assistant',content='saved'),dict(role='tool',content={'outcome':'complete'})]
        save(p/'supervisor/session.json',dict(phase='interrupted',turn=2,submission=None,brief='brief',transcript=history))
        save(p/'budget.json',dict(limits=limits,active=None,calls_used=5,gpu_used=0,jobs=[],started=0))
        save(p/'status.json',dict(status='failed',elapsed_seconds=579.1))
        save(p/'persistence_check/receipt.json',dict(status='passed'))
        save(p/'supervisor/turn_002/collection/events.jsonl',dict(type='error',message='Selected model is at capacity'))
        # Event stream is one JSON object per line, unlike pretty-printed receipt.
        (p/'supervisor/turn_002/collection/events.jsonl').write_text(json.dumps(dict(type='error',message='Selected model is at capacity'))+'\n')
        (p/'supervisor/turn_002/collection/prompt.txt').write_text('brief\n'+TOOL_INSTRUCTIONS+'\nComplete visible history:\n'+json.dumps(history))
        (p/'development/candidate/policy.py').write_text('saved work')
        save(p/'development/candidate/submission.json',dict(entrypoint='policy.py'))
        return p,out,limits
    def test_restore_preserves_history_work_and_charged_budget(self):
        with tempfile.TemporaryDirectory() as t:
            p,out,l=self.fixture(Path(t));original=(p/'supervisor/session.json').read_bytes()
            restore(p,out,'brief',l)
            self.assertEqual(original,(p/'supervisor/session.json').read_bytes())
            b=Budget(out/'budget.json',**l);self.assertEqual(b.state['calls_used'],5)
            self.assertLess(b.remaining(),72000-579)
            self.assertEqual((out/'development/candidate/policy.py').read_text(),'saved work')
            seen=[]
            def collect(brief,history,folder):
                seen.append(history);return json.dumps(dict(tool='submit',directory='candidate',description='fixture'))
            s=SupervisorSession(out/'supervisor',out/'development',b,'brief',lambda *a:self.fail('repeated job'),collect)
            s.step();self.assertEqual(s.state['phase'],'submitted');self.assertEqual(b.state['calls_used'],6)
            self.assertEqual(len(seen[0]),2)
    def test_changed_prompt_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p,out,l=self.fixture(Path(t));(p/'supervisor/turn_002/collection/prompt.txt').write_text('wrong')
            with self.assertRaisesRegex(ValueError,'prompt differs'):restore(p,out,'brief',l)
    def test_completed_response_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p,out,l=self.fixture(Path(t));(p/'supervisor/turn_002/response.txt').write_text('response')
            with self.assertRaisesRegex(ValueError,'already produced'):restore(p,out,'brief',l)
    def test_active_job_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p,out,l=self.fixture(Path(t));b=json.loads((p/'budget.json').read_text());b['active']={'job':1};save(p/'budget.json',b)
            with self.assertRaisesRegex(ValueError,'active job'):restore(p,out,'brief',l)
    def test_different_error_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p,out,l=self.fixture(Path(t));(p/'supervisor/turn_002/collection/events.jsonl').write_text(json.dumps(dict(type='error',message='other')))
            with self.assertRaisesRegex(ValueError,'capacity failure'):restore(p,out,'brief',l)


if __name__=='__main__':unittest.main()
