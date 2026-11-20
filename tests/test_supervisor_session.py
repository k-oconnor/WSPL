import json
from pathlib import Path
import tempfile
import unittest
from skillforge.supervisor_session import SupervisorSession,parse_request
from skillforge.supervisor_workspace import Budget,verify_bundle


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.work=self.root/'work';self.work.mkdir();self.budget=Budget(self.root/'budget.json',60,30,4)
    def tearDown(self):self.temp.cleanup()
    def test_scripted_supervisor_edits_then_submits(self):
        def collect(brief,transcript,folder):
            if not transcript:return json.dumps(dict(tool='python',code='synthetic update',gpu=False,seconds=1))
            self.assertEqual(transcript[-1]['content'],{'metric':1})
            return json.dumps(dict(tool='submit',directory='candidate',description='synthetic policy'))
        def execute(request,folder):
            target=self.work/'candidate';target.mkdir()
            (target/'submission.json').write_text('{"entrypoint":"policy.py"}')
            (target/'policy.py').write_text('value=2')
            return dict(metric=1)
        session=SupervisorSession(self.root/'session',self.work,self.budget,'brief',execute,collect)
        session.step()
        resumed=SupervisorSession(self.root/'session',self.work,self.budget,'brief',execute,collect)
        resumed.step();self.assertEqual(resumed.state['phase'],'submitted')
        verify_bundle(resumed.state['submission'])
        with self.assertRaises(RuntimeError):resumed.step()
    def test_interrupt_does_not_retry_or_refund(self):
        def fail(*args):raise RuntimeError('capacity')
        session=SupervisorSession(self.root/'session',self.work,self.budget,'brief',fail,fail)
        with self.assertRaises(RuntimeError):session.step()
        self.assertEqual(self.budget.state['calls_used'],1)
        with self.assertRaises(RuntimeError):session.step()
    def test_bad_schema_returns_feedback(self):
        session=SupervisorSession(self.root/'session',self.work,self.budget,'brief',None,lambda *a:'{}')
        self.assertIn('error',session.step());self.assertEqual(session.state['phase'],'ready')
    def test_submission_cannot_escape(self):
        session=SupervisorSession(self.root/'session',self.work,self.budget,'brief',None,
            lambda *a:json.dumps(dict(tool='submit',directory='..',description='escape')))
        with self.assertRaises(ValueError):session.step()
    def test_nonboolean_gpu_rejected(self):
        with self.assertRaises(ValueError):parse_request('{"tool":"python","code":"pass","gpu":1,"seconds":1}')


if __name__=='__main__':unittest.main()
