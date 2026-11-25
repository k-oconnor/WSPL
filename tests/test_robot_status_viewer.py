import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import robot_status_viewer as viewer


class ViewerTests(unittest.TestCase):
    def save(self,path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
    def test_saved_completion_requires_receipt_and_clean_exit(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(viewer,'process_alive',return_value=False):
            r=Path(folder)/'run';r.mkdir()
            self.save(r/'results.json',dict(status='complete'))
            self.assertEqual(viewer.snapshot(r)['state'],'Finishing')
            self.save(r/'source_receipt.json',dict(status='complete'))
            self.save(r.with_suffix('.exit.json'),dict(returncode=0))
            self.assertEqual(viewer.snapshot(r)['state'],'Complete')
    def test_counts_completed_not_started_and_keeps_inspection_separate(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(viewer,'process_alive',return_value=True):
            r=Path(folder)/'run';r.mkdir()
            self.save(r.with_suffix('.launch.json'),dict(pid=1,started_unix=1))
            self.save(r/'status.json',dict(status='author_call'))
            self.save(r/'native0/turn_00/usage.json',dict(usage={'input_tokens':9}))
            self.save(r/'native0/turn_00/inspection/diagnostics.json',dict(projections={'p0':'layer.q'},examples=2,action_correct=1,mean_target_nll=1,seconds=1,tokens='all',weights=True))
            (r/'native0/turn_01').mkdir()
            self.save(r/'native0/status.json',dict(inspection_attempts=1,proposal_attempts=0))
            s=viewer.snapshot(r)
            self.assertEqual(s['responses'],1);self.assertEqual(s['authors'][0]['started'],2)
            self.assertEqual(s['authors'][0]['evaluations'],0);self.assertEqual(s['inspections'],1)
            self.assertEqual(s['phase'],'Waiting for Astra response')
    def test_partial_write_retains_last_valid_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'status.json';self.save(p,{'status':'author_call'})
            reader=viewer.Reader();self.assertEqual(reader.read(p)['status'],'author_call')
            p.write_text('{broken')
            self.assertEqual(reader.read(p)['status'],'author_call');self.assertTrue(reader.warnings)
    def test_stopped_worker_and_failed_author_are_not_success(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(viewer,'process_alive',return_value=False):
            r=Path(folder)/'run';r.mkdir();self.save(r.with_suffix('.launch.json'),dict(pid=1))
            self.save(r/'native0/frozen.json',dict(status='authoring_failed'))
            self.save(r/'all_frozen.json',dict(native0={'status':'authoring_failed'}))
            s=viewer.snapshot(r);self.assertEqual(s['state'],'Stopped');self.assertEqual(s['finished_authors'],1)
            self.assertEqual(s['final_expected'],448)
    def test_final_current_arm_ignores_stale_author_fields(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(viewer,'process_alive',return_value=True):
            r=Path(folder)/'run';r.mkdir();self.save(r.with_suffix('.launch.json'),dict(pid=1))
            self.save(r/'status.json',dict(status='final_evaluation',final_arm='gradient',author_arm='wspl'))
            self.save(r/'final_gradient/episodes.json',[dict(success=True,condition='ordinary')])
            s=viewer.snapshot(r);self.assertEqual(s['current'],'gradient');self.assertEqual(s['final_done'],1)
    def test_recovery_combines_history_without_counting_failed_reply(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(viewer,'process_alive',return_value=True):
            base=Path(folder);old=base/'old';r=base/'new';old.mkdir();r.mkdir()
            self.save(old/'native0/frozen.json',dict(status='frozen'))
            self.save(old/'native0/turn_00/usage.json',dict(usage={'input_tokens':4}))
            self.save(old/'native1/turn_00/usage.json',dict(usage={'input_tokens':5}))
            (old/'native1/turn_01').mkdir()
            self.save(r/'native1/turn_02/usage.json',dict(usage={'input_tokens':6}))
            self.save(r/'recovery_context.json',dict(prior_output=str(old),completed_authors=['native0'],resume_author='native1',prior_elapsed_seconds=10,failed_invocations=1))
            s=viewer.snapshot(r)
            self.assertEqual(s['finished_authors'],1);self.assertEqual(s['responses'],3)
            self.assertEqual(s['tokens']['input_tokens'],15);self.assertEqual(s['failed_author_requests'],1)


if __name__=='__main__':unittest.main()
