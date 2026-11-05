import json
from pathlib import Path
import tempfile
import unittest
from run_go_authoring import author_sessions,ORDER
from skillforge.go_authoring import GoAuthoringStudy
from skillforge.go_replay import sha


class ResumeTests(unittest.TestCase):
    def fixture(self,parent):
        rows=[]
        for i,(draw,arm) in enumerate(ORDER[:4]):
            name=f'{draw}_{arm}';folder=parent/name;folder.mkdir()
            model=GoAuthoringStudy();model.arrays['b7'][0]=i+1;model.save(folder/'final.npz')
            rows.append(dict(name=name,draw=draw,arm=arm,status='frozen',final_sha256=sha(folder/'final.npz'),turns=[dict(collection=dict(thread_id=str(i)))]))
        rows.append(dict(name='2_python',status='authoring',turns=[]))
        (parent/'status.json').write_text(json.dumps(dict(status='incomplete',error='RuntimeError: call_wall_guard',active_seconds=123,trials=rows)))
        return dict(resume=dict(parent=str(parent),files={p.relative_to(parent).as_posix():sha(p) for p in parent.rglob('*') if p.is_file()}))

    def test_only_remaining_calls_and_exact_frozen_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=root/'parent';parent.mkdir();out=root/'out';out.mkdir();config=self.fixture(parent);calls=[]
            def collector(folder,*args):
                folder.mkdir();calls.append(folder.parent.name)
                return 'FINAL',dict(thread_id=str(folder))
            def compiler(*args):raise AssertionError('no construction expected')
            rows,frozen=author_sessions(GoAuthoringStudy,out,'guide',config,collector,compiler,lambda:False,lambda rows:None)
            self.assertEqual(calls,['2_python','2_wspl']);self.assertEqual(len(rows),6);self.assertEqual(len(frozen),7)
            for name in ('0_python','0_wspl','1_wspl','1_python'):
                self.assertEqual(sha(parent/name/'final.npz'),sha(out/name/'final.npz'))

    def test_changed_archive_refused_before_call(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=root/'parent';parent.mkdir();out=root/'out';out.mkdir();config=self.fixture(parent)
            (parent/'0_python/final.npz').write_bytes(b'changed')
            def forbidden(*args):raise AssertionError('must not call')
            with self.assertRaisesRegex(ValueError,'archive changed'):
                author_sessions(GoAuthoringStudy,out,'guide',config,forbidden,forbidden,lambda:False,lambda rows:None)
