import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from skillforge.reliable_json import save,status


class LoggingTests(unittest.TestCase):
    def test_transient_lock_retries_and_commits(self):
        import os
        real=os.replace;calls=[]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'status.json';path.write_text('{"old":true}')
            def replace(a,b):
                calls.append(1)
                if len(calls)<3:raise PermissionError('simulated WinError5')
                real(a,b)
            with patch('skillforge.reliable_json.os.replace',side_effect=replace),patch('skillforge.reliable_json.time.sleep'):
                save(path,dict(new=True))
            self.assertEqual(json.loads(path.read_text()),dict(new=True));self.assertEqual(len(calls),3)
            self.assertEqual(len(list(Path(d).iterdir())),1)

    def test_persistent_status_lock_nonfatal_but_artifacts_fail(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'status.json';value={}
            with patch('skillforge.reliable_json.os.replace',side_effect=PermissionError('locked')),patch('skillforge.reliable_json.time.sleep'):
                self.assertFalse(status(path,value))
                with self.assertRaises(PermissionError):save(path,value)
            self.assertEqual(value['status_write_failures'],1)
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_other_io_errors_not_retried(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('skillforge.reliable_json.os.replace',side_effect=OSError('disk failure')) as replace:
                with self.assertRaises(OSError):save(Path(d)/'result.json',{})
                self.assertEqual(replace.call_count,1)


class RecoveryTests(unittest.TestCase):
    def test_preserves_two_turns_and_remaining_budget(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
        from run_attention_probe import session
        from test_attention_probe import bank
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=root/'parent';parent.mkdir();r=bank();b=bank();b['alpha'][0,0]=1;b['V'][0,0,0]=1
            np.savez_compressed(parent/'reference.npz',**r);turns=[]
            for i in range(2):
                turn=parent/'forced'/f'turn_{i:02d}';(turn/'constructed').mkdir(parents=True)
                np.savez_compressed(turn/'constructed/candidate.npz',**b)
                (turn/'answer.md').write_text('program'+str(i));save(turn/'feedback.json',dict(accepted=True))
                turns.append(dict(step=i,status='accepted'))
            np.savez_compressed(parent/'forced/current.npz',**b)
            save(parent/'status.json',dict(status='incomplete',step=1,turns=turns))
            prompts=[]
            def collector(folder,text):folder.mkdir();prompts.append(text);return 'FINAL',{}
            def forbidden(*args):raise AssertionError('must not repeat construction or evaluation')
            result=session('forced',r,r,{},root/'resumed','guide',{},collector,forbidden,forbidden,lambda *a:None,resume=parent)
            self.assertEqual(result['V'].tobytes(),b['V'].tobytes());self.assertEqual(len(prompts),1)
            self.assertIn('Response 3/6',prompts[0]);self.assertIn('program0',prompts[0]);self.assertIn('program1',prompts[0])
            for i in range(2):
                rel=Path(f'turn_{i:02d}')/'constructed/candidate.npz'
                self.assertEqual((root/'resumed'/rel).read_bytes(),(parent/'forced'/rel).read_bytes())
