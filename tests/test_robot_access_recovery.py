import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.robot_access_recovery import restore_author,copy_regular_tree
from skillforge.robot_authoring_access_resume import run_session
from skillforge.robot_collector import check_event


def prior_fixture(root):
    p=root/'old';(p/'sandbox/inputs').mkdir(parents=True);(p/'sandbox/work').mkdir()
    (p/'sandbox/inputs/data.txt').write_text('prior diagnostic')
    (p/'sandbox/work/script.py').write_text('saved = 7\n')
    tr=[]
    for i in range(21):tr.extend([dict(role='assistant',content='{}'),dict(role='tool',content={'status':'saved'})])
    (p/'transcript.json').write_text(json.dumps(tr))
    (p/'status.json').write_text(json.dumps(dict(turns=21,proposal_attempts=2,inspection_attempts=5,candidate='saved-candidate')))
    return p


class Workspace:
    def __init__(self,root,runtime):
        self.root=root;self.inputs=root/'inputs';self.work=root/'work'
        self.inputs.mkdir(parents=True);self.work.mkdir();self.grants=[]
        self.profile=SimpleNamespace(grant=lambda path,**kw:self.grants.append((str(path),kw)))


class RecoveryTests(unittest.TestCase):
    def test_restores_exact_history_files_and_readonly_historical_paths(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);prior=prior_fixture(r);w=Workspace(r/'new','unused')
            result=restore_author(w,dict(author_folder=str(prior),consumed_responses=22))
            self.assertEqual(result['start_turn'],22);self.assertEqual(result['attempts'],2);self.assertEqual(result['queries'],5)
            self.assertEqual(len(result['transcript']),42);self.assertEqual(result['accepted'],'saved-candidate')
            self.assertEqual((w.work/'script.py').read_bytes(),(prior/'sandbox/work/script.py').read_bytes())
            self.assertTrue(all(not kwargs.get('writable',False) for _,kwargs in w.grants))
    def test_no_extra_response_budget_and_no_implicit_candidate_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);prior=prior_fixture(r);common=r/'common';common.mkdir();(common/'catalog.json').write_text('{}')
            seen=[]
            def collect(*args,**kwargs):
                seen.append(args[4]);return json.dumps(dict(tool='unknown')),{}
            cfg=dict(native_runtime='unused',author_runtime='unused',author_responses=24,inspection_attempts=8,proposal_attempts=4,
                cli='unused',catalog='unused',instructions='unused',author_call_seconds=1)
            value=run_session('native',common,{},cfg,r/'new','manifest','pin',lambda:None,lambda:False,lambda **kw:None,lambda *a:None,lambda s:None,
                collect_fn=collect,workspace_factory=Workspace,resume=dict(author_folder=str(prior),consumed_responses=22))
            self.assertEqual(len(seen),2);self.assertTrue(all('Only 2 responses remain' in p for p in seen))
            self.assertEqual(value,'saved-candidate')
            self.assertFalse((r/'new/turn_21').exists());self.assertTrue((r/'new/turn_22').exists());self.assertTrue((r/'new/turn_23').exists())
            frozen=json.loads((r/'new/frozen.json').read_text());self.assertEqual(frozen['turns'],24);self.assertEqual(frozen['prior_failed_invocations'],1)
            self.assertEqual(json.loads((prior/'status.json').read_text())['turns'],21)
    def test_rejects_wrong_accounting_before_granting_history(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);prior=prior_fixture(r);w=Workspace(r/'new','unused')
            with self.assertRaisesRegex(ValueError,'accounting'):restore_author(w,dict(author_folder=str(prior),consumed_responses=21))
            self.assertEqual(w.grants,[])
    def test_usage_error_is_preserved_without_retry(self):
        with self.assertRaisesRegex(ValueError,'usage limit'):
            check_event(dict(type='error',message='You have hit your usage limit'))


if __name__=='__main__':unittest.main()
