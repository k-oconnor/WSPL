import json
from pathlib import Path
import tempfile
import unittest
from skillforge.supervisor_workspace import Budget,freeze,verify_bundle,inventory


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'candidate';self.source.mkdir()
        (self.source/'submission.json').write_text('{"entrypoint":"policy.py"}')
        (self.source/'policy.py').write_text('value=1\n')
        self.t=100.
    def tearDown(self):self.temp.cleanup()
    def budget(self):return Budget(self.root/'budget.json',60,20,2,clock=lambda:self.t)
    def test_snapshot_roundtrip_and_independence(self):
        frozen=freeze(self.source,self.root/'store',{'kind':'synthetic'})
        (self.source/'policy.py').write_text('value=2\n')
        self.assertEqual((frozen/'policy.py').read_text(),'value=1\n')
        verify_bundle(frozen)
    def test_tampering_rejected(self):
        frozen=freeze(self.source,self.root/'store',{})
        (frozen/'policy.py').write_text('value=2\n')
        with self.assertRaises(ValueError):verify_bundle(frozen)
    def test_duplicate_snapshot(self):
        a=freeze(self.source,self.root/'store',{})
        self.assertEqual(a,freeze(self.source,self.root/'store',{}))
    def test_reserved_name(self):
        (self.source/'bundle_manifest.json').write_text('{}')
        with self.assertRaises(ValueError):freeze(self.source,self.root/'store',{})
    def test_nested_store_rejected(self):
        with self.assertRaises(ValueError):freeze(self.source,self.source/'store',{})
    def test_missing_declaration(self):
        (self.source/'submission.json').unlink()
        with self.assertRaises(ValueError):freeze(self.source,self.root/'store',{})
    def test_hardlink_rejected(self):
        import os
        os.link(self.source/'policy.py',self.source/'alias.py')
        with self.assertRaises(ValueError):inventory(self.source)
    def test_call_budget(self):
        b=self.budget();b.reserve_call();b.reserve_call()
        with self.assertRaises(RuntimeError):b.reserve_call()
    def test_gpu_budget_and_resume(self):
        b=self.budget();self.assertEqual(b.begin('gpu',True,100),20)
        self.t+=12;b.finish('complete');self.assertEqual(self.budget().remaining(True),8)
    def test_unresolved_job_blocks_resume(self):
        b=self.budget();b.begin('gpu',True,10)
        with self.assertRaises(RuntimeError):self.budget().begin('next',False,1)
    def test_limits_cannot_change(self):
        self.budget()
        with self.assertRaises(ValueError):Budget(self.root/'budget.json',61,20,2)
    def test_wall_budget_includes_downtime(self):
        self.budget();self.t+=61
        with self.assertRaises(RuntimeError):self.budget().begin('late',False,1)


if __name__=='__main__':unittest.main()
