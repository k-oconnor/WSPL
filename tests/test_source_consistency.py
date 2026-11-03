"""Source identity checks use tiny subprocess fixtures, never training jobs."""
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'experiments'))
from source_consistency import digest, inventory, write_json, verify, ConsistencyError

class SourceConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'src').mkdir()
        self.model = self.root / 'src/pinned_model.py'
        self.model.write_text('class Model:\n    value = 7\n', encoding='utf-8')
        self.runner = self.root / 'src/pinned_runner.py'
        self.runner.write_text("def run(constructor, output_dir, config):\n    model = constructor()\n    (output_dir / 'value.txt').write_text(str(model.value))\n", encoding='utf-8')
        self.manifest = self.root / 'candidate.json'
        self.output = self.root / 'run'
        self.pin()

    def pin(self, **updates):
        self.spec = dict(schema=1, model_id='fixture-only', source_roots=['src'],
                         constructor=dict(module='pinned_model', symbol='Model', source='src/pinned_model.py'),
                         runner=dict(module='pinned_runner', symbol='run', source='src/pinned_runner.py'),
                         constructor_kwargs={}, runner_config={}, files=inventory(self.root, ['src']))
        self.spec.update(updates)
        write_json(self.manifest, self.spec)
        self.expected = digest(self.manifest)

    def run_gate(self):
        code = "import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from source_consistency import execute; execute(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4],Path(sys.argv[5]))"
        return subprocess.run([sys.executable, '-c', code, str(REPO/'experiments'), str(self.root), str(self.manifest), self.expected, str(self.output)], capture_output=True, text=True)

    def test_verified_constructor_and_receipt(self):
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.output/'value.txt').read_text(), '7')
        receipt = json.loads((self.output/'source_receipt.json').read_text())
        self.assertEqual(receipt['status'], 'complete')
        self.assertEqual(receipt['constructor_calls'], 1)
        self.assertEqual(receipt['manifest_sha256'], self.expected)

    def test_changed_source_fails_before_runner(self):
        self.model.write_text('raise RuntimeError("must never execute")\n')
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Source inventory mismatch', result.stderr)
        self.assertFalse(self.output.exists())

    def test_manifest_rehash_cannot_self_approve(self):
        self.model.write_text('class Model:\n    value = 9\n')
        self.spec['files'] = inventory(self.root, ['src'])
        write_json(self.manifest, self.spec)
        result = self.run_gate()
        self.assertIn('independently approved SHA-256 pin', result.stderr)
        self.assertFalse(self.output.exists())

    def test_added_or_missing_source_rejected(self):
        extra = self.root/'src/extra.py';extra.write_text('x=1\n')
        with self.assertRaises(ConsistencyError):verify(self.root,self.manifest,self.expected)
        extra.unlink();self.model.unlink()
        with self.assertRaises(ConsistencyError):verify(self.root,self.manifest,self.expected)

    def test_wrong_callable_source_rejected(self):
        self.pin(constructor=dict(module='pinned_model', symbol='Model', source='src/pinned_runner.py'))
        result = self.run_gate()
        self.assertIn('different source', result.stderr)
        self.assertFalse(self.output.exists())

    def test_runner_must_use_constructor(self):
        self.runner.write_text('def run(constructor, output_dir, config):\n    pass\n')
        self.pin();result=self.run_gate()
        self.assertIn('did not call the pinned constructor',result.stderr)
        self.assertEqual(json.loads((self.output/'source_receipt.json').read_text())['status'],'failed')

    def test_stale_bytecode_is_not_executed(self):
        original=self.model.read_bytes();stat=self.model.stat()
        self.model.write_bytes(original.replace(b'7',b'9'))
        os.utime(self.model,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        py_compile.compile(str(self.model),doraise=True)
        self.model.write_bytes(original);os.utime(self.model,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        result=self.run_gate();self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual((self.output/'value.txt').read_text(),'7')

    def test_postrun_mutation_fails(self):
        self.runner.write_text("def run(constructor, output_dir, config):\n    constructor()\n    from pathlib import Path\n    Path(__file__).write_text('# changed')\n")
        self.pin();result=self.run_gate()
        self.assertIn('Source inventory mismatch',result.stderr)
        self.assertEqual(json.loads((self.output/'source_receipt.json').read_text())['status'],'failed')

    def test_dependency_shadowing_is_rejected(self):
        (self.root/'src/pinned_helper.py').write_text('value=7\n')
        outside=self.root/'outside';outside.mkdir()
        (outside/'pinned_helper.py').write_text('value=9\n')
        self.runner.write_text("def run(constructor, output_dir, config):\n    import sys\n    from pathlib import Path\n    sys.path.insert(0,str(Path(__file__).parents[1]/'outside'))\n    import pinned_helper\n    constructor()\n")
        self.pin();result=self.run_gate()
        self.assertIn('Pinned local module is shadowed',result.stderr)

    def test_existing_run_is_untouched(self):
        self.output.mkdir();receipt=self.output/'source_receipt.json';receipt.write_text('existing run')
        result=self.run_gate();self.assertIn('Output already exists',result.stderr)
        self.assertEqual(receipt.read_text(),'existing run')

    def test_unapproved_registry_identity_fails(self):
        result=subprocess.run([sys.executable,str(REPO/'experiments/source_consistency.py'),'run','--model-id','unapproved-fixture','--output',str(self.output)],capture_output=True,text=True)
        self.assertIn('no approved source pin',result.stderr);self.assertFalse(self.output.exists())

    def test_direct_training_guard_rejects_unregistered_entry(self):
        environment=os.environ.copy()
        environment['PYTHONPATH']=str(REPO/'src')
        code="import sys;sys.path.insert(0,sys.argv[1]);from source_consistency import require_registered_runner;require_registered_runner('fixture_runner.py')"
        result=subprocess.run([sys.executable,'-c',code,str(REPO/'experiments')],
            cwd=self.root,capture_output=True,text=True,env=environment)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Direct training is disabled',result.stderr)
        self.assertFalse(self.output.exists())

if __name__=='__main__':unittest.main()
