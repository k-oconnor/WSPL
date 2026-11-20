import importlib.util
from pathlib import Path
import tempfile
import json
import unittest

path=Path(__file__).resolve().parents[1]/'tools/verify_supervisor_runtime.py'
spec=importlib.util.spec_from_file_location('runtime_verifier',path)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


class RuntimeTests(unittest.TestCase):
    def test_content_and_inventory_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'binary';f.write_bytes(b'original')
            m=root/'inventory.json';m.write_text(json.dumps({'files':{str(f):v.digest(f)}}))
            identity=v.digest(m);self.assertEqual(v.verify(m,identity)['files'],1)
            f.write_bytes(b'changed')
            with self.assertRaises(ValueError):v.verify(m,identity)
            with self.assertRaises(ValueError):v.verify(m,'wrong')
    def test_missing_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=Path(tmp)/'inventory.json';m.write_text(json.dumps({'files':{str(Path(tmp)/'missing'):'x'}}))
            with self.assertRaises(FileNotFoundError):v.verify(m,v.digest(m))


if __name__=='__main__':unittest.main()
