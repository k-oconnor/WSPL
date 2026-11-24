"""Trusted-host collection must not follow sandbox-controlled output links."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from skillforge.author_python_workspace import bounded_file


class OutputCollectionTests(unittest.TestCase):
    def test_caps_returned_log_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'log.txt';path.write_bytes(b'x'*100)
            result=bounded_file(path,d,10)
            self.assertEqual(result,dict(text='x'*10,truncated=True))

    def test_rejects_outside_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d)/'work';work.mkdir()
            outside=Path(d)/'private';outside.write_text('not for author')
            self.assertEqual(bounded_file(outside,work)['rejected'],'linked output')

    def test_rejects_hardlinked_output(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d)/'work';work.mkdir()
            outside=Path(d)/'private';outside.write_text('not for author')
            target=work/'log';os.link(outside,target)
            self.assertEqual(bounded_file(target,work)['rejected'],'linked output')


if __name__=='__main__':unittest.main()
