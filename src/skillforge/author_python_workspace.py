"""Persistent file workspace and audited full-Python calls inside AppContainer.

This module launches engineering or separately source-approved author requests.
It does not authorize scientific runs or validate deployment artifacts. Host-side
consumers must replay WSPL / validate native arrays independently before use.
"""
import hashlib
import json
from pathlib import Path
import uuid

from .windows_python_sandbox import AppContainer


def bounded_file(path, work, maximum=65536):
    """Do not follow sandbox-controlled links when collecting a tool response."""
    path, work = Path(path), Path(work).resolve(strict=True)
    if not path.exists():
        return dict(text='', missing=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(work) or path.is_symlink() or path.stat().st_nlink != 1:
        return dict(text='', rejected='linked output')
    # Child job is already terminated; no concurrent sandbox writer remains.
    with path.open('rb') as f:
        data = f.read(maximum+1)
    return dict(text=data[:maximum].decode('utf8', errors='replace'), truncated=len(data)>maximum)


class PythonWorkspace:
    def __init__(self, root, runtime):
        self.root = Path(root).resolve()
        self.runtime = Path(runtime).resolve(strict=True)
        if self.root.exists():
            raise ValueError('New workspace root required; keep one instance per session')
        self.root.mkdir(parents=True)
        self.work, self.requests, self.receipts, self.inputs = (self.root/n for n in ('work', 'requests', 'receipts', 'inputs'))
        for path in (self.work, self.requests, self.receipts, self.inputs):
            path.mkdir()
        self.profile = AppContainer('WSPL.AuthorPython.Session.'+uuid.uuid4().hex)
        self.profile.grant(self.runtime)
        self.profile.grant(self.requests)
        self.profile.grant(self.inputs)
        self.profile.grant(self.work, writable=True)
        self.index = 0
        (self.root/'session.json').write_text(json.dumps(dict(profile=self.profile.name,
            sid=self.profile.sid_string, runtime=str(self.runtime), work=str(self.work),
            inputs=str(self.inputs), lpac=False, language='full Python', network_capabilities=[],
            persistence='files persist; each call starts a fresh Python process'), indent=2)+'\n')

    def execute(self, code, *, timeout=120):
        if not isinstance(code, str) or len(code.encode('utf8')) > 1_000_000:
            raise ValueError('Python request must be text under 1 MB')
        if not 0 < timeout <= 1200:
            raise ValueError('Timeout must be within 20 minutes')
        self.index += 1
        key = f'{self.index:03d}'
        source = self.requests/(key+'.py')
        wrapper = self.requests/(key+'-entry.py')
        source.write_text(code, encoding='utf8')
        log = self.work/('call-'+key+'-'+uuid.uuid4().hex+'.log')
        wrapper.write_text('import runpy, sys, traceback\n'
            f'sys.stdout = sys.stderr = open({str(log)!r}, "x", encoding="utf8", buffering=1)\n'
            f'sys.argv = [{str(source)!r}]\n'
            'try:\n'
            f'    runpy.run_path({str(source)!r}, run_name="__main__")\n'
            'except BaseException:\n'
            '    traceback.print_exc()\n'
            '    raise\n', encoding='utf8')
        receipt = self.profile.run(self.runtime/'python.exe', wrapper, self.work, timeout=timeout)
        receipt.update(request=key, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                       wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                       stdout=bounded_file(log, self.work))
        (self.receipts/(key+'.json')).write_text(json.dumps(receipt, indent=2)+'\n')
        return receipt
