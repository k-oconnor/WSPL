"""Verify Linux runtime paths natively, preserving symlinks and filename case."""
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify(path,expected):
    if digest(path)!=expected:raise ValueError('Linux inventory identity changed')
    manifest=json.loads(Path(path).read_text())
    for name,value in manifest['files'].items():
        if digest(name)!=value:raise ValueError('Linux resource changed: '+name)
    return dict(status='verified',files=len(manifest['files']),inventory_sha256=expected)


if __name__=='__main__':print(json.dumps(verify(sys.argv[1],sys.argv[2])))
