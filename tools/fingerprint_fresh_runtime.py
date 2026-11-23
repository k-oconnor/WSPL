"""Inventory interpreter, standard library, installed code/native libraries/assets.

No model invocation. The caller runs this under each actual execution interpreter.
Bytecode/cache and installation transient files are excluded; .py/.pyd/.dll and
all simulator assets are included. The learner currently binds shared packages.
"""
import hashlib
import json
import os
from pathlib import Path
import site
import sys


def inventory():
    roots = [Path(sys.base_prefix)/'Lib', Path(sys.base_prefix)/'DLLs']
    roots += [Path(p) for p in site.getsitepackages() if Path(p).is_dir() and Path(p).name == 'site-packages']
    if site.ENABLE_USER_SITE and Path(site.getusersitepackages()).is_dir():
        roots.append(Path(site.getusersitepackages()))
    files = set()
    for root in roots:
        for folder, directories, names in os.walk(root):
            excluded = {'__pycache__', '.git', '.cache'}
            if root.name != 'site-packages':
                excluded.add('site-packages')
            directories[:] = [d for d in directories if d not in excluded]
            for name in names:
                path = Path(folder)/name
                if path.suffix not in ('.pyc', '.pyo'):
                    files.add(path.resolve())
    files.add(Path(sys.executable).resolve())
    for prefix in (sys.prefix, sys.base_prefix):
        for pattern in ('python*.exe', '*.dll', '*.zip', 'pyvenv.cfg', 'windows_patch.json', 'copied_dependencies.json'):
            files.update(p.resolve() for p in Path(prefix).glob(pattern) if p.is_file())
    result = {}
    for path in sorted(files):
        with path.open('rb') as stream:
            result[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(executable=sys.executable, version=sys.version, prefix=sys.prefix,
                shared_packages=sys.prefix != sys.base_prefix and any(
                    str(sys.base_prefix) in p for p in site.getsitepackages()), resources=result)


if __name__ == '__main__':
    # file_digest is unavailable on Python 3.10.
    if not hasattr(hashlib, 'file_digest'):
        def file_digest(stream, algorithm):
            result = hashlib.new(algorithm)
            while chunk := stream.read(1024*1024):
                result.update(chunk)
            return result
        hashlib.file_digest = file_digest
    report = inventory()
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(executable=report['executable'], resources=len(report['resources']))))
