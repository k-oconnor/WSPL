"""Build independent standalone Python from already-installed runtime copies."""
import json
from pathlib import Path
import shutil
import sys


def build(base, learner, target):
    base, learner, target = map(Path, (base, learner, target))
    if target.exists():
        raise ValueError('Fresh destination required')
    target.mkdir(parents=True)
    for name in ['python.exe', 'pythonw.exe', 'python3.dll', 'python313.dll',
                 'vcruntime140.dll', 'vcruntime140_1.dll', 'LICENSE.txt']:
        shutil.copy2(base/name, target/name)
    ignore = shutil.ignore_patterns('site-packages', '__pycache__', '*.pyc', '*.pyo')
    for name in ['Lib', 'DLLs']:
        shutil.copytree(base/name, target/name, ignore=ignore)
    shutil.copytree(learner/'Lib/site-packages', target/'Lib/site-packages',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    # Explicit isolated path file excludes registry paths and user-site packages.
    (target/'python313._pth').write_text('Lib\nDLLs\nLib/site-packages\nimport site\n')
    report = dict(base=str(base), copied_learner=str(learner), destination=str(target),
                  hardlinks=False, dependencies=json.loads((learner/'copied_dependencies.json').read_text()))
    (target/'build_receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    build(*sys.argv[1:])
