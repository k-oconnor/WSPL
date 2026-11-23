"""Copy the installed dependency closure into a new venv, without hardlinks.

Reuses locally installed GPU wheels while excluding unrelated editable projects.
Call with the original interpreter and a newly created destination venv.
"""
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    target = Path(sys.argv[1]).resolve()
    cfg = target/'pyvenv.cfg'
    if not cfg.is_file() or target == Path(sys.prefix).resolve():
        raise ValueError('Separate venv destination required')
    site = target/'Lib/site-packages'
    pending = ['torch', 'transformers', 'numpy', 'setuptools']
    seen = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in seen:
            continue
        dist = metadata.distribution(name)
        seen[name] = dist.version
        for spec in dist.requires or []:
            requirement = Requirement(spec)
            if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                pending.append(requirement.name)
        if dist.files is None:
            raise ValueError('Distribution has no installed inventory: '+name)
        for rel in dist.files:
            if '..' in rel.parts or rel.suffix in ('.pyc', '.pyo') or '__pycache__' in rel.parts:
                continue
            src = Path(dist.locate_file(rel)).resolve()
            dst = (site/str(rel)).resolve()
            if not dst.is_relative_to(site):
                raise ValueError('Distribution path escapes target')
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    cfg.write_text(cfg.read_text().replace('include-system-site-packages = true',
                                         'include-system-site-packages = false'))
    report = dict(purpose='independent copies of installed wheels', packages=seen)
    (target/'copied_dependencies.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
