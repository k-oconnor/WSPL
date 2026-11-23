"""Reproducible Windows-only robosuite 1.4.1 installation repair.

Run with the dedicated simulation interpreter. Modifies only this venv, following
robosuite's Windows installation notes; records original and replacement hashes.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sys.platform != 'win32' or sys.prefix == sys.base_prefix:
        raise ValueError('Dedicated Windows virtual environment required')
    if importlib.metadata.version('robosuite') != '1.4.1':
        raise ValueError('Only robosuite 1.4.1 patch is supported')
    packages = Path(sys.prefix)/'Lib/site-packages'
    target = packages/'robosuite/utils/binding_utils.py'
    before = sha(target)
    text = target.read_text(encoding='utf-8')
    old = 'os.environ["MUJOCO_GL"] = "egl"'
    new = 'os.environ["MUJOCO_GL"] = "wgl" if _SYSTEM == "Windows" else "egl"'
    if text.count(old) == 1:
        target.write_text(text.replace(old, new), encoding='utf-8', newline='\n')
    elif text.count(new) != 1:
        raise ValueError('Unexpected robosuite binding source')
    original = packages/'mujoco/mujoco.dll'
    copied = packages/'robosuite/utils/mujoco.dll'
    shutil.copyfile(original, copied)
    report = dict(package='robosuite', version='1.4.1', binding_before_sha256=before,
                  binding_after_sha256=sha(target), dll_sha256=sha(copied),
                  source='https://robosuite.ai/docs/installation.html#installing-on-windows')
    receipt = Path(sys.prefix)/'windows_patch.json'
    if not receipt.exists():
        receipt.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
