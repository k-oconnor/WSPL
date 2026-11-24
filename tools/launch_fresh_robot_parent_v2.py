"""Foreground watchdog for a separately registered source-gated run.

Start this wrapper hidden with PowerShell. It neither creates nor changes approval.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
from source_consistency import REGISTRY, contained, verify


def main():
    pin = json.loads(REGISTRY.read_text())['models'].get('fresh-robot-parent-v2')
    if pin is None:
        raise ValueError('Exact source approval must be separately registered first')
    manifest = verify(ROOT, contained(ROOT, pin['manifest']), pin['sha256'])
    cfg = manifest['runner_config']
    out = ROOT/'results/fresh_robot_parent_v2'
    if out.exists():
        raise ValueError('Prior output exists; no automatic retry/resume')
    out.parent.mkdir(exist_ok=True)
    with out.with_suffix('.stdout.log').open('xb') as stdout, \
            out.with_suffix('.stderr.log').open('xb') as stderr:
        p = subprocess.Popen([cfg['learner_python'], '-I', str(ROOT/'experiments/source_consistency.py'),
             'run', '--model-id', manifest['model_id'], '--output', str(out)],
             cwd=ROOT, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        record = dict(pid=p.pid, manifest_sha256=pin['sha256'], started_unix=time.time())
        out.with_suffix('.launch.json').write_text(json.dumps(record, indent=2)+'\n')
        try:
            result = p.wait(timeout=cfg['wall_guard_seconds'])
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if p.poll() is None:
                subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], check=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
                p.wait(timeout=30)
            record.update(status='terminated_by_watchdog', returncode=p.returncode)
        else:
            record.update(status='exited', returncode=result)
        record['finished_unix'] = time.time()
        out.with_suffix('.exit.json').write_text(json.dumps(record, indent=2)+'\n')
        if record['returncode'] != 0:
            raise SystemExit(record['returncode'] or 1)


if __name__ == '__main__':
    main()
