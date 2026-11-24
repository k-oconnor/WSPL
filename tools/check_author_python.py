"""Engineering-only isolation probes; no model scoring or author requests."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from skillforge.windows_python_sandbox import AppContainer

PROBE = r'''
import sys, os, json, traceback
from pathlib import Path
log = open('probe.log', 'w', buffering=1)
sys.stdout = sys.stderr = log
out = {}
try:
    import numpy as np
    out['python'] = sys.version
    out['numpy'] = np.__version__
    out['sys_path'] = sys.path
    print(json.dumps(out), flush=True)
    import torch
    out['torch'] = torch.__version__
    out['cuda_available'] = torch.cuda.is_available()
    print(json.dumps(out), flush=True)
    if out['cuda_available']:
        x = torch.ones((32,32), device='cuda', requires_grad=True)
        (x.square().sum()).backward()
        out['cuda_backward'] = bool(torch.equal(x.grad, torch.full_like(x, 2)))
        torch.save({'value': x.detach()}, 'synthetic.pt')
        out['cuda_checkpoint'] = bool(torch.equal(torch.load('synthetic.pt', weights_only=True)['value'], x))
    out['complete'] = True
except BaseException:
    out['error'] = traceback.format_exc()
    traceback.print_exc()
Path('probe.json').write_text(json.dumps(out, indent=2))
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=r'C:\Users\kevin\wspl-author-python-v1')
    p.add_argument('--regular', action='store_true')
    args = p.parse_args()
    root = Path(args.root)
    work = root/('preflight-regular' if args.regular else 'preflight-lpac')
    work.mkdir(parents=True, exist_ok=True)
    script = work/'probe.py'
    script.write_text(PROBE)
    container = AppContainer('WSPL.AuthorPython.Preflight.'+('regular' if args.regular else 'lpac'))
    container.grant(root/'runtime')
    container.grant(work, writable=True)
    result = container.run(root/'runtime/python.exe', script, work, timeout=90, lpac=not args.regular)
    (work/'host_receipt.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    for name in ('probe.json', 'probe.log'):
        if (work/name).exists():
            print((work/name).read_text())


if __name__ == '__main__':
    main()
