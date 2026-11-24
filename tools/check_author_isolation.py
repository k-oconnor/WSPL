"""Adversarial engineering fixtures for the full Windows Python sandbox.

No scientific examples, model predictions, optimizers, or hosted author calls.
All host file-read probes target harmless canaries, never credentials.
"""
import argparse
import ctypes as C
from ctypes import wintypes as W
import json
from pathlib import Path
import socket
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from skillforge.windows_python_sandbox import AppContainer, api

PROBE = r'''
import ctypes as C
from ctypes import wintypes as W
import json, os, socket, subprocess, sys, traceback
from pathlib import Path
sys.stdout = sys.stderr = open('checks.log', 'w', buffering=1)
config = json.loads(Path(sys.argv[1]).read_text())
out = {'checks': {}, 'detail': {}}
def record(name, value, detail=None):
    out['checks'][name] = bool(value)
    if detail is not None:
        out['detail'][name] = detail
    print(name, value, detail, flush=True)
def denied(name, operation):
    try:
        operation()
    except PermissionError as e:
        record(name, True, str(e))
    except Exception as e:
        record(name, False, 'Not evidence of denial: '+repr(e))
    else:
        record(name, False, 'Unexpectedly permitted')
def token_is_appcontainer():
    k, a = C.WinDLL('kernel32', use_last_error=True), C.WinDLL('advapi32', use_last_error=True)
    k.GetCurrentProcess.restype = W.HANDLE
    a.OpenProcessToken.argtypes = [W.HANDLE, W.DWORD, C.POINTER(W.HANDLE)]
    a.GetTokenInformation.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.POINTER(W.DWORD)]
    k.CloseHandle.argtypes = [W.HANDLE]
    token, val, length = W.HANDLE(), W.DWORD(), W.DWORD()
    assert a.OpenProcessToken(k.GetCurrentProcess(), 8, C.byref(token))
    try:
        assert a.GetTokenInformation(token, 29, C.byref(val), 4, C.byref(length))
        return val.value == 1
    finally:
        k.CloseHandle(token)
try:
    record('appcontainer_token', token_is_appcontainer())
    record('environment_scrubbed', 'WSPL_HOST_CANARY' not in os.environ and 'CODEX_HOME' not in os.environ)
    Path('persistent.txt').write_text('first-call')
    record('workspace_write', Path('persistent.txt').read_text() == 'first-call')
    record('input_read', Path(config['input']).read_text() == 'readonly-input')
    denied('input_write_denied', lambda: Path(config['input']).write_text('bad'))
    denied('runtime_write_denied', lambda: (Path(sys.executable).parent/'tamper.txt').write_text('bad'))
    denied('host_canary_read_denied', lambda: Path(config['private']).read_bytes())
    denied('repository_canary_read_denied', lambda: Path(config['repo_canary']).read_bytes())
    denied('other_author_read_denied', lambda: Path(config['other']).read_bytes())
    denied('host_write_denied', lambda: Path(config['private']).write_text('bad'))
    def connect(host, port):
        with socket.socket() as s:
            s.settimeout(3)
            s.connect((host, port))
    try:
        connect('127.0.0.1', config['port'])
    except (PermissionError, TimeoutError) as e:
        record('loopback_unreachable', True, repr(e))
    else:
        record('loopback_unreachable', False)
    denied('external_network_denied', lambda: connect('1.1.1.1', 443))
    child_source = "from pathlib import Path\ntry:\n Path("+repr(config['private'])+").read_bytes()\nexcept PermissionError:\n Path('child-denied.txt').write_text('yes')\nelse:\n raise RuntimeError('Host read escaped')\n"
    child = subprocess.run([sys.executable, '-I', '-B', '-c', child_source], timeout=15)
    record('child_inherits_isolation', child.returncode == 0 and Path('child-denied.txt').read_text() == 'yes')
    denied('job_breakaway_denied', lambda: subprocess.run([sys.executable, '-I', '-c', 'pass'],
                                                        creationflags=0x01000000, timeout=15))
    import numpy as np
    x = np.arange(12, dtype=np.float32).reshape(3,4)
    np.savez('arrays.npz', x=x)
    record('numpy_io', np.array_equal(np.load('arrays.npz', allow_pickle=False)['x'], x))
    from skillforge.wspl_lora import identity, describe, source_for_pair, construct_wspl, construct_native
    parent = {'fixture.lora_A': np.array([[1,2,3],[0,-1,1]], dtype=np.float32),
              'fixture.lora_B': np.zeros((4,2), dtype=np.float32)}
    initial_id = identity(parent)
    pair_id = describe(parent)['projections']['fixture']['pair_id']
    program = source_for_pair(pair_id, 'for i in range(4) { B[i,0]=i/4; B[i,1]=-i/8; }')
    wspl, receipt = construct_wspl(parent, {'fixture': program}, expected_parent_id=initial_id)
    native, _ = construct_native(parent, {'fixture': {'A':parent['fixture.lora_A'].copy(),
                       'B':np.array([[i/4,-i/8] for i in range(4)], dtype=np.float32)}}, expected_parent_id=initial_id)
    record('native_wspl_bitwise_parity', identity(native) == identity(wspl))
    record('construction_preserves_parent', identity(parent) == initial_id)
    np.savez('synthetic-adapter.npz', **wspl)
    Path('synthetic-construction.json').write_text(json.dumps(receipt, indent=2))
    import torch, transformers
    record('transformers_import', bool(transformers.__version__))
    record('cuda_available', torch.cuda.is_available())
    t = torch.arange(32, device='cuda', dtype=torch.float32, requires_grad=True)
    t.square().sum().backward()
    record('cuda_backward', torch.equal(t.grad, 2*t))
    torch.save({'value': t.detach().cpu()}, 'synthetic.pt')
    record('checkpoint_write', torch.equal(torch.load('synthetic.pt', weights_only=True)['value'], t.cpu()))
    out['versions'] = {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                       'transformers': transformers.__version__}
except BaseException:
    out['error'] = traceback.format_exc()
    traceback.print_exc()
out['passed'] = bool(out['checks']) and all(out['checks'].values()) and 'error' not in out
Path('checks.json').write_text(json.dumps(out, indent=2))
sys.exit(0 if out['passed'] else 1)
'''

SECOND = r'''
import json, sys, traceback
from pathlib import Path
sys.stdout = sys.stderr = open('second.log', 'w', buffering=1)
try:
    import torch
    assert Path('persistent.txt').read_text() == 'first-call'
    assert torch.equal(torch.load('synthetic.pt', weights_only=True)['value'], torch.arange(32, dtype=torch.float32))
    Path('second.json').write_text(json.dumps({'persistent_workspace': True, 'checkpoint_reload': True}))
except BaseException:
    traceback.print_exc()
    raise
'''

TIMEOUT = r'''
import subprocess, sys, time
from pathlib import Path
p = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(600)'])
Path('descendant.pid').write_text(str(p.pid))
time.sleep(600)
'''


def alive(pid):
    k = C.WinDLL('kernel32', use_last_error=True)
    handle = api(k, 'OpenProcess', [W.DWORD, W.BOOL, W.DWORD], W.HANDLE)(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = W.DWORD()
        if not api(k, 'GetExitCodeProcess', [W.HANDLE, C.POINTER(W.DWORD)])(handle, C.byref(code)):
            raise C.WinError(C.get_last_error())
        return code.value == 259
    finally:
        api(k, 'CloseHandle', [W.HANDLE])(handle)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=r'C:\Users\kevin\wspl-author-python-v1')
    args = p.parse_args()
    root = Path(args.root)
    run_id = uuid.uuid4().hex[:10]
    test = root/('isolation-'+run_id)
    work, inputs, private, other = (test/n for n in ('work', 'inputs', 'private', 'other-author'))
    for path in (work, inputs, private, other):
        path.mkdir(parents=True)
    (inputs/'input.txt').write_text('readonly-input')
    (private/'canary.txt').write_text('private-host-canary')
    (other/'canary.txt').write_text('different-author-canary')
    repo_canary = Path(__file__).resolve().parents[1]/'data/robot_authoring_v1'/('host-canary-'+run_id+'.txt')
    repo_canary.parent.mkdir(parents=True, exist_ok=True)
    repo_canary.write_text('private-repository-canary')
    c = AppContainer('WSPL.AuthorPython.Check.'+run_id)
    c_other = AppContainer('WSPL.AuthorPython.Other.'+run_id)
    c.grant(root/'runtime')
    c.grant(inputs)
    c.grant(work, writable=True)
    c_other.grant(other, writable=True)
    import os
    os.environ['WSPL_HOST_CANARY'] = 'must-not-be-inherited'
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(4)
        port = listener.getsockname()[1]
        with socket.create_connection(('127.0.0.1', port), timeout=3):
            pass
        control, _ = listener.accept()
        control.close()
        config = dict(input=str(inputs/'input.txt'), private=str(private/'canary.txt'),
                      repo_canary=str(repo_canary), other=str(other/'canary.txt'), port=port)
        (work/'config.json').write_text(json.dumps(config))
        (work/'probe.py').write_text(PROBE)
        first = c.run(root/'runtime/python.exe', work/'probe.py', work, args=[work/'config.json'], timeout=120)
        listener.settimeout(.25)
        try:
            unexpected, _ = listener.accept()
        except TimeoutError:
            loopback_no_connection = True
        else:
            unexpected.close()
            loopback_no_connection = False
    result = dict(profile=c.name, first=first, root=str(test),
                  loopback_host_positive_control=True, loopback_no_sandbox_connection=loopback_no_connection)
    if (work/'checks.json').exists():
        result['checks'] = json.loads((work/'checks.json').read_text())
    (work/'second.py').write_text(SECOND)
    result['second'] = c.run(root/'runtime/python.exe', work/'second.py', work, timeout=90)
    if (work/'second.json').exists():
        result['persistence'] = json.loads((work/'second.json').read_text())
    (work/'timeout.py').write_text(TIMEOUT)
    result['timeout'] = c.run(root/'runtime/python.exe', work/'timeout.py', work, timeout=3)
    time.sleep(.3)
    if (work/'descendant.pid').exists():
        result['timeout_killed_descendant'] = not alive(int((work/'descendant.pid').read_text()))
    (work/'orphan.py').write_text(TIMEOUT.replace('time.sleep(600)\n', '# parent exits immediately\n'))
    # The replacement above leaves the quoted child command intact.
    result['ordinary_exit'] = c.run(root/'runtime/python.exe', work/'orphan.py', work, timeout=10)
    result['ordinary_exit_killed_descendant'] = not alive(int((work/'descendant.pid').read_text()))
    result['host_inputs_unchanged'] = ((private/'canary.txt').read_text() == 'private-host-canary'
                                       and (inputs/'input.txt').read_text() == 'readonly-input')
    result['passed'] = (first['exit_code'] == 0 and result.get('checks', {}).get('passed')
                        and result['second']['exit_code'] == 0 and result.get('persistence', {}).get('checkpoint_reload')
                        and result['timeout']['timed_out'] and result.get('timeout_killed_descendant')
                        and result['ordinary_exit']['exit_code'] == 0 and result['ordinary_exit_killed_descendant']
                        and result['host_inputs_unchanged'] and loopback_no_connection)
    (test/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
