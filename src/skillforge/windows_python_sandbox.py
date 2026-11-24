"""Full Python in a Windows AppContainer; never an AST-based substitute.

Host-side launcher. Each author must have a distinct profile and workspace.
No capabilities, no inherited handles, sanitized environment, job-tree timeout.
Runtime and input ACL grants must be to dedicated copies, not private host trees.
"""
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import subprocess
import time


class SI(C.Structure):
    _fields_ = [('cb', W.DWORD), ('lpReserved', W.LPWSTR), ('lpDesktop', W.LPWSTR),
                ('lpTitle', W.LPWSTR), ('dwX', W.DWORD), ('dwY', W.DWORD),
                ('dwXSize', W.DWORD), ('dwYSize', W.DWORD), ('dwXCountChars', W.DWORD),
                ('dwYCountChars', W.DWORD), ('dwFillAttribute', W.DWORD),
                ('dwFlags', W.DWORD), ('wShowWindow', W.WORD), ('cbReserved2', W.WORD),
                ('lpReserved2', C.c_void_p), ('hStdInput', W.HANDLE),
                ('hStdOutput', W.HANDLE), ('hStdError', W.HANDLE)]


class SIX(C.Structure):
    _fields_ = [('StartupInfo', SI), ('lpAttributeList', C.c_void_p)]


class PI(C.Structure):
    _fields_ = [('hProcess', W.HANDLE), ('hThread', W.HANDLE),
                ('dwProcessId', W.DWORD), ('dwThreadId', W.DWORD)]


class Caps(C.Structure):
    _fields_ = [('sid', C.c_void_p), ('capabilities', C.c_void_p),
                ('count', W.DWORD), ('reserved', W.DWORD)]


class BasicLimit(C.Structure):
    _fields_ = [('processTime', C.c_longlong), ('jobTime', C.c_longlong),
                ('flags', W.DWORD), ('minWorking', C.c_size_t), ('maxWorking', C.c_size_t),
                ('activeProcesses', W.DWORD), ('affinity', C.c_size_t),
                ('priority', W.DWORD), ('scheduling', W.DWORD)]


class IO(C.Structure):
    _fields_ = [(n, C.c_ulonglong) for n in ('readOps', 'writeOps', 'otherOps',
                                           'readBytes', 'writeBytes', 'otherBytes')]


class ExtendedLimit(C.Structure):
    _fields_ = [('basic', BasicLimit), ('io', IO), ('processMemory', C.c_size_t),
                ('jobMemory', C.c_size_t), ('peakProcess', C.c_size_t), ('peakJob', C.c_size_t)]


class Accounting(C.Structure):
    _fields_ = [(n, C.c_longlong) for n in ('userTime', 'kernelTime', 'periodUser', 'periodKernel')] + [
        (n, W.DWORD) for n in ('faults', 'totalProcesses', 'activeProcesses', 'terminatedProcesses')]


def api(dll, name, args, result=W.BOOL):
    fn = getattr(dll, name)
    fn.argtypes, fn.restype = args, result
    return fn


def check(value):
    if not value:
        raise C.WinError(C.get_last_error())
    return value


class AppContainer:
    def __init__(self, name):
        if os.name != 'nt':
            raise RuntimeError('Windows required')
        self.name = name
        self.k = C.WinDLL('kernel32', use_last_error=True)
        self.u = C.WinDLL('userenv', use_last_error=True)
        self.a = C.WinDLL('advapi32', use_last_error=True)
        self.sid = C.c_void_p()
        create = api(self.u, 'CreateAppContainerProfile',
                     [W.LPCWSTR, W.LPCWSTR, W.LPCWSTR, C.c_void_p, W.DWORD, C.POINTER(C.c_void_p)], C.c_long)
        hr = create(name, name, 'Isolated WSPL Python author workspace', None, 0, C.byref(self.sid))
        if hr == C.c_long(0x800700b7).value:
            hr = api(self.u, 'DeriveAppContainerSidFromAppContainerName',
                     [W.LPCWSTR, C.POINTER(C.c_void_p)], C.c_long)(name, C.byref(self.sid))
        if hr < 0:
            raise OSError(f'AppContainer profile HRESULT {hr & 0xffffffff:08x}')
        value = W.LPWSTR()
        check(api(self.a, 'ConvertSidToStringSidW', [C.c_void_p, C.POINTER(W.LPWSTR)])(self.sid, C.byref(value)))
        self.sid_string = value.value
        api(self.k, 'LocalFree', [C.c_void_p], C.c_void_p)(value)

    def grant(self, path, *, writable=False):
        path = Path(path).resolve(strict=True)
        # Require callers to explicitly dedicate each granted directory.
        if not path.is_dir() or len(path.parts) < 4:
            raise ValueError('Grant requires a dedicated, non-root directory')
        rights = '(OI)(CI)(M)' if writable else '(OI)(CI)(RX)'
        subprocess.run(['icacls', str(path), '/grant', f'*{self.sid_string}:{rights}', '/T', '/Q'],
                       check=True, capture_output=True)
        if writable:
            subprocess.run(['icacls', str(path), '/setintegritylevel', '(OI)(CI)L', '/T', '/Q'],
                           check=True, capture_output=True)

    def run(self, python, script, workspace, *, args=(), timeout=120, lpac=False, memory_gib=16):
        python, script, workspace = (Path(p).resolve(strict=True) for p in (python, script, workspace))
        temp = workspace/'tmp'
        temp.mkdir(exist_ok=True)
        env = dict(SystemRoot=os.environ['SystemRoot'], WINDIR=os.environ['SystemRoot'],
                   PATH=str(python.parent)+';'+os.environ['SystemRoot']+'\\System32',
                   TEMP=str(temp), TMP=str(temp), USERPROFILE=str(workspace),
                   LOCALAPPDATA=str(workspace), APPDATA=str(workspace), HOME=str(workspace),
                   HF_HOME=str(workspace/'hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                   CUDA_CACHE_PATH=str(workspace/'cuda-cache'), OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        block = C.create_unicode_buffer('\0'.join(f'{k}={v}' for k,v in sorted(env.items()))+'\0\0')
        command = C.create_unicode_buffer(subprocess.list2cmdline([str(python), '-I', '-B', str(script), *map(str,args)]))
        init = api(self.k, 'InitializeProcThreadAttributeList', [C.c_void_p, W.DWORD, W.DWORD, C.POINTER(C.c_size_t)])
        update = api(self.k, 'UpdateProcThreadAttribute', [C.c_void_p, W.DWORD, C.c_size_t, C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p])
        size = C.c_size_t()
        count = 2 if lpac else 1
        init(None, count, 0, C.byref(size))
        attrs = C.create_string_buffer(size.value)
        check(init(attrs, count, 0, C.byref(size)))
        caps = Caps(self.sid, None, 0, 0)
        check(update(attrs, 0, 0x20009, C.byref(caps), C.sizeof(caps), None, None))
        policy = W.DWORD(1)
        if lpac:
            check(update(attrs, 0, 0x2000f, C.byref(policy), C.sizeof(policy), None, None))
        si = SIX()
        si.StartupInfo.cb = C.sizeof(si)
        si.lpAttributeList = C.cast(attrs, C.c_void_p)
        pi = PI()
        close = api(self.k, 'CloseHandle', [W.HANDLE])
        job = check(api(self.k, 'CreateJobObjectW', [C.c_void_p, W.LPCWSTR], W.HANDLE)(None, None))
        limits = ExtendedLimit()
        limits.basic.flags = 0x2000 | 0x8 | 0x200  # Kill on close, process count, aggregate memory.
        limits.basic.activeProcesses = 16
        limits.jobMemory = memory_gib * 1024**3
        started = time.monotonic()
        try:
            check(api(self.k, 'SetInformationJobObject', [W.HANDLE, C.c_int, C.c_void_p, W.DWORD])(
                job, 9, C.byref(limits), C.sizeof(limits)))
            check(api(self.k, 'CreateProcessW', [W.LPCWSTR, W.LPWSTR, C.c_void_p, C.c_void_p,
                  W.BOOL, W.DWORD, C.c_void_p, W.LPCWSTR, C.c_void_p, C.POINTER(PI)])(
                str(python), command, None, None, False, 0x80000|0x400|0x4|0x8000000,
                block, str(workspace), C.byref(si), C.byref(pi)))
            check(api(self.k, 'AssignProcessToJobObject', [W.HANDLE, W.HANDLE])(job, pi.hProcess))
            if api(self.k, 'ResumeThread', [W.HANDLE], W.DWORD)(pi.hThread) == 0xffffffff:
                raise C.WinError(C.get_last_error())
            wait = api(self.k, 'WaitForSingleObject', [W.HANDLE, W.DWORD], W.DWORD)
            waited = wait(pi.hProcess, int(timeout*1000))
            if waited not in (0, 258):
                raise C.WinError(C.get_last_error())
            timed_out = waited == 258
            if timed_out:
                check(api(self.k, 'TerminateJobObject', [W.HANDLE, W.UINT])(job, 124))
                wait(pi.hProcess, 10000)
            code = W.DWORD()
            check(api(self.k, 'GetExitCodeProcess', [W.HANDLE, C.POINTER(W.DWORD)])(pi.hProcess, C.byref(code)))
            # Drain children before the host reads any sandbox-controlled output.
            # Kill-on-close alone does not establish that no writer remains yet.
            check(api(self.k, 'TerminateJobObject', [W.HANDLE, W.UINT])(job, 125))
            accounting = Accounting()
            deadline = time.monotonic()+10
            while True:
                check(api(self.k, 'QueryInformationJobObject',
                          [W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p])(
                              job, 1, C.byref(accounting), C.sizeof(accounting), None))
                if not accounting.activeProcesses:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('Sandbox descendants did not drain; refuse output collection')
                time.sleep(.02)
            return dict(exit_code=code.value, timed_out=timed_out, pid=pi.dwProcessId,
                        seconds=time.monotonic()-started, profile=self.name, sid=self.sid_string,
                        lpac=lpac, network_capabilities=[], inherited_handles=False,
                        active_processes_at_return=accounting.activeProcesses)
        finally:
            # Also kills orphaned descendants on ordinary completion or launcher failure.
            if pi.hProcess:
                api(self.k, 'TerminateProcess', [W.HANDLE, W.UINT])(pi.hProcess, 125)
                close(pi.hProcess)
            if pi.hThread:
                close(pi.hThread)
            close(job)
            api(self.k, 'DeleteProcThreadAttributeList', [C.c_void_p], None)(attrs)

