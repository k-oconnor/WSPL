# Full isolated author Python environment

Kevin selected full isolated Python on November 23, 2026. Engineering preparation
is complete. No new Astra requests, scientific training, Qwen task predictions,
or adaptation evaluation ran during this work. The earlier bounded prototype is
archived under `artifacts/robot_authoring_bounded_prototype` and was never launched.

## Runtime and execution

The dedicated runtime is `C:\Users\kevin\wspl-author-python-v1\runtime`:
Python 3.13.13, NumPy 2.4.4, PyTorch 2.12.1+cu126 and Transformers 5.14.1,
plus the dependency closure and WSPL LoRA bridge. Files are independent copies,
not hardlinks; the historical model, simulator and learner runtimes were not
modified. `python313._pth` selects only this runtime's standard library, DLLs and
packages. User-site paths and host environment credentials are not inherited.

`skillforge.author_python_workspace.PythonWorkspace` creates a distinct Windows
AppContainer profile per author. Its `execute(code, timeout=...)` method runs full
Python, including imports, numerical libraries, filesystem operations inside the
workspace and child processes. Files persist across calls; interpreter globals
do not. The runtime, supplied inputs and host-recorded request sources are read
only. The author workspace is writable. Receipts remain outside author access.
The author gets no network capabilities or inherited handles. A job object caps
aggregate committed process memory at 16 GiB and process count at 16, kills child
processes on exit, and enforces per-call wall time (at most 20 minutes). Logs are
bounded when returned to the author; linked output is rejected by the host reader.

This is native Windows OS process isolation, not a virtual machine or dedicated
GPU. Windows system resources granted to ordinary AppContainers remain readable;
the kernel and GPU are shared. The memory job limit is not a VRAM quota. Author
sessions should run serially under the study's additional aggregate budget.
Regular AppContainer is used: an initial zero-capability LPAC probe could not
start Python (exit 0xc0000022). No broad host ACLs or administrative changes were
made to work around that failure. WSL is not installed and was not needed.

The implementation follows Microsoft's [AppContainer launch API](https://learn.microsoft.com/en-us/windows/win32/secauthz/implementing-an-appcontainer).
The policy is checked on this machine rather than inferred solely from the API.

## Verified engineering fixtures

`tools/check_author_isolation.py` passed all 21 in-container assertions, plus
host-side controls, a second-process persistence test and descendant termination:

- Actual AppContainer token; host environment sentinel absent.
- Writable workspace; readable supplied input; writes to that input and runtime
  denied by Windows permissions.
- Reads of private-host, repository and other-author canaries denied; host
  canary write denied; host input bytes unchanged.
- External TCP connection denied with WinError 10013. Loopback connection timed
  out while the host positive control connected successfully; the host listener
  accepted no sandbox connection. An earlier test required a direct permission
  exception for loopback and failed; the corrected test checks both ends and
  preserves the actual timeout result instead of calling it an access error.
- A child Python process retained host-file denial. Attempted job breakaway was
  denied. Both timeout and ordinary parent exit terminated sleeping descendants;
  the launcher verifies zero active job processes before collecting any output.
- NumPy I/O, Transformers import, CUDA tensor computation/backpropagation,
  PyTorch checkpoint save and fresh-process reload all passed.
- Full-Python native factor construction and WSPL replay produced bit-identical
  synthetic LoRA arrays; the parent remained unchanged.

`tools/check_python_workspace.py` passed three wrapper checks: successful full
Python execution, persistent NumPy arrays across calls, and captured failure
traceback/nonzero exit status. All computations used synthetic fixtures.
Fifteen LoRA bridge unit checks and three output-collection checks also pass,
including hash-pinned construction bindings and refusal to read hardlinked or
out-of-workspace log output.

Receipts: `author_python_isolation_v1.json` and `author_python_workspace_v1.json`
beside this document. Full local logs and failed preliminary probe receipts are
under `C:\Users\kevin\wspl-author-python-v1`. The independent runtime file inventory
is `environment_manifest.json` in that directory: 22,111 files, 4,451,375,973 bytes,
SHA-256 `40edaf0b63eab1881648b5ca4d4ddc077ddb4c48c2e6ace0167091368681e84b`.
This environment inventory is not a scientific source manifest or approval.

## Reproduction and remaining study work

Build with `tools/build_author_python.py` from a dedicated installed interpreter
and learner environment, then `tools/package_author_wspl.py`. Run
`tools/check_author_isolation.py`, `tools/check_python_workspace.py`, and finally
`tools/inventory_author_python.py`. The tools record concrete local paths and
hashes; no downloads or model calls are implicit.

The matched author study still needs full-Python tool integration, bounded
development-evaluation accounting, independent native/WSPL artifact validation,
all-candidates-frozen final evaluation, a fixed adaptation control, and complete
source/resource pinning. Sandbox success does not establish those scientific
properties. Host consumers must validate data-only arrays and replay WSPL;
arbitrary sandbox output must never be imported or unpickled in the trusted host.
The exact scientific manifest approval required by AGENTS.md remains separate.
