# Reworked Breakout launch identity

Completion update: run and source receipt completed at03:00PST, November4.
The [audited results](BREAKOUT_REWORKED_V1_RESULTS.md) report mean reward
370.516native/391.685WSPL and a passed descriptive progression rule. The text
below preserves launch history; it is not current running status.

November 4, 2026. **Approved and launched at 01:22 PST (09:22 UTC).**

Kevin explicitly selected "Approve this manifest and launch the reworked
Breakout comparison" for the digest below. It was reverified and registered
in both main and the isolated checkout. Hidden launcher PID724, worker PID38504.
The source gate passed and the first native proposal's fresh thread/turn started.
Launch stderr is empty. This is startup evidence, not a completed result.

[Live status](../../../Skillforge-wsp-breakout-v1/results/wsp_breakout_v1/status.json)
and [startup snapshot](wsp_breakout_v1_launch.json). Do not launch a duplicate.
No inventoried source or bound resource may change in the isolated checkout.

- [Protocol](BREAKOUT_REWORKED_V1.md) and [shared interface](BREAKOUT_REWORKED_INTERFACE_V1.md).
- Model ID: `wsp-breakout-v1`.
- Manifest: `experiments/source_pins/wsp_breakout_v1.manifest.json`.
- SHA-256: `e842840acc4c8229f370e37c456bf6898f88f676a13e3dde36b2b3ba5085acc4`.
- Isolated checkout: `C:/Users/kevin/OneDrive/Desktop/Skillforge-wsp-breakout-v1`.
- Output: `results/wsp_breakout_v1` in that checkout.
- Runtime: `C:/Users/kevin/skillforge-runtime/venv-wspl-match/Scripts/python.exe`.

The isolated checkout is a frozen snapshot over Git base `29eb1cd`, with new
source copied from main. The Git base alone does not identify the implementation;
all 95 inventoried Python files match the exact manifest. All ten bound auxiliary
files, core runtime and CLI executable hashes matched immediately before launch.
The exact new approval is registered; prior approvals were not reused.

Validation: [full receipt](wsp_breakout_preparation_validation.json),
[unit log](wsp_breakout_preparation_tests.txt),
[offline prompt checks](wsp_breakout_offline_preflight.json),
[isolated verification](wsp_breakout_prelaunch_verification.json).
426 primary tests:425pass/one native skip, covered by12/12native tests separately.
All19new harness tests pass; all three archived native controllers reconstruct
bitwise. Both actual prompts pass offline in main and the isolated checkout.
No model request or scored game episode was part of preparation. The separately
approved collection is now active; inspect live state for progress.

Launched through `experiments/source_consistency.py run --model-id
wsp-breakout-v1 --output results/wsp_breakout_v1`, with OPENBLAS_NUM_THREADS,
OMP_NUM_THREADS and MKL_NUM_THREADS set to1 before interpreter startup. Stdout,
stderr, PID, approval and launch receipts are preserved alongside the output
directory. Do not automatically retry. Main remains available for development.

Six fresh Astra/xhigh calls;20minutes/invocation,4hours overall, expected1-2hours.
Existing Codex account usage; no hard dollar/output-token cap or API-key fallback.
1,000 episodes per valid proposal, CPU only; no training, local GPU or feedback.
The deployed tanh policy is evaluated empirically; formal checking is unsupported.
Kinetix is the next requested domain, with its own preparation and source approval.
