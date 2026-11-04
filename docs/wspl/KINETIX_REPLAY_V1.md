# Released Kinetix car-repair replay

Preparation only, November 4, 2026. No model calls, training or scored episodes
have run. The next study compares the supplied rounded parent and released repair
on `l/h2_car_ramp`. This is simulator integration and fixed-artifact replication;
it does not test WSPL authoring advantage.

## Identity and precision

Reference release: `RikiyaT/lm_understand-logs` at
`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`, experiment 4. Provenance is in
`provenance/csail_kinetix_reference_v1.json`. The training receipt identifies
official Kinetix commit `80ee9c292837c825cafaf9889323fdcc212efd1b` and JAX 0.4.38.
The isolated WSL CPU environment pins that commit, JAX/JAXlib 0.4.38 and Jax2D
1.0.1. Installed package metadata and critical Python/native files are recorded
and hashed separately from the repository source inventory.

Actor: 679 inputs, 128-unit ReLU embedding, five 128-unit tanh layers, 16 logits;
171,664 parameters. Parse the rounded integer-times-scale tensors shown to the
original author, rather than importing the full-precision PPO/value checkpoint.
The released edited NPZ is float64. The 51 published literal edits reproduce
every weight numerically; 3,145 zero signs differ because the printed integer
representation loses negative zero. Preserve the released edited artifact and
report this distinction instead of claiming bitwise reconstruction.

Use NumPy float64 actor arithmetic and stable float64 categorical probabilities;
the simulator observations and physics remain FP32. Original evaluator precision,
implementation and RNG procedure are not released. This explicit local choice
makes this a statistical replay, not exact episode reproduction. Future WSPL FP32
deployment must be tested separately; this replay does not silently cast the
released artifact to the editor's FP32 format.

## Fixed panel

| Setting | Value |
| --- | --- |
| Policies | Rounded parent; released edited actor |
| Episodes | 1,024 per policy, interleaved, all retained |
| Reset | Identical fixed level state each episode, no autoreset |
| Horizon | 256 environment calls, frame skip 2 |
| Physics | Released level parameters; dt 1/60; dense reward scale 0.2 |
| Actions | Independent categorical sizes 3/3/3/3/2/2; all six sampled |
| RNG | NumPy PCG64, seed 9,100,000 + episode index, paired across policies |
| Resources | CPU only, one BLAS/OMP/MKL thread, two-hour wall guard |
| Saving | Every episode; first three full observation/action trajectories per policy |

Report success/red/timeout/invalid counts, return and length. Invalid episodes
remain failures. Interruptions are labelled incomplete and prevent a full-panel
claim. Keep a single actor archive per policy; no repeated checkpoints. Atomic
`control.json` with `{"stop":true}` requests a clean stop. No automatic restart,
training or fresh authoring follows. Preserve source receipts, exact input bytes,
reconstructed parent, released repair and all episode records.

Upstream reports parent 0/256 successes and repaired 1,024/1,024. Those remain
external results. Inspect local replication before any new authoring comparison;
do not tune against replay scores. Any repair requires a new source identity.

## Preparation boundary

The bounded fixture ran three zero-action ticks, not policy episodes. It verified
679 FP32 inputs, initial body and green-wheel coordinates, finite state and three
clock advances. Seven focused tests passed in WSL, including that native fixture
and a full-budget synthetic one-tick orchestration test (not scored physics
episodes). Six non-native checks also pass under Windows NumPy 1.26.4. WSL
`pip check` reports no broken dependencies. The optional JAX profiler emits an
import warning; physics execution succeeds. No profiling measurements are claimed.

Constructor: `skillforge.kinetix_replay.ReplayStudy`. Runner:
`experiments/run_kinetix_replay.py::run`. Exact source-manifest approval is required
by `AGENTS.md`; no registry entry or approval exists yet.
