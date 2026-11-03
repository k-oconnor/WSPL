# November 3 repaired-horizon result and numerical interface repair

Kevin explicitly approved horizon-fix digest
`75e33ff7d7d55a1500cfd9521a95f26dc1443f5a6354c3cde595749b87f6931b`.
It was registered and launched through the unchanged source gate as PID 18736,
using the external isolated CPU environment. Both run result and source receipt
ultimately record failure; no process remains active. Original artifacts and
checkpoint remain unchanged in results/approach_v1_horizonfix.

## What ran

- Development: all four reference starts succeeded; all four zero controls
  exhausted their proper 120-call skill budget. All paired start hashes matched.
- Source: all four reference trajectories succeeded, yielding 154 command-labelled
  rows, shared by both gradient arms.
- Fitting: both arms completed 2,000 full-batch AdamW updates, 308,000 row
  presentations each. Last pre-update command MSE was 0.000213251 for MLP
  (initial 0.680640), and 0.003921668 for RMV (initial 0.857984). These are
  training imitation errors, not physical success or held-out prediction scores.
- Evaluation seed 1001: zero timed out; reference, analytic MLP and analytic RMV
  each succeeded in 43 calls, at approximately 5.918 mm final target error.
  The analytic controllers encode the known proportional teacher; these are
  inspected engineering constructions, not a blinded LLM authorship trial.
- The scalar check then raised a type exception before either learned model's
  physical evaluation. Seven evaluation seeds were never attempted. No full
  evaluation success rate or learned motor performance is established.

Recorded run wall time: 23.296 seconds. Sixteen episode attempts, 320 preparation
calls and 1,044 approach calls; all reset/preparation costs and traces are retained.
The external environment was unchanged after execution. Prior failed-launch and
original-horizon-run costs remain separate historical costs; they are not erased
by this attempt's totals. There were no external API calls or GPU operations.

## Failure and additional fidelity finding

The physical adapter's geometry-derived aperture was a finite NumPy float64
scalar. PyTorch accepted it, but the scalar reference deliberately accepts only
builtin numeric values. Source fitting had passed Python floats from tensor
tolist; evaluation passed raw frame channels. The interface mismatch, not a NaN,
triggered `ValueError: Expected finite numeric value` after the analytic RMV's
successful episode. The original gate correctly reports failure.

A read-only audit then evaluated frozen checkpoint weights on all 1,044 previously
saved input rows. It made no new simulator calls or optimizer updates and did
not alter the checkpoint (SHA-256
`d1dabf6e20ba511159fb7fd355eb06918a3f538918b04ca696981a2ac76a1b74`).
It found an additional numerical issue: the fitted RMV's FP32 arithmetic versus
the double-precision scalar graph differed by up to 3.4810e-6 on those probes.
Using exactly matching FP32 inputs reduced that to 2.1917e-6, still above the
declared 2e-6 tolerance. The fitted RMV had passed the original source-row check
(1.6721e-6); these broader saved probes exposed the limitation. These are numerical
probes of existing trajectories, not learned-model physical rollouts.

## Prepared replacement

1. The checker constructs the same FP32 input tensor used by the controller and
   feeds its tolist values to the independent scalar reference. This converts
   NumPy scalars to Python floats and compares identical input-port values.
   Both architectures' controller input ports remain FP32. No input channel,
   range, target or physical threshold changes.
2. RMV normalization, literals, logical accumulation and affine readout now use
   FP64 arithmetic; final commands cast back to input dtype (FP32 in the study).
   Parameters, gradient storage and AdamW state remain FP32. Casts are ordinary
   differentiable operations; hard clamps and nonnegative projection remain.
   MLP arithmetic remains FP32. This is an explicit numerical implementation
   change, so the replacement is not a compute-matched comparison and may follow
   a slightly different RMV optimization trajectory. No tolerance is relaxed.
3. New unit regressions exercise actual adapter NumPy scalar types and dense
   high-gain logical arithmetic, checking fidelity and FP32 parameter gradients.
   All 102 unit tests pass.

Applying the repaired arithmetic to the same frozen checkpoint/input probes
gives maximum discrepancies of 2.9803e-8 for analytic RMV and 2.9601e-8 for fitted
RMV. This is a read-only numerical repair audit, not a rerun or a new physical
learning result. Saved weights, old receipts and original manifests are intact.

The replacement starts fresh with the same architecture, seed, four source
trajectories, development/evaluation seed lists, optimizer/update budgets and
physical task. Evaluation seed 1001 has now been partly inspected; seeds
1002-1008 have not run. This remains an engineering panel, not a sealed general
transfer benchmark. Repeated attempts and their costs must be reported together.

Candidate: experiments/source_pins/approach_v1_fidelityfix.manifest.json

SHA-256:
`d770313a57ec51094e3147e6876c975835e5ccd55950e057d7bde2c3e247aaa4`

Model ID: approach-v1-fidelityfix. Proposed output:
results/approach_v1_fidelityfix. Four inventoried Python files change relative to
the approved horizon fix: neural.py, run_approach.py, test_neural.py and
test_approach.py. All constructor/configuration values are unchanged; external
package/assets remain identical. No new source is approved or queued. AGENTS.md
requires exact replacement-digest approval before registering and launching.

Expected runtime remains several minutes with a 20-minute wall guard, CPU only,
no APIs, one final/clean-stop checkpoint. Launch after approval through the source
gate using the ignored sim_python binding; never transfer prior approvals.
