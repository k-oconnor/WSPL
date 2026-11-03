# Completed pregrasp positioning pilot — November 3, 2026

The fidelity-repaired run completed all 60 episodes and both 2,000-update fitting
budgets. Run result and source-gate receipt both say complete. Project source
verified, external content unchanged, one pinned study constructor call. No
process remains active and no follow-on run is approved or queued.

Approved source digest:
`d770313a57ec51094e3147e6876c975835e5ccd55950e057d7bde2c3e247aaa4`.
The original horizon and scalar-interface failures are preserved in the earlier
launch/repair reports; they are not overwritten by this successful attempt.

## Physical outcomes

The task was positioning at a supplied standoff 12 cm above the settled object's
origin, with held orientation and open gripper. It was not grasping or pickup.
All methods received identical starting-frame hashes on each evaluation seed.

| Method | Evaluation success | Median final target error | Median approach calls |
| --- | ---: | ---: | ---: |
| Zero translation | 0/8 | 253.8 mm | 120 |
| Reference proportional controller | 8/8 | 6.9 mm | 40 |
| Analytic MLP weights | 8/8 | 6.9 mm | 40 |
| Analytic RMV weights | 8/8 | 6.9 mm | 40 |
| AdamW MLP | 5/8 | 9.0 mm | 45.5 |
| AdamW RMV | 3/8 | 56.4 mm | 120 |

Medians include all eight attempted starts, including timeouts. The fractional
median is the average of two integer call counts. Completion required <=1 cm
position error and <=0.03 m/s speed for three consecutive samples, not position
alone. The four development references also succeeded and all four zero controls
timed out; all four source demonstrations succeeded.

The MLP's three evaluation timeouts ended 41-43 mm from target with final speeds
around 0.009 m/s. RMV's five timeouts ended 56-86 mm away at around 0.016 m/s.
Thus those failures missed the position criterion; they were not merely cases
of arriving at the target too fast. These endpoint observations do not establish
the mechanism of the failure. Inspect full trajectories before claiming a stable
equilibrium, optimization failure or data-distribution cause.

## Training and numerical evidence

Both gradient arms used the same 154 rows from four teacher trajectories, 2,000
full-batch AdamW updates and 308,000 row presentations. Last pre-update training
MSE: MLP 0.000213251, RMV 0.003911980. Low imitation loss did not guarantee all
closed-loop trials succeeded. The architectures can represent a working controller
for this task, as the analytic constructions demonstrate; this study does not
separate limited state coverage, optimization and generalization as explanations
for the learned arms' errors.

RMV uses FP64 normalization/logical/readout arithmetic with FP32 parameters,
gradient storage, optimizer state and controller input/output ports. MLP uses
FP32 arithmetic. There is no compute-matched efficiency claim. The 2e-6 tolerance
was unchanged. Read-only final-checkpoint audits on each RMV's recorded visited
inputs found maximum scalar/neural motor discrepancies:

- Analytic RMV: 2.9803e-8.
- AdamW RMV: 2.9364e-8.

All recorded checks passed, including the runner's source/input fidelity checks.
Analytic parameters were confirmed unchanged bitwise from their construction;
all eight evaluation panels had identical paired starts. Checkpoint source digest
matches the approved manifest; both learned models record exactly 2,000 updates.

The analytic construction is directly readable: each translational output is
`clip(20 * target_error_m, -1, 1)` and gripper output is -1. In RMV it uses three
single-literal clauses/heads plus the signed affine readout. The full 56-clause
architecture still executes and is counted. The dense gradient-learned RMV is
exported faithfully but has not been simplified or shown to be human-readable.

## Interpretation and limits

This establishes that small fixed neural and grounded RMV weight artifacts can
execute the supplied positional feedback program through the simulated motor
interface. The low-level adapter contains no approach routine. It supplies the
ordinary OSC servo, explicit sensors and target binding.

It does not establish an LLM acquisition advantage. The analytic weights were
inspected constructions encoding the exact known teacher; the gradient models
received only four demonstrations. Writer/design cost is unmeasured, and there
was no blinded writer, independent skill invention, API call or matched prior
information. There is no grasp/contact-ready-pose validation, collision robustness,
physical hierarchical execution, learned causal mechanism or new-task transfer
claim. Eight starts in one scene family are an engineering screen, not a strong
reliability estimate. Seed 1001 had been partly inspected in the earlier failed
attempt; the failed attempts and all costs remain part of the record.

The next useful experiment should keep the architectures and observer fixed and
test richer source-state coverage or logged on-policy teacher corrections. This
would distinguish simple trajectory imitation from recovery under its own
closed-loop errors before expanding to grasp/contact or making acquisition claims.
It is a recommendation, not an approved next run.

## Cost and artifacts

Successful attempt: 56.266 recorded wall seconds on CPU; 60 episode resets,
1,200 preparation calls and 4,037 approach calls. No external APIs or GPU use.
Earlier failed pilot attempts add 34.296 recorded seconds and 1,546 attempted
motor/preparation calls, including one rejected native call. Across the three
physical attempts this is 90.562 recorded seconds and 6,783 attempted calls.
Installation, engineering preparation, conversation and the first source-gate
startup rejection are additional unmeasured costs, so these are not end-to-end
acquisition-efficiency numbers.

Primary artifacts: results/approach_v1_fidelityfix/results.json, source_receipt.json,
episode_*.json, source_pairs.json, two RMV exports, final.pt and result_audit.json.
Only one final checkpoint was created in the successful output. Its SHA-256 is
`a6ed5542ffacab0324bae3a2f105787bb4dd883a0ceab6aa7c24bb6ad6053146`.

The compact [audited summary](approach_v1_results_summary.json) records exact
counts, per-seed errors, gradient audits, fitting losses and checkpoint identity.
No weights or bulk trajectories are committed to Git. All 102 prior engineering
unit checks passed; no inventoried Python changed during the approved run.
