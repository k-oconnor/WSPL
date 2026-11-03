# Released Breakout artifact replay

Prepared November 3, 2026. This is the first step of CSAIL_COMPARISON.md: replay
the authors' released Breakout policy and a uniform zero-weight control. No new
Astra authoring, PPO training, WSPL advantage or product win is tested here.
Scientific execution requires the exact source approval in ../../AGENTS.md.

## Reference identity

Upstream: https://github.com/RikiyaT/lm_understand-logs at commit
`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.
The [Experiment 5 protocol](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch/README.md)
and original prompt define the observation conversion, architecture and evaluation.
Nine downloaded reference files were verified against Git blob identities and
SHA-256. Metadata and license provenance are in
[provenance](../../provenance/csail_breakout_reference_v1.json); weights and prompts
remain in ignored data/csail_breakout_reference_v1, with upstream notices retained.

Artifact hand.npz SHA-256:
`6e8d42658aed3704c7bf6fb67015e9468cc03f1d67cb8a4fd192e6534fcccb4a`.
It contains six FP32 tensors, 20,486 parameters, input-by-output matrix orientation,
600 -> 32 -> 32 -> 6 with tanh hidden activations. The loader rejects different
keys/shapes/dtypes, nonfinite values, unsafe pickle loading and stale hashes.
The installed MinAtar Breakout source matches the game source embedded in the
original prompt. The PyPI wheel identity is recorded in provenance.

## Fixed evaluation

| Setting | Value |
| --- | --- |
| Arms | Released fixed policy; same-shape uniform zero-weight control |
| Episodes | 1,000 per arm, all retained |
| Horizon | 10,000 environment calls per episode |
| Environment | MinAtar 1.0.15 Breakout, sticky actions 0.1 |
| Input | Paddle -> channel 0; bricks -> 2; ball -> 3; trail -> 4; channels 1 and 5 empty; row-major flattening |
| Inference | NumPy 1.26.4 FP32 matmul/tanh; stable FP64 softmax over all six logits |
| Action sampler | NumPy PCG64; categorical inverse CDF; no greedy or minimal-action masking |
| Seeds | Environment 310000 + episode index; action RNG 410000 + index; same mapping in both arms |
| Reset | Fresh environment per episode, seeded then reset; wrapper previous action starts at no-op |
| Resources | CPU only, one BLAS/OMP/MKL thread; no rendering, GPU, APIs or optimizer |
| Wall guard | 7,200 seconds; partial run marked incomplete/paused |

The upstream evaluator implementation and its original RNG procedure/seeds are
not supplied in the released Experiment 5 folder. Our new seed schedule,
fresh-environment reset, FP32 NumPy implementation and FP64 action sampler are
explicit implementation choices. This is a protocol-level statistical replay,
not bitwise reproduction of the original episodes or a verified numerical match.

## Output and interpretation

Each completed episode logs seeds, return, length, sampled-action counts,
environment termination and horizon truncation. Truncated episodes retain their
earned return. Incomplete interrupted episodes are labelled and excluded from
completed-episode summaries; their absence prevents a full-run claim.

Report mean return, sample standard error, extrema, mean length and truncations
for both arms. Compare the released-policy mean with the upstream recorded mean
517.411 and SE 7.416952829044303 using a descriptive difference/combined-SE value.
This is a diagnostic, not a predeclared significance test, proof or performance
win gate. Episode variability of one fixed policy is not authoring variability.
PPO scores remain external references; their checkpoints are not in this release.

The source-gated runner checks the reference bundle, critical NumPy/MinAtar
Python/native-binary fingerprints, all installed package versions, auxiliary
protocol/requirements files and frozen policy identities before/after use. It
instantiates the supplied pinned constructor once per arm. Rendering packages are
version-recorded but their code is outside the critical-core fingerprint.

Outputs: source_receipt.json, approved_manifest.json, dependencies.json,
reference_bundle.json, episodes.jsonl, status.json and results.json. Progress
summaries appear every ~30 seconds at episode boundaries; status also updates
at half-hour boundaries and on final/clean stop. No new model checkpoints or
frames accumulate. A control.json containing {"stop": true} requests a clean stop;
write it atomically. Interrupted-episode exact resume is not implemented.
The source receipt may say complete after a controlled exit; inspect the study's
results/status to distinguish a completed 2,000-episode run from a pause.

## Preparation and checks

tools/prepare_csail_breakout.ps1 downloads only pinned reference files and checks
their identities. It refuses an existing destination. The isolated runtime is
outside the repository; requirements/csail-breakout-replay.txt records its packages.
tools/audit_csail_breakout.py checks 32 synthetic binary vectors against an
independent scalar math.fsum/tanh reference, with tolerance 1e-4. Observed maximum
logit difference was 1.4862e-5. This read-only audit performs zero environment steps.

tests/test_csail_breakout.py includes known-answer inference, channel mapping,
malformed/hash-mutated artifacts, categorical behavior, frozen identity, terminal
versus truncated trials, partial-stop exclusion, dependency rejection and full
runner orchestration with a synthetic one-tick environment. A separate native
environment fixture executes three ticks with synthetic weights, not the released
policy. All twelve focused checks pass in the isolated CPU runtime, and pip check
reports no broken requirements. No released-policy game evaluation ran in setup.

## Prepared launch boundary

Constructor: skillforge.csail_breakout.BreakoutPolicy.
Runner: experiments/run_csail_breakout.py::run.
The candidate pin is prepared with tools/prepare_csail_replay_pin.py and remains
unregistered until the user approves its exact digest. Launch only from the repo
root through experiments/source_consistency.py, using the isolated replay Python
and setting OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 and
MPLBACKEND=Agg before starting the process.

Expected duration is approximately 10-30 minutes on CPU, with a two-hour guard;
this is an estimate, not a measured replay runtime. API spend is zero. This approval
would cover only the fixed-artifact replay and uniform control, not fresh Astra
calls, matched WSPL authoring, PPO training or expanded experiments. The proposed
$150 authoring budget remains unapproved and is not consumed by this step.
