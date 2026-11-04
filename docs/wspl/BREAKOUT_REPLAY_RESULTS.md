# Completed Breakout artifact replay

November 3, 2026. Both empirical evaluation arms and the approved-source gate
completed successfully. The released CSAIL Breakout policy produced a mean return
close to the upstream report under our explicit new seeds and NumPy runtime.

| Policy | Episodes | Mean return | Standard error | Mean episode length |
| --- | ---: | ---: | ---: | ---: |
| CSAIL released policy, our replay | 1,000 | 511.828 | 7.605 | 7,271.530 |
| Uniform zero-weight control | 1,000 | 0.551 | 0.0248 | 11.634 |
| Upstream released-policy report | 1,000 | 517.411 | 7.417 | 7,417.202 |

Our mean is 5.583 lower, or 1.08% below the reported mean. The difference is
0.526 combined standard errors, a descriptive agreement diagnostic rather than
a formal equivalence/noninferiority test. The original evaluator/RNG procedure
was not supplied; these are newly sampled episodes, not bitwise reproduced ones.
The upstream mean and settings come from the pinned Experiment 5 release; see
[protocol and provenance](BREAKOUT_REPLAY_V1.md).

The released policy reached the 10,000-step horizon on 497 episodes. Those
episodes' earned returns remain in the summary. Every episode in both arms was
retained. Evaluation took 183.969 active seconds (about 3.1 minutes), excluding
the earlier development/preparation work. CPU only; zero API calls, optimizer
updates or GPU use. No new weights/checkpoints were trained or accumulated.

## Completion audit

The independent read-only audit passed: exactly 2,000 completed records; unique
consecutive episode indices per arm; prescribed seed schedules; action counts
equal step counts; terminal/truncation consistency; recomputed means, sample
standard errors, lengths and extrema; frozen reference/auxiliary identities;
all 59 pinned Python source files; two checked constructor calls; empty stderr.
Study results and source receipt both report complete, and both processes exited.

Approved manifest SHA-256:
`46379f57b9ebd29f36390be998f89352249246f917263afecb2c4b5e2d2d963c`.
Execution checkout was Skillforge-breakout-replay-v1 at commit 8488733. Raw records,
source receipt, dependency inventory and result audit remain in its ignored
results/csail_breakout_replay_v1 directory. The compact
[results and audit](breakout_replay_results.json) bind the raw-file hashes.

## What this establishes

This provides a practical replication check on loading the released tensors,
converting the observations, sampling actions and scoring the environment.
It does not test a fresh Astra proposal, compare WSPL with native editing or
ordinary code, establish formal properties of the tanh policy, reproduce PPO
training, or satisfy the product win conditions.

The subsequent, separately approved [fresh Astra native authoring screen](ASTRA_BREAKOUT_NATIVE_RESULTS.md)
has now completed three draws with no evaluation feedback. Its result and usage
records are separate from this fixed-artifact replay. A shared-ReLU WSPL
comparison remains a future extension requiring its own approval.
