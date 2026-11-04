# Kinetix car-repair replay: completed

Completed November 4, 2026, 15:51 UTC / 07:51 PST. Source receipt and study both
complete. Approved manifest:
`ef3ca419ffae13e156be92a3eb2d220762c035cc2eb00839e4d8c3c8d0b0046a`.
CPU execution took 1,715.60 active seconds (28.59 minutes). No training, model
calls or GPU use.

| Fixed policy | Success | Timeout | Red / invalid | Mean return | Mean steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rounded parent | 0 / 1,024 | 1,024 | 0 / 0 | 0.151870 | 256 |
| Released repair | 1,024 / 1,024 | 0 | 0 / 0 | 1.373846 | 73 |

This reproduces the qualitative released result: the parent stalls until timeout,
while the edited controller succeeds. The upstream report used 256 parent and
1,024 repaired episodes; ours used 1,024 each with paired action RNG seeds.
The original evaluator and precision/RNG procedure are unavailable. This is a
statistical replication under the explicitly documented local implementation,
not bitwise reproduction of their episodes.

The read-only [audit](kinetix_replay_v1_results.json) verified all 99 source files
and 11,287 bound resources, source receipt, actor archives, numerical reconstruction
of the 51 published edits, all 2,048 episode records and independently recomputed
summaries. It replayed all 987 sampled actions from the six saved trajectories
using their recorded observations and RNG seeds, and checked recorded return
sums. It did not rerun physics. Auditor: `tools/audit_kinetix_replay_v1.py`.

The released actor stores float64 weights; the local actor uses float64 arithmetic
and simulator observations/physics use FP32. Reconstruction is numerically exact,
with 3,145 signed-zero differences attributable to integer text losing negative
zero. The optional JAX profiler import warning did not prevent physics execution;
no profiler measurements are claimed.

Raw evidence remains in the unchanged isolated checkout:
`C:/Users/kevin/OneDrive/Desktop/Skillforge-kinetix-replay-v1/results/kinetix_replay_v1`.
Both actors, every episode and the first three trajectories per policy are retained.

## Interpretation and next step

This establishes a functioning local car-repair benchmark and reproduces a supplied
repair. It is one fixed level and start state, with stochastic action sampling;
1,024 successes do not establish robustness across scenes, disturbances or novel
tasks. No fresh authorship, WSPL advantage, formal tanh controller proof or robot
transfer is demonstrated.

Next prepare a fresh native-versus-WSPL repair comparison with equal task access,
numeric deployment, tensor access, topology and proposal budgets. Withhold the
published answer and replay scores from proposers. Resolve the 577 KB prompt's
file/stdin preflight and FP32-versus-float64 interface alignment before collecting
any proposal. Fixed topology and graph augmentation should be separate studies.
No follow-on run has been approved or launched.
