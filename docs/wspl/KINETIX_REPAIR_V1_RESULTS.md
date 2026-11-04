# Fresh Kinetix repairs: both interfaces reach the success ceiling

Completed November 4, 2026 at 18:16:31 UTC / 10:16 PST, in 6,801.16 active seconds
(113.35 minutes). Source receipt and study complete, seven constructors. Approved
digest `0cd0a7a6f7e7e0b6c5a8e2348788dc127f3d435a187c048c35bc56ed760f23e3`.
No optimizer updates or local GPU; six fresh Astra/xhigh proposals, no tools,
feedback or retries. All proposals valid.

| Controller | Success | Mean episode length | Changed parameters | Output tokens, including reasoning |
| --- | ---: | ---: | ---: | ---: |
| Untouched FP32 parent | 0/1,024 | 256 | 0 | — |
| Native draw0 | 1,024/1,024 | 74 | 3,820 | 8,701 |
| Native draw1 | 1,024/1,024 | 78 | 3,557 | 10,466 |
| Native draw2 | 1,024/1,024 | 79.999 | 19,151 | 19,148 |
| WSPL draw0 | 1,024/1,024 | 79 | 4,842 | 10,124 |
| WSPL draw1 | 1,024/1,024 | 79 | 4,333 | 9,087 |
| WSPL draw2 | 1,024/1,024 | 77 | 3,187 | 10,495 |

Both arm means are 100% success, so the predeclared descriptive feasibility rule
passes. The untouched parent times out in every episode. These are six independent
author draws on one supplied fixed scene/start, with stochastic actions; 6,144
successful repair episodes are not 6,144 independent authors or evidence of
general robustness. Neither interface demonstrates superior task performance.

## Authoring efficiency and interface use

| Aggregate over three proposals | Native | WSPL |
| --- | ---: | ---: |
| Input tokens | 1,059,450 | 1,059,459 |
| Output tokens, including reasoning | 38,315 | 29,706 |
| Reasoning output tokens | 36,486 | 28,124 |
| Non-reasoning output tokens | 1,829 | 1,582 |
| Collection seconds | 1,760.23 | 1,343.11 |
| Normalized stored program bytes | 2,649 | 2,203 |

WSPL used 22.5% fewer aggregate output tokens and 23.7% less collection time in
this screen. The native third draw drives much of that difference: the median
output-token reduction is only 3.3%, and the first two draw pairs combined have
almost identical output totals (19,167 native versus 19,211 WSPL). Input dominates:
combined input+output tokens differ by only 0.78%. No cached input tokens were
reported. These small-sample observations do not establish a reliable cost or
latency advantage; dollar cost and hosted compute are unmeasured.

All three WSPL programs used `native_numpy_v1`, zero loops, and direct literal
tensor edits. None exercised exact-arithmetic loops, circuit placement or graph
surgery. Program byte counts use normalized archived construction source; native
sources include a post-hoc WSPL wrapper and alias mapping, so they are not a pure
surface-syntax comparison. Both arms reached the same FP32 function class.

### Construction work and why richer features were unused

| Across three proposals | Native | WSPL |
| --- | ---: | ---: |
| Edit statements | 112 | 97 |
| Charged construction operations | 2,487,951 | 1,949,335 |
| Median charged operations per proposal | 637,177 | 623,376 |

The WSPL total is 21.6% lower for charged operations and 13.4% lower for
statements. The median operation difference is only 2.2%; the large third native
repair again explains much of the aggregate difference. These are bounded editor
accounting counters, including charges for processing entire destination tensors
and selected elements. They are not hardware instructions, measured FLOPs or
inference compute. Zeroing weights does not make the deployed dense NumPy network
execute sparse kernels. No inference-efficiency advantage is established.

Five of six fresh proposals used 24–36 statements, versus 51 in the published
repair; the remaining native proposal used 57. This supports the observation that
compact alternative repairs exist. The published repair used float64 in our replay
and a different evaluation panel, so this is a descriptive historical comparison.

The absence of rich constructs has two distinct explanations. First, this
protocol explicitly prohibited circuits, regions, placement, descriptions and
graph edits to hold topology fixed. Their value was therefore not tested.
Second, constants, exact arithmetic and bounded loops were allowed, but all three
WSPL authors chose the native-compatible literal profile. The short threshold/
control paths could be expressed directly without loops. The guide also presented
the compatibility profile first, which may have affected selection; that hypothesis
has not been tested. Unused features should not be described as ineffective.

To test structured surgery fairly, give the native comparator equivalent raw graph
and tensor access, equal topology budgets and the same preservation obligations.
Compare repair success, preservation violations and authoring effort. Changing
topology only for WSPL would confound language assistance with model capacity.

Authors described momentum-building run-up controllers using position/velocity
thresholds; one native proposal additionally described pitch guards. Those are
author interpretations, not independently verified circuit semantics. The tests
establish sampled task behavior, not formal tanh/IEEE controller proofs.

## Audit and archive

[Windows audit](wsp_kinetix_v1_audit_windows.json) verifies 104 source files,
518 bound Windows files, all six fresh collection threads and event/usage logs,
prompt/context/transport identities, exact reply-to-plan/artifact/FP32-weight
reconstruction, parent identity and independently recomputed episode summaries.
[Linux audit](wsp_kinetix_v1_audit_linux.json) verifies 11,286 bound resources and
replays 2,169 actions from all 21 saved trajectories using their observations and RNG.
All 7,168 episode records are accounted for. No new physics was simulated by audits.
Auditor: `tools/audit_wsp_kinetix_v1.py`.

Compact [replies and source programs](../../artifacts/wsp_kinetix_v1/README.md)
are preserved byte-for-byte. Full prompts, events, plans, candidate artifacts,
weights, all episode records and trajectories remain in the unchanged isolated
`Skillforge-wsp-kinetix-v1/results/wsp_kinetix_v1` directory.

The large-prompt preflight used a digest sentinel to verify context configuration;
the full prompt was delivered through stdin with byte/hash accounting. This is
not an independent server-side receipt proof. The original parent trace is supplied
historical information. Published repairs and local replay outcomes were withheld
from the proposer packets; prior exposure in model training cannot be excluded.

## Next useful test

Further fixed-scene success sampling cannot resolve this ceiling. A stronger
comparison would require repairs with preservation obligations or bounded layout/
dynamics variation, and separately test structured surgery against equally
expressive raw graph edits. Freeze that next protocol before authoring. No new
scientific run is approved or queued.
