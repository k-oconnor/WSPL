# Reworked WSPL Breakout: completed comparison

November 4, 2026. **The rebuilt interface passes the predeclared feasibility
rule. Comparative superiority is not established.** All six fresh Astra/xhigh
proposals were valid and completed 1,000 paired evaluation episodes each.

| Draw | Native mean reward (episode SE) | WSPL mean reward (episode SE) |
| --- | ---: | ---: |
| 0 | 498.366 (5.520) | 524.558 (4.782) |
| 1 | 260.406 (6.298) | 374.460 (6.186) |
| 2 | 352.777 (7.556) | 276.037 (7.015) |
| Mean of three proposals | **370.516** | **391.685** |
| Across-proposal sample SD | 119.968 | 125.153 |

WSPL's mean is 21.169 points higher, or5.71%, with substantial variation across
only three independent authoring draws per arm. Draw labels block collection
order and share evaluation seeds; they do not pair model sampling randomness.
Episode SEs describe a fixed policy, not author reliability. The 6,000 episodes
are not 6,000 independent authoring replications. No significance,
noninferiority or superiority conclusion follows from this small screen.

The prospective rule required a useful native mean>=100, all three WSPL policies
evaluable, WSPL mean>=100 and >=90% of native. All conditions hold:
`feasible_for_next_domain`. No invalid proposal penalties were needed.
The prior native screen's mean512.205 is historical context, not a paired
control: the common prompt/interface package and evaluation seeds changed.
The lower current native mean cannot be attributed to any single change here.

## What the language actually did

All three WSPL responses used `exact_to_storage` and construction loops, with
136/136/142 loop iterations and1626/609/1356 expanded statements. These build
ordinary fixed-network weights; loops do not execute during gameplay. Examples
use nested grid loops to populate spatially patterned weights and explicit
tables/constants for reflected landing estimates. The latter semantic roles
are the authors' explanations, not independently established circuit meanings.

All six policies use the original20,486-parameter600->32tanh->32tanh->6 network,
zero initialization, raw observations and the same deployed NumPy evaluator.
WSPL had full tensor access and no scalar relay allocation. No old compiler
capacity failure blocked these proposals. This supports useful, replayable
neural-weight programs through the rebuilt interface. It does not yet test
graph surgery, preservation under repair, reusable modules or causal semantics.

| Observed authoring measure | Native | WSPL |
| --- | ---: | ---: |
| Input tokens | 12,177 | 12,186 |
| Cached input tokens (included above) | 3,840 | 3,840 |
| Output tokens, including reasoning | 66,085 | 53,167 |
| Reasoning output tokens (included above) | 55,737 | 49,137 |
| Stored executable source bytes, all three | 16,185 | 7,131 |
| Original reply bytes, all three | 18,236 | 8,208 |
| Collection seconds, all three | 2,610.829 | 2,362.296 |
| Construction seconds, all three | 0.983 | 5.350 |

WSPL used19.55% fewer reported output tokens and55.94% fewer stored source bytes
in this screen. Native stored sources include the post-hoc WSPL normalization
wrapper; those sources were never shown to either writer. Original reply bytes
also differ substantially, but brevity is not a measured human-review benefit.
No reliable population-level cost advantage is established from three draws.
Construction time includes artifact preparation; all edits are interpreted
as data, not arbitrary executable proposal Python.

Both arms rejected malformed proposals atomically under the frozen protocol,
unlike the historical native skip-invalid rule. WSPL selected exact arithmetic
and native used its original FP32 update semantics; this compares interface
packages, not surface syntax alone. Tanh formal verification is unsupported.
Numerical checks and game scores do not certify IEEE behavior universally or
physical controller safety.

## Audit and retained evidence

Run and source receipt are complete. Started09:22:02UTC/01:22PST; finished
11:00:10UTC/03:00PST. Recorded study time5,885.156seconds (98.09minutes).
Six constructor calls, six fresh threads, one completed reply and zero tool calls
each. No retries, feedback, gradient updates or local GPU use. Launch stderr
is empty; auxiliary CLI stderr remains archived and is not model attestation.
Explicit model arguments and all collection events were checked.

The [read-only audit](wsp_breakout_v1_results.json), implemented in
[the auditor](../../tools/audit_wsp_breakout_v1.py), verified:

- Approved95-file source inventory, ten bound resources, runtime and CLI identities.
- All indexed artifact hashes, prompt contents/previews, collection commands,
  unique threads, reported usage and complete six-proposal accounting.
- Every reply reconstructed the exact saved candidate, plan, construction receipt
  and deployed weights. All70 equations per policy and32-input numerical parity
  reports reproduced.
- All6,000 episode records, seeds, action counts, termination/horizon accounting
  and independently recomputed summary statistics. Native1129/3000 and
  WSPL1381/3000 episodes hit the horizon; all were retained.
- All18 saved first-three-episode trajectories: observation/action/reward/terminal
  accounting and127,378 requested actions reproduced from frozen observations,
  deployed policy and recorded action RNG. This did not rerun simulator episodes.

Original raw files remain unchanged in
`Skillforge-wsp-breakout-v1/results/wsp_breakout_v1`. The
[compact archive](../../artifacts/wsp_breakout_v1/README.md) retains all six
byte-preserved replies, all six executable sources and a content-hashed index
linking the complete weights/plans/trajectory archives. No candidate was omitted.
Audit generated no model calls, training or new scored episodes.

Source pin: `e842840acc4c8229f370e37c456bf6898f88f676a13e3dde36b2b3ba5085acc4`.
[Protocol](BREAKOUT_REWORKED_V1.md). The earlier matched-ReLU interpretation pause
is unchanged; these results belong to a new independently specified screen.

## Next domain

Proceed to Kinetix learned-controller repair as requested. First reconstruct the
released rounded parent and edited actor and establish simulator/inference replay;
then prepare fresh matched native/WSPL repair. Keep fixed-architecture editing
separate from graph augmentation and measure preservation explicitly. The
[reference readiness note](KINETIX_REPLAY_READINESS.md) records downloaded inputs
and remaining integration work. No Kinetix scientific run has launched.
