# Graph surgery stress and controller integration checks

Completed November 3, 2026. [Machine-readable results](wsp_stress_v1_results.json).
Reproducer: `tools/stress_wsp_graph.py --output <new-directory> --seeds 100`.
Python 3.11.9, NumPy 1.26.4, CPU PyTorch 2.5.1. No hosted model calls, training,
new game episodes, GPU use or scientific performance evaluation.

## Generated graphs and sequential edits

100 seeded DAGs (41000–41099), each with width 2–5, shared affine parameters,
branches/skips and randomly chosen affine/ReLU/add/identity/scale/concat/slice
nodes. Node declaration order is shuffled. Each graph follows twelve edit steps,
covering untie, direct tensor edits, equivalent region replacement, residual
insertion, transplantation and rejection/rollback. The operation schedule is
fixed across seeds; the graph, parameters and probe inputs vary.

All 1,200 steps passed, including 1,000 changed states and 400 explicitly rejected
operations (late invalid writes and cyclic redirects). Four dyadic input vectors
per graph are checked at initialization and after each step. A separately written
recursive evaluator independently decodes stored bits and executes exact rational
and PyTorch FP32 semantics without using the production topological traversal.
All rational outputs agreed exactly; maximum PyTorch/NumPy difference was zero
on this bounded dyadic panel. These values are deliberately numerically simple;
this result does not bound arbitrary long-chain floating-point accumulation.

Operation-specific checks establish expected residual addition, equivalent
replacement/transplant/untie behavior, unchanged parent identities, and unchanged
outputs at nodes classified as having unaffected dependencies. Raw invalid
transactions publish no candidate. All artifact IDs are retained in the local
raw record `work/wsp_graph_stress_v1/results.json`; its hash is in the summary.

## Archived controller scale

Each of the three existing Astra native Breakout replies was reimported into the
20,486-parameter dense tanh artifact and explicitly converted to a DAG. The final
identity node was replaced through the WSPL source frontend, then planned,
replayed, serialized and reloaded. This deliberately semantics-preserving edit
tests the integration path; it is not a new controller repair result.

All tensor bits remained identical. NumPy dense and edited-DAG logits were
bitwise identical on 32 inputs per controller: zero, one, seeded Boolean vectors,
and continuous vectors around the usual observation range. These synthetic
vectors are not claimed to be physically reachable game observations.

An independent PyTorch execution agreed within the declared absolute/relative
3e-5 tolerances; maximum absolute difference across all three was
7.62939453125e-6. Original archived reply hashes remained unchanged. Each complete
controller integration check took under 0.71 seconds. No tanh formal proof was
attempted. The entire combined stress pass took 17.86 seconds on this host.

## Decision

These checks support proceeding to a bounded authoring/repair pilot within the
implemented operator and size limits. They do not establish solver soundness,
IEEE proofs, arbitrary graph import, untrusted service hardening or scientific
repair efficacy. No source repair was required by these checks. The proposed
[repair screen](REPAIR_SCREEN_V1.md) has separate external success criteria and
requires its own exact-source approval before model collection.
