# Direct versus structured neural repair: first feasibility screen

Prepared November 4, 2026. Engineering stress checks completed; no scientific
proposal collection, source registration or experiment launch has occurred.
The exact source digest must be approved before the new screen runs.

## Question and scope

Does access to structured WSPL surgery help Astra repair an existing neural graph
while preserving an independently specified behavior, relative to the same model
with competent raw tensor and graph access?

This first screen uses **three supplied synthetic stateless neural circuits**.
Their weights are explicit engineering constructions, not trained controllers.
The targets are fully specified and deliberately small. This is an interface and
repair feasibility test, not evidence of robot control, learned-policy repair,
novel algorithm discovery, human engineering savings or recursive improvement.
A ceiling result should prompt harder tasks rather than a superiority claim.

## Frozen design

Two arms, two fresh draws per task, three tasks: **12 invocations** total.
The proposer is `gpt-6-astra`, reasoning effort `xhigh`, through the unchanged
ChatGPT-backed Codex collector. Each invocation gets one response, no tools,
evaluation feedback, harness retry or previous answer. A successful proposal is
not replaced by a search for a better one. Collection errors stop the queue and
retain unattempted denominators; invalid proposals count as failures and the
remaining scheduled trials continue.

| Task | Requested change | Preserved behavior on [-1,1]^2 |
| --- | --- | --- |
| shared_gain | drive = 3*x0; parent branches share gain two | keep = 2*x1 |
| clipped_drive | drive = clip(2*x0,-1/2,1/2) | keep = x1 |
| isolated_deadband | drive = ReLU(x0-1/4)-ReLU(-x0-1/4); remove cross-channel dependence | keep = ReLU(x1) |

Both arms receive identical task/graph/parameter data, external claim text and the
entire [shared interface packet](REPAIR_INTERFACE_V1.md). Only the selected-arm
suffix changes. Node IDs are neutral; the task may explicitly mention observable
sharing. No corrective tensor/module answer is supplied. The preparer's known
answers are test fixtures, excluded from the collector prompt and empty workspace.
Task IDs, prompt bytes and source are frozen before collection.

The direct arm can add/replace/remove raw nodes/tensors, change public output
connections, and use compact tensor assignments/slices/arithmetic. The WSPL arm
has the same raw path plus source for regions, replacement, transplant, residual
and untie operations. Modules must be authored in the reply. The transport hashes
submitted modules and expands explicit hash placeholders before replay; it does
not invent a module or infer a repair. WSPL may use only raw edits, and actual
structured-feature usage will be reported. No source-expression allocator is
required in either arm.

Both arms use the same finite FP32 storage and direct exact-to-FP32 rounding,
operators, evaluator and public interface. Identical caps apply: 24 nodes,
48 edges, 128 stored parameter elements, depth 12 and 128 cumulative traced
activation values. Operators: affine, ReLU, identity, add, scale, concat, slice.
Disconnected nodes and unused tensors still count. Full internal replacement is
allowed in both arms. No hidden requirement to use a particular repair strategy.
The raw phase is atomic; intermediate structured graph steps must each be valid.
That workflow distinction is part of the tested interface and is disclosed.

Serial order alternates which arm comes first across tasks and draws. There is
no provider sampling-seed matching; pairing means shared task and declared draw
index, not coupled randomness. Fresh thread IDs are checked. Existing prefix
caching is recorded and does not imply a shared conversation.

## Outcomes and interpretation

Primary: valid candidate **and** both external real-graph claims PROVED **and**
finite FP32 execution with maximum absolute error <=5e-6 on a fixed 249-input
panel. The panel includes exact breakpoints, neighboring dyadic inputs and 128
seeded dyadic pairs. The exact panel is evaluator-only; the entire required
function and input box are public. No adaptive points or thresholds after results.

Formal verification uses stored FP32 coefficients interpreted as rational reals,
five seconds per solver query, non-vacuity and independent counterexample replay.
UNKNOWN/UNSUPPORTED/INVALID never count as success. The numeric panel is an
execution diagnostic, not a universal IEEE or physical proof. Formal and numerical
outcomes are saved separately so proof failure and runtime discrepancy stay visible.

Record all twelve trial outcomes, six paired WSPL-minus-direct success differences,
success by task, valid-only and all-attempt counts, reported input/cached/output
usage, authoring/construction/evaluation times, candidate size, tensor/node/edge
changes, structured operations, raw replies and final artifacts. Reasoning usage
is retained if the provider reports it; it is not estimated. Human setup/review
time and hosted compute/dollars are unmeasured. Token usage is not human-time savings.

Prospective descriptive next-step rule:

- Incomplete collection: unresolved; never interpret unattempted draws as data.
- Direct fewer than 4/6 successes: baseline feasibility unresolved.
- Otherwise WSPL must achieve at least 5/6 and no fewer successes than direct to
  justify a larger repair study using this configuration.
- Both 6/6: feasible with a ceiling; no advantage established. Examine costs and
  prepare harder tasks before drawing comparative conclusions.

These are internal pilot thresholds, not statistical noninferiority or product
validation. Two draws on each of three task families cannot support a broad claim.
Frozen equalities measure controlled program repair; they do not evaluate an
unknown learned controller or transfer to real dynamics.

## Runtime, limits and records

CPU construction/evaluation only; no training, GPU use, environment episodes or
paid API-key fallback. Twelve new Codex invocations consume existing account usage.
Per invocation wall guard 1,200 seconds; overall guard 14,400 seconds. Expected
duration approximately 1–3 hours, uncertain until actual collection. There is no
enforceable dollar/output-token cap; the 200 kB response limit applies after
collection. CLI transport retries may occur. These terms require explicit approval
of this new study identity; earlier authoring approvals do not cover it.

The collector's offline preflight rejects extra context/tools except the existing
advertised question tool; any actual tool/error event invalidates collection and
stops the queue. This is a restricted CLI process, not a literally tool-free API.
One final candidate and one construction receipt per valid trial; no rolling
training checkpoints. Source receipt and bound resource/dependency hashes accompany
status/results, replies, exact candidates and evaluator evidence. All failures and
partial usage records are preserved. No automatic follow-on or resume.

Constructor `skillforge.wsp_repair.RepairStudy`; runner
`experiments/run_wsp_repair.py::run`. The checked runner verifies prompt/task,
CLI, solver/numeric dependency and auxiliary identities before and after execution.
The isolated checkout will be used for approved collection so other development
can continue independently.

## Engineering preparation

The [stress report](WSP_STRESS_V1.md) covers 100 generated graphs / 1,200 sequential
edit steps and conversion/edit/replay of all three archived 20,486-parameter
controllers. Source-gate checks and fake collection use known answers only. No
new Astra proposal or scored game episode has been requested during preparation.

The fake twelve-trial queue verifies success accounting, failure preservation,
stop behavior and the all-success ceiling decision. Naive shared-weight repair
is deliberately checked to improve the target while failing preservation. All
three unchanged parents must fail their repair claim and pass preservation.

Validation records: [unit suite](wsp_repair_preparation_validation.json),
[test output](wsp_repair_preparation_tests.txt) and
[six offline prompt preflights](wsp_repair_offline_preflight.json).
