# Supervision with WSPL-authored LoRA factors

Status: preparation in progress, November 23, 2026. No scientific run, author
request, trained-model evaluation, source registration, or launch has occurred.
The original machine and checkpoints are unavailable. Kevin explicitly requested
a fresh local rebuild. The prerequisite is specified in FRESH_ROBOT_PARENT_V1.md;
no artifact transfer is required. This author-study protocol remains a proposed
design, not a ready-to-approve manifest.

## Question

Can Astra construct useful persistent LoRA-factor edits for the Qwen robot
executive through WSPL? Does the interface help relative to native factor
construction under the same conditions?

This is the missing fourth presentation section. Its outcome is unknown. A
failure is retained and published as evidence; preparation does not presume a
positive result or general language-model capability installation.

## Fixed common boundary

Use Qwen2.5-Coder-1.5B-Instruct and a newly trained ordinary-practice adapter,
provided the separately pinned prerequisite passes its readiness gate.
Keep the rank-16, alpha-32 adapter topology, base model, tokenizer, frozen motor
library, sensor/mission semantics, and all runtime budgets identical.

Use the archived successful supervisor's **raw-observation** codec and single
unconstrained full-vocabulary argmax token per decision in every condition.
This supplies a previously engineered interface equally to both author arms;
it is not rediscovered by them and is not untouched prior knowledge. The codec
may format/alias measurements but cannot infer the regime or override decisions.
The frozen motors are newly supplied analytic WSPL-compiled nominal and inverse
calibration controllers, not the unavailable historical artifact.
No new hook, prompt, decoder, motor controller, external solver, online learner,
or inference-time access to Astra is permitted in a submitted condition.

The previous 8/32 baseline used a different policy interface. Do not reuse that
number as the new matched baseline. Evaluate the freshly frozen parent under the
common codec, on new paired missions, after all candidates freeze.

## Proposed comparisons

1. **Unchanged parent:** ordinary-practice weights, common codec.
2. **Native factor construction:** three separate Astra sessions, same parent;
   write A/B arrays directly with native numerical construction.
3. **WSPL factor construction:** three separate Astra sessions, same parent;
   submit WSPL source that reconstructs selected A/B pairs from that exact parent.
4. **Fresh gradient-trained adaptation control:** construct a new control from
   the same fresh parent and codec. Its supervision and compute budget must be
   specified and pinned before launch. No historical adapter is available;
   a successful positive control is not assumed.

The six author sessions receive the same task documentation, parent-factor
access, development observations/diagnostics and per-session resources. Alternate
interface order across the three collection blocks; each session is fresh. Do
not supply the other arm's transcripts, final controls' scores, or successful
gradient adapter weights to the authors. A draw labels collection order; it does
not match uncontrollable model sampling randomness.

Primary authoring scope is direct construction, without gradient/optimizer
training. Analytic numerical construction may use development data and must be
disclosed. Generated research code and tool use must be audited: deploying only
WSPL source does not prove that its coefficients were not fitted beforehand.
If an author trains or fits iteratively, preserve the attempt and classify it as
training-assisted; do not present it as gradient-free authorship. A general
Python sandbox is not a semantic proof of the no-training rule.

Both direct interfaces have access to the same factor entries and the same
unchanged deployed function class. No rank, base-weight, scaling, or topology
change is admitted. Every failed proposal consumes its response allowance.
Freeze the last accepted candidate in each session. A session without a valid
candidate is an authoring failure; do not silently substitute the baseline or
retry until it succeeds.

## WSPL-to-LoRA bridge

`skillforge.wspl_lora` treats each projection's A/B pair as an opaque FP32 tensor
bank. The whole adapter hash binds the pair identities, shapes and exact bits.
The source target pins the pair; construction uses fixed `exact_to_storage`
semantics, bounded arithmetic and loops. There are no external tensor bindings.
Source, parent/candidate hashes, changed elements and construction counters are
archived. Imported plans replay through the existing editor.

Sharding avoids serializing all 18,464,768 factor coefficients into one enormous
JSON edit plan. Unedited pairs retain their exact parent bits. Native candidates
pass the same shape, dtype, rank and finiteness checks. Installation touches only
the explicitly named factors; base weights and alpha/rank scaling remain frozen.

A source receipt is a construction identity, not a semantic skill certificate.
Editing factors with zero effective change must not be called acquisition.
Read-only audits must report both factor differences and effective update
differences, and verify that the final deployed adapter matches its source.

## Development and evaluation

Use the previous bounded reaching domain and four conditions: ordinary,
changed calibration from the start, mid-mission switch, and return to original.
Expose the development simulator and diagnostics only in the isolated author
sandbox. Keep the independent final evaluator and other sessions inaccessible.
The original Linux/bubblewrap isolation must be preserved or an equivalent
boundary tested; the current Windows host has no WSL installation.

Proposed final panel: 64 new missions, 16 per condition, three to five ordered
waypoints. Generate them in the sealed evaluator only after **all** six candidates
and both controls freeze. Eight conditions yield 512 mission executions.
Archive every trajectory, Qwen decision, output token, probe and failure; paired
initial-state hashes must match across conditions. The old 32 final missions are
historical evidence and are excluded from fresh confirmation.

Primary measures: mission success, per-condition corrections/regressions versus
the common parent, ordered delivery/waypoint completion, invalid output, probes,
Qwen calls, physical ticks and deployment time. Report per-proposal outcomes and
between-author variation; episodes are not independent author replications.

Proposed per-candidate exploratory capability gate: at least 14/16 ordinary
successes, no more than one ordinary regression against the common parent, at
least eight net adaptation corrections across the other 48 missions, and no
condition-level net loss. If the parent scores below 12/16 ordinary, mark the
baseline-readiness gate failed and report the full panel without claiming
retention. These thresholds are provisional until the protocol is pinned.

Compare native versus WSPL descriptively across three proposals per arm. Report
validity, all calls and failures, input/output/reasoning tokens, construction
time, diagnostic/runtime usage and source size separately. No general cost or
population-level superiority claim follows from a small screen. The fresh
gradient control's supervision and compute must be reported separately; it
cannot establish efficiency unless the comparison is explicitly matched.

## Proposed resource envelope

At most 16 Astra/xhigh responses per session, six sessions, 96 responses total;
20 minutes per response. Proposed 24-hour development wall guard, eight aggregate
GPU job-hours, and a separate six-hour final guard. This is a maximum envelope,
not a runtime estimate. Actual budgets must be reviewed after checking the
new local runtime and a synthetic full-size preflight. Existing Codex
usage; no hard dollar/output-token cap or model fallback is implied.

No automatic retries after transport/infrastructure failure. No post-final-score
repair or follow-on. Keep compact accepted source revisions and one rolling
resumable checkpoint per active job plus final/clean-stop, without copying base
models into every candidate. No optimizer training is part of the direct arms.

## Current engineering evidence and remaining work

Fourteen synthetic bridge checks pass on Python 3.13 / NumPy 2.4.4 / Torch
2.12.1+cu126. They cover native/WSPL factor identity, exact no-op preservation,
stale parents, rank/shape/dtype/nonfinite rejection, failure atomicity, resource
limits, synthetic PyTorch forward parity, frozen base preservation and exact
restoration. No real Qwen forward or task prediction was generated.

Still required before a launch approval can be requested:

- Complete the separately pinned fresh ordinary-parent prerequisite and freeze
  its artifact if ready; specify a fresh adaptation control. The native Windows
  simulator is operational, but author isolation is still required.
- Complete the fixed-policy deployment loader, per-arm author tool/runner and
  final multi-arm evaluator integration; the previous broad supervisor runner
  cannot be reused unchanged for this narrower deployment contract.
- Validate actual adapter schema/dtype, full-size source reconstruction, finite
  synthetic forward parity and frozen-base identity without task scoring.
- Test matched development access, sandbox denial of final data, budget cleanup,
  rejection accounting and the all-candidates-frozen evaluation boundary.
- Produce concrete runtime configuration and complete source/resource inventory;
  run offline prompt preflight; calculate the candidate manifest SHA-256.
- Obtain user approval of that exact digest, register it separately, then launch
  only through `experiments/source_consistency.py`.

Historical approvals and source digests do not authorize this study.
