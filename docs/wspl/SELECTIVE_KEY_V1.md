# Key-only selectivity follow-up

Prepared November5,2026; unapproved/unlaunched until an exact source pin is approved.
One six-response native Astra/xhigh trajectory, existing Codex usage. No WSPL arm,
no gradient updates. Closed-form supervised key calibration is permitted and must
be described as such; no zero-shot/no-learning claim. Original successful value,
layer26/group0 and base model remain frozen. Only its64-dimensional key changes.

## Acquisition and final cases

All13 original pilot questions become disclosed acquisition data (target plus12
former panel questions; reversed copies excluded). Therefore none remain held-out
evidence here. Supply original query vectors and native log partitions for the
three affected heads per case, plus applicability labels and baseline/current-key
feedback. This addresses the previous absence of measured negative query examples.
Baseline answers only three of eight negative examples correctly (two evaporation
and one lexical); retention is reported with this explicit denominator.

New manually authored final panel: eight physical settings, each with four questions
about conduction/convection/radiation/evaporation,32questions total. Each quartet
shares vocabulary but asks about a different mechanism. Same four single-token
answer words and fixed question-prefix scoring; no additional ARC test examples.
The panel is designer-visible, frozen before calls, unseen by the author, and only
opened for scoring after candidate freeze. It is not an independent standardized
benchmark or a statistical replication across authors. Cases within each setting
are correlated; report setting-level results as well as individual questions.

## Comparisons and decision criteria

Evaluate untouched, fixed forced delivery, original broad key, newly authored key,
and zero key, all with the same frozen value. Report every prediction/logprob,
attention mass, positive/negative mean mass, routing AUROC, per-mechanism accuracy,
corrected/broken cases, and retention on initially correct negative cases. Runtime
masking is solely final-prefix position; no text/ID/applicability gate is installed.
Labels are available during key construction only, not inference.

Predeclared descriptive selectivity: positive mean mass>=.5,negative mean<=.1,
AUROC>=.9. Useful behavior additionally requires improved positive accuracy and
more corrected than broken answers. Retention needs no losses among initially
correct negatives, with at least4such final cases; otherwise retention is marked
unassessable. Always report the denominator and its mechanism coverage, including
zero-coverage mechanisms. No tuning or extra author calls after final evaluation.
Report these separately; no single improved accuracy number establishes selection.

If it fails, we have not distinguished author failure from the capacity of one
linear key. A separately approved fitted-key diagnostic may follow; none launches
automatically here. No broader skill/causal-world-model claim follows from success.

## Execution and evidence

Reuse validated native sandbox, frozen FP32CUDA model and attention hook. Reproduce
all13 original authored-key prediction/mass records exactly before calls; disabled
intervention must reproduce baseline. New read-only train_Q/logZ tensors are
validated against synthetic attention in engineering tests. Parent artifacts,
model/runtime and source manifest are bound. Save all prompts/replies/programs,
accepted banks/feedback,usage and final panel records. Final uses last accepted
state, no best-of-six selection. Only static tensors execute at inference.

Expected15-45minutes;20-minute per-call/four-hour overall guard. At most6Astra/xhigh
calls, no automatic retry, no enforceable dollar/output-token cap or API-key
fallback. CPUconstruction, localGPU inference; no optimizer state or checkpoint
proliferation. Atomic essential writes plus nonfatal/retried status telemetry.
