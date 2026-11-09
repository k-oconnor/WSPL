# Bounded relational neural-program test

One Astra/xhigh session, at most six responses, native Python authoring, no gradient
training of the candidate. This is the agreed final bounded conceptual-ASP test
under the current fixed-slot interface, before recommending any expanded language
or harness. It tests joint delivery and transformation, not a general ARC claim.

## One artifact and execution hierarchy

Save a compact program.npz containing K, V, alpha and W27, plus program.json with
fixed attachment points, tensor/source/base hashes and execution semantics. A
NeuralProgram applies both within ordinary model execution. Layer26/group0 virtual
attention delivery precedes the layer27 attention-output weight edit. No author
code, labels, physical rule engine, task-ID routing or answer postprocessor is in
the candidate's forward pass. The symbolic label generator is evaluation-only.
The original runtime hook implements the fixed primitive; this is not yet a
native architecture rewrite, an ASP language, distillation or weight-only merging.

Keep the same permitted scope as the joint pilot: two64-number vectors and one
576x576 weight matrix,5% Frobenius bound, all other weights unchanged. Start from
the original conduction slot and original model weights; the author can replace
K/V. Acquisition diagnostics are the same prefix residual/projection inputs plus
decoded answer-token strings, native acquisition queries and attention log
normalizers. No gradients or runtime model access in sandbox.

## Data and counterfactuals

Acquisition:24 designer cases (eight physical states, three consequences each)
plus eight previously correct ARC retention cases. Reserved:48 designer cases
(sixteen states, three consequences each) plus twelve other retention cases.
Cases ask which object warms, cools or changes temperature. Four answer phrases
are fixed across those tasks. Perfect insulation, conducting contact and equal
temperature have explicit idealized semantics. Objects have no other heat source.

Adjacent states form physical-intervention pairs: reverse temperature ordering,
remove contact, or change equal temperatures to unequal. Acquisition has12 paired
task comparisons; reserved has24. Some pairs keep the correct answer unchanged
for a task (e.g. both objects change after swapping hot/cold roles); report the
number of answer-changing pairs separately. Reserved wording reverses the order
of object descriptions and uses different numeric combinations. This is a small
correlated synthetic panel, not independent physical experiments or natural ARC.
Task forms repeat. No claim of unseen task types or pretraining decontamination.

Retention cases are selected by a fixed hash ordering from previously used ARC
validation predictions, excluding heat-related keywords. Acquisition/reserved
retention IDs are disjoint. Reuse of prior validation is explicit; no ARC test
split is opened. Candidate correctness is not used to select reserved examples.

## Controls and fixed decision rule

Freeze last accepted author artifact and answer-bias control before opening
reserved examples. Evaluate unchanged, combined, final WSP-only, final ASP-only
and answer-bias reference. Component removal is an ablation, not retraining.

The bias control uses acquisition baseline scores only. Enumerate625 combinations
of four offsets from {-2,-1,0,1,2}, center each vector, and choose minimum mean
multiple-choice cross-entropy (ties: smaller norm, then lexicographic). Apply
these constant offsets to matching answer strings; other strings receive zero.
It receives no question features or reserved labels. This comparator is an
explicit score postprocessor, not part of the candidate. Raw baseline likelihood
sums are retained alongside calibrated normalized scores. It is not a matched
author-compute or editing-capacity control.

Recommend pursuing conceptual ASP under this interface only if ALL descriptive
criteria hold on reserved data:
- At least six more family answers correct than both unchanged and bias control.
- Improvement over both references on at least two of the three task types.
- At least three more fully correct physical-intervention pairs than both references.
- All originally correct retention answers preserved.
- Combined family accuracy strictly exceeds both final component-only ablations.
- At least two baseline-failed family cases corrected only by the combination,
  not either final component alone.

Report every count and failure even if a criterion is unmet. Thresholds are
engineering triage, not statistical significance or proof of a causal mechanism.
If unmet, park the conceptual claim under this interface rather than expanding
the harness or retrying until success. Any new direction requires discussion.
No automatic next run is authorized in either outcome.

## Budgets and validation

Six calls maximum including inspections/rejections/FINAL;20-minute call guard,
four-hour study guard; expected30–90minutes. Existing Codex account usage, no hard
dollar/output-token cap or API-key fallback. CPU sandbox construction and FP32
GPU inference. All prompts, programs, accepted small banks, diagnostics, bias
selection, compact artifact and per-case scores saved. No whole-model snapshots.
Static tensors restored exactly after each evaluation. Compact/full artifact
equivalence, explicit hierarchy, reserved-access barrier, pair labels and bias
selection have synthetic/unit checks. Source/runtime/resources pinned. Exact
source-manifest approval required before scientific scoring or author calls.
