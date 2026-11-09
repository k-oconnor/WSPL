# ARC three-arm authoring screen

Preparation only. No author calls or test evaluation before exact source approval.

Nine fresh Astra/xhigh sessions: native Python/NumPy, WSPL direct edits, WSPL
LoRA factor edits, three draws each. Counterbalanced arm order by draw:
native/WSPL/LoRA, WSPL/LoRA/native, LoRA/native/WSPL. Every session starts from
the same frozen 134,515,008-parameter SmolLM2-135M-Instruct checkpoint. Four
responses per session (maximum36 calls), including any inspection or FINAL.
Full visible history is serialized into fresh one-shot calls; no hidden reasoning
continuation. No gradient updates, automatic retries, validation-best selection,
or programmatic search. Invalid proposals consume budget and retain prior state.

Direct arms edit exactly four576x576 o_proj matrices, layers26–29 (1,327,104
coefficients). LoRA rank8/scale1 on those same matrices:36,864 factor coefficients,
both factors zero initially, read-only original matrices. Common10% per-matrix
effective FP32 delta Frobenius bound; numeric validation uses float64 norms.
This is an interface comparison for native versus direct WSPL; the LoRA comparison
also changes parameterization/capacity. It is not compute-matched. No native-LoRA
arm in this initial screen.

All arms receive identical32 train examples (IDs sorted by SHA256 of ID), baseline
Challenge validation statistics and the same first16 detailed validation records.
Each accepted edit gets all299 validation aggregate scores and those same16
details. Inspector permits512 tensor scalars per response,16 rectangles maximum.
Native stdout is archived but not returned; bounded construction errors remain
available. Programs access tensor banks only, not evaluator, data files or model
execution. No activations or gradients supplied. Return all calls, failures,
inspections, code lengths, token categories, construction and inference timing.

Baseline source pin ac191cb02d23a0916be9730e6eb0dfdd2d01d02e0c2b549cc1e0a4ba1a4bfa4c
completed with97/299 normalized Challenge accuracy and285/570 Easy validation
accuracy. Before authoring, reproduce baseline per-example records exactly and
the full model tensor hash; stop on mismatch. Existing baseline figures are
development measurements, not hidden test scores.

Freeze all nine final candidates before loading/scoring test examples. Then
evaluate untouched control and all nine on full ARC-Challenge test1172 and
ARC-Easy test2376:35,480 example evaluations total. Primary token-normalized
accuracy and secondary summed-likelihood accuracy follow the fixed baseline
scorer. ARC-Easy is related-task retention, not a general-language guarantee.
No prompt tuning/test feedback. Public pretraining contamination is unknown.
Model/dataset revisions and all files are hash-bound. GPUFP32/eager, TF32 off,
same tokenizer, no generation or truncation. Every proposal remains counted.

Report nine candidate outcomes, per-draw differences, target/retention changes
versus untouched, and observed spread over three author replicates. No claim
of broad superiority from this small screen. Preserve program/factor/effective
weight identity and exact histories. Restore base and hash all model tensors
between sessions and at completion to detect unintended modifications.

Call guard20minutes; overall12hours. Expected2–6hours, primarily authoring;
maximum call guards alone can consume12hours, so the overall guard may leave a
max-duration run incomplete. No automatic resume. Existing Codex account usage,
no enforceable dollar/output-token cap or API-key fallback. Local model holds
roughly1GiB GPU memory during authoring; final inference time is separately recorded.
One source-gated model constructor. Intermediate current+final banks and per-edit
archives are small tensor banks, not repeated full-model checkpoints.

See ARC_AUTHORING_INTERFACE_V1.md for the full proposer contract. Existing
WSPL syntax is unchanged; this adds adapter execution/validation and matched
inspection, not whole-network formal guarantees.
