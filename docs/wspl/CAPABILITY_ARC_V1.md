# ARC science before/after the 6×6 capability screen

Prepared November 9, 2026. No launch until the exact manifest is approved.

## Question and frozen conditions

Does copy training and subsequent spatial-movement adaptation preserve or change
SmolLM2-135M-Instruct's performance on ARC science questions? This is not an
ARC-AGI grid evaluation. All artifacts already exist; no fitting, author calls,
prompt selection or model selection occurs in this run.

Evaluate seven conditions: original Hugging Face model; copy-trained foundation;
and that foundation with each of the five completed capability6 adapters
(gradient, native_0, wspl_0, wspl_1, native_1). Keep every adapter, without choosing
the best grid result. Load the exact final graph/tensor artifacts and their
original attention-projection attachments. No external solver or grid formatting.

## Scoring and analysis

Use all 1,172 ARC-Challenge and 2,376 ARC-Easy test questions, 24,836 row-condition
evaluations. These test splits were already used by earlier ARC editing studies;
this is descriptive transfer/retention, not fresh held-out confirmation.

Reuse `arc_eval.score_row` unchanged: zero-shot `Question: ...\nAnswer:` prefix,
full candidate answer text, separate prefix/answer tokenization, FP32 eager
attention, TF32 off, right-padded choices, answer-token-only log likelihood.
Primary prediction maximizes mean log probability per answer token; raw summed
log probability is secondary. No chain of thought, chat template, demonstrations,
generation, or truncation. Overlength/nonfinite inputs fail closed.

Report both accuracies separately for Challenge and Easy. Report paired
corrections, regressions and net differences against both the original model and
the copy-trained foundation. Archive every choice score and prediction, not just
aggregate accuracy. Differences do not establish general reasoning gains or WSPL
superiority; these adapters were trained for a different domain, and their
topologies, training recipes and parameter counts differ.

## Execution and integrity

One GPU model is loaded; after scoring the original, restore the frozen
copy-trained weights once. Attach/detach each adapter in sequence. Verify original
and foundation tensor hashes, adapter immutability and base identity after every
condition; verify bound resources before and after. Synthetic preflight checks
all five final artifacts against their training checkpoints, choice-batch versus
singleton scoring, exact hook removal and frozen tensor identity.

No model copies or new training checkpoints. Per-row JSONL scores flush to disk;
live status records condition, split, row counts and elapsed time. Transient status
write failures are nonfatal; essential output failures abort. Four-hour wall guard
and `control.json` stop support; incomplete runs remain marked incomplete, with no
automatic retry or follow-on. Expected tens of minutes to roughly two hours,
depending on GPU contention. No paid/model-authoring calls.

Output: `Skillforge-capability-arc-v1/results/capability_arc_v1`.
The approval registry remains unchanged during preparation.
