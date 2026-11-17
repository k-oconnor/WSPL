# Unified WSPL executive repeat

Prepared November 17, 2026. No training or reserved generation launched.

## Question

Can the executive retain the previous curriculum's performance when task
composition uses the unified WSPL 0.3 compiler and identity-bound module library?
This is a language-integration repeat, not evidence of WSPL superiority or a
new controller capability.

Keep the six existing author-generated module artifacts exactly unchanged.
Audit that their saved source compiles through the unified module frontend to
the same rational artifact hashes. No block-author retraining is necessary for
this migration: its module grammar and target weights are unchanged. Editing
and rounding bridges receive engineering checks, not a new authoring benchmark.

## Matched training and evaluation

Fresh Qwen2.5-Coder-1.5B-Instruct rank16/alpha32 executive LoRA, 18,464,768 trainable
parameters. Same seed, 16,384 scene/task records, 256 development records, two
epochs/4,096 updates, task order and optimizer as expanded executive v2. BF16 base,
FP32 adapters/AdamW, LR0.0002, warmup160, cosine to10%, decay0.01, clip1,
microbatch1/accumulation8. Completion-only cross-entropy, context2048, no truncation.
Retain final epoch. Training now predicts WSPL header, library binding and typed
calls; token counts and compute will differ despite matched tasks and updates.

Use the identical 256 reserved v2 tasks: familiar3/4/5, long6/7/8, renamed8 and
jammed8,32each. These are reused evaluation panels, not fresh confirmation.
Generate256 new trained and256 untouched-base programs with the same greedy
512-token cap. Freeze all512 before execution. Reuse the previous trained
executive's256 saved replies without a new model call. Replay their execution
and require exact agreement with every saved score field. Add256 reference
programs, for1,024 execution slots. Original baseline generation usage is labeled
historical/reused, never counted as newly spent compute.

Same physical-state scorer and runtime:64expanded calls,160ticks per skill,
2,048ticks total. Only task composition is allowed for the executive; module
definitions or edits in its reply are rejected. Incorrect binding hashes,
unknown skills, bad types, failures and truncations stay in denominators. One
optional WSPL fence may be removed; no repair, feedback or resampling.

Report construction, per-panel task/safe-failure scores, paired gains/regressions
versus the210/224 ordinary-mission predecessor, missing deliveries, ordering
errors, tokens and wall time. Descriptive migration target: at least210/224 ordinary
successes and28/32 safe jammed failures, with all reference missions passing.
Report the actual numbers regardless of meeting these targets. One training seed;
the comparison changes prompt/output syntax and length together.

## Artifacts and operation

Archive linked library descriptor, corpus identities, complete replies, compilation
provenance, per-mission traces/scores, model/source identities and training logs.
One rolling checkpoint every30minutes and before epoch validation, plus final
adapter/clean-stop. No full base copies, API calls, automatic retry or follow-on.
Expected3–5hours;12-hour guard. Source/runtime/model, old replies/tasks/scores and
the six module artifacts are bound in the manifest. GPU preflight is synthetic
only and checks frozen base, zero-adapter logits and exact next-update restoration.

AGENTS.md requires exact manifest approval before this scientific run. This
protocol and implementation preparation do not register approval or launch it.
