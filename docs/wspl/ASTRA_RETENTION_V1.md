# Astra neural controller selection with old-task retention

Prepared November19,2026. No new scientific run or calls launched. Kevin requested
that old-task retention be an explicit requirement, then authorized preparation.
This is a new experiment, not a revision of the completed adaptation scores.

## Design

Preserve both original movement weights and Astra's completed27-parameter repair.
Preserve all other blocks and three Qwen executive adapters. Astra authors a
neural selector plus one diagnostic command; it cannot edit these controllers
or executive weights. This tests adding automatic selection to an existing skill
bank, not rediscovering the two mechanisms or training another LoRA.

The executable artifact is a WSPL-compiled ReLU selector, at most64parameters,
four inputs(u,v,dx,dy) and two scores(original,repaired). At each episode start,
the SAME agent-chosen command is physically executed once with zero gripper
command. Each component is bounded to[-.01,.01]. Observed displacement is measured
from before/after positions. The network receives only command and displacement;
no scene identifier, regime label, mapping or previous selection. Larger score
wins; ties choose original. The host performs generic score selection, never
mechanism identification. Selection then stays fixed for that mission.

Physical probe motion is retained, not reset or simulated away, and counts as
one tick. The executive observes its normal state after this motion. The old
controller without a probe remains the retention baseline. Probe coordinates
are explicit and noiseless. Regime changes occur only at episode boundaries;
each new episode resets physical mission state. Selector weights persist but
selection is recomputed, so this does not test detecting mid-mission switches,
recurrent identification or noisy sensing.

## Starting models and history

Reuse Qwen2.5-Coder-1.5B-Instruct with the completed executive LoRAs113/227/341,
each trained on16,384decision snapshots over2–5object missions for two epochs/
4,096updates. Rank16,alpha32,18,464,768adapter parameters; completion-only SFT,
AdamW.0002,decay.01,clip1,microbatch1/accumulation8,warmup160,cosine to10%,
BF16base andFP32adapter/state. The original neural blocks came from the separately
trained small WSPL author. No local gradient training or author-model generation
occurs here. The saved author remains available as provenance, not an active tool.

Astra is given both movement artifacts, the selector interface and feedback on
both regimes. It is told explicitly that the prior repair broke old behavior
and that this is unacceptable. The original adaptation result is prior evidence;
this is not a blind mechanism-discovery comparison. No old final scenes are reused.

## Astra interaction

One Astra/xhigh session, at most8responses. Fresh CLI invocation per response,
complete visible history, no hidden-reasoning resume. Requests: install a selector
source and command; evaluate; finish. Each request includes a prediction about
both regimes. Up to4candidate calibration evaluations plus initial feedback.
Invalid requests consume responses. Atomic installation, last accepted artifact
retained, no best-of selection. Start with zero probe and constant-original
selector. Freeze on finish or response-budget exhaustion.

The host executes declared operations; no arbitrary shell/Python, model-training,
environment-source or evaluator access. Preserve every prediction, program,
feedback and usage record. No automatic retries.20minute call guards,eight-hour
study guard. Existing Codex account usage without a hard dollar/output-token cap
or API fallback. Expected1–3hours; GPU for frozen local inference only.

## Evaluation

Calibration: four new scenes(two3-object,two5-object), each under both dynamics;
executive113 fixed in advance. Give diagnostic command/displacement/scores,
mission results and first three executive decisions. Do not send a runtime regime
label to the selector. Calibration case IDs are for the engineering agent only.

Freeze the selector/command before final evaluation. Sixteen fresh missions,
four each3/5/8/12objects; run the same set in three ordered phases:
original-before, changed, original-return. Phase names are only evaluator metadata.
Each of three executive seeds runs three arms:

-Original controller, no diagnostic motion: baseline.
-Original controller with the submitted diagnostic motion: motion-cost control.
-Neural-selected controller with the same diagnostic motion: proposed system.

432learned-executive episodes plus48observed-state reference episodes =480total.
The reference explicitly uses the correct controller for its regime, for feasibility
only, not as a learned selector or competitor. Original-return repeats the same
original scenes to check recovery under unchanged installed artifacts; these are
paired repeats, not additional independent tasks. Fresh calibration seeds21M,
final22M,4-step offsets; no final outcomes before artifact freeze.

Primary descriptive go rule: zero losses of baseline-successful original missions
in both original phases for EVERY executive seed; at least20percentage-point
changed-regime gain for EVERY seed; reference succeeds everywhere. Report all
results even if any criterion fails. Report paired improvements/regressions,
selector choices, one-step diagnostic costs and mission outcomes separately.
This is finite-panel retention across two known regimes, not universal safety,
new-mechanism generalization or evidence of WSPL superiority.

## Artifacts and authorization

Keep compact rolling/final selector artifacts, immutable original/repaired blocks,
all authoring history, local generations and compressed diagnostic/motor traces.
No full-model copies or automatic follow-on. Verify original executive/base/block
identities before and after. Separate exact source approval is required before
calls or benchmark execution. The previous run's approval does not transfer.
