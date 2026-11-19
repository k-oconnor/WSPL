# Astra-led actuator adaptation: first bounded pilot

Prepared November 19, 2026. Unapproved, unlaunched proposal. This replaces Opus
with Astra as requested. It is the first narrow implementation of the README's
broader agent-led capability-engineering direction, not unrestricted self-modification.

## Question and scope

Can a frontier agent infer an unfamiliar actuator response from observations and
produce a reusable neural movement repair, with no frontier agent at deployment?
The host exposes actuator probes, saved small-author generation, WSPL compilation,
literal neural-weight installation, warm-start AdamW fitting and calibration
rollouts. It does not supply the mechanism equation, inverse controller, hidden
source, final tasks or final scores to Astra. Astra's tool requests are returned
as JSON; the host executes them and supplies the result on the next turn.
No arbitrary shell or Python execution is provided. The pretrained executive,
other five neural blocks, evaluator and training-tool implementation are frozen.
Thus this pilot does not yet test executive retraining, harness editing or general
tool invention. A success permits preparation of broader experiments, not an
automatic expansion.

## Starting assets

Use all three completed closed-loop executive adapters (seeds113/227/341), the
original six-block library and the saved WSPL controller-author adapter. These
are frozen Qwen2.5-Coder-1.5B-Instruct plus rank16/alpha32 LoRA with18,464,768
trainable adapter parameters at their original training time. Author:1,536
compiler-generated specifications, two epochs/384updates. Executives:16,384
reference decision snapshots from2–5object missions, two epochs/4,096updates per
seed. All used completion-only SFT, AdamW peak.0002, decay.01, clip1, microbatch1
and accumulation8, BF16 base and FP32 adapters/state,2,048context without truncation.
Author warmup20, executive warmup160, cosine decay to10%. No executive or author
training occurs in this pilot. Original checkpoint/source identities are bound.

## Hidden environment and available information

The host uses a fixed cross-axis actuator mapping with unequal gains. Its exact
matrix is pinned in private runner configuration, excluded from the agent packet.
This is one deterministic mechanism, not a distribution of independent discoveries.
The public packet explains geometric error ports, original weights, command range,
tools and budgets. Isolated probes return actual displacement from the same
Workcell transition used in deployment. They are noiseless and reset at the origin;
identification is deliberately easy before adding noise or partial observability.

The movement interface stays x,y -> vx,vy,grip, with27parameters maximum. No
external runtime solver or frontier model is added. Candidate admission checks
ports, budget, finite sampled outputs and zero sampled gripper output; these are
engineering checks, not a universal controller or physical-safety certificate.

## Agent and budgets

One Astra/xhigh session, at most12responses, through the existing Codex ChatGPT
login collector. Each response is a fresh invocation carrying the complete visible
history; hidden reasoning is not resumed. At most32probe commands,4saved-author
calls,4candidate calibration evaluations and4,096total CPU AdamW updates. Initial
calibration is supplied separately. Every request must record a predicted outcome
before feedback. Invalid requests consume a response. Last accepted artifact is
retained; explicit finish or response exhaustion freezes it. Collection errors
stop the study with no automatic retries.

Training is optional and agent-directed: Astra supplies up to128input/target rows
per fit call and chooses its update count within the total allowance. The host
fits the current movement block, preserving its architecture, using AdamW LR.003,
zero decay, batch32, deterministic per-turn seed. Rows must be finite and bounded;
the host supplies no inverse labels. No comparison against a matched-gradient
optimizer is claimed in this feasibility run.

Calls have20minute guards; overall8hour guard; expected1–3hours depending on
author behavior. Existing Codex usage has no enforceable dollar/output-token cap
or API-key fallback. GPU is used for local frozen Qwen inference; fitting is CPU.

## Panels, freeze and reporting

Eight calibration missions (four3-object, four5-object), using executive113 fixed
in advance. Full traces are archived; feedback includes episode scores, up to four
early executive decisions and four early motor ticks. Later tool feedback is
also archived, along with full prompts, replies and usage. Final evaluation:

-32new changed-dynamics missions: eight each3/5/8/12objects.
-16new original-dynamics missions: eight each3/5objects.
-All three frozen executives, original versus repaired block:288episodes.
-Observed-state reference with analytic inverse under changed dynamics and
 original block under original dynamics:48additional feasibility episodes.

336final episodes total. No disturbances are added: this isolates mechanism
adaptation from the preceding study's disturbance curriculum. Panels use fresh
seed families11M(calibration),12M(changed),13M(original), with4-step seed offsets
covering dependency families. All final predictions occur after artifact freeze.
Report paired successes, corrections/regressions and controller-only replacement
identity separately for every seed and panel. Compare repair success with each
executive's available performance; do not attribute planning failures to the motor
block without trace evidence. Report original-dynamics regressions explicitly:
one fixed inverse need not work under two unlabelled actuator mappings. No learned
context selector is installed, and retaining the old block is not automatic retention.

Primary feasibility: at least20percentage-point changed-task gain for each seed,
with reference feasibility passing all final tasks. Report whether repaired
changed-dynamics success reaches80% per seed and original-dynamics degradation;
these descriptive thresholds do not establish generalization across mechanisms
or a WSPL advantage. All outcomes remain reported if thresholds fail.

## Memory, safeguards and interpretation

History records request/prediction, result, usage and resulting artifact hash.
The initial packet contains the available interfaces and current calibration
evidence, not the prior known-reversal solution. This avoids handing the agent
the mechanism family answer. Previous findings motivate the study design but
are not an independent memory ablation. SIA/SEAL motivate agent-directed updates;
SIM motivates prospective predictions and durable experiment records; EMPIRIC
motivates identification through interaction. This pilot is much narrower.

Keep compact last/final movement artifacts, all local generations and compressed
traces. No full-model copies. No automatic continuation, API fallback or extra
training. The precise source-manifest digest must be approved and registered
before any model calls, scientific rollouts or fitting.
