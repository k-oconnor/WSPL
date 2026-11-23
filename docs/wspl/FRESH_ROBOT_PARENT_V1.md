# Fresh local robot executive parent

November 23, 2026. Preparation only; no source approval, training, Qwen task
evaluation, or Astra request has occurred. This is a prerequisite to the missing
WSPL-authored LoRA comparison, not that comparison's result.

## Why this run exists

The original runtime, ordinary parent, trained adaptation adapter and motor
checkpoint are unavailable on this machine. Kevin requested rebuilding locally.
This study starts from the official Qwen2.5-Coder-1.5B-Instruct revision
`2e1fd397ee46e1388853d2af2c993145b0f1098a`. It does not reconstruct the archived
checkpoints or inherit their performance claims.

The historical raw-observation/action codec is copied byte-for-byte into
`skillforge.robot_codec`. It formats public measurements and aliases identifiers;
Qwen selects one full-vocabulary argmax token. There is no action mask, automatic
probe, module selector, waypoint selector, retry, or stopping decision outside
the model at deployment. Invalid tokens consume a decision and remain failures.

The robot domain is the existing MuJoCo/robosuite Panda reaching simulation,
not a physical robot. Frozen motor blocks are **supplied analytic controllers**
compiled from WSPL: nominal proportional control and the explicit inverse of
the known cross-axis calibration. They replace the unavailable original motor
artifact and are never described as freshly learned or Astra-authored skills.
The prerequisite uses only nominal calibration. A later study must separately
establish adaptation and retention under all four conditions.

## Fixed protocol

- Frozen BF16 base; FP32 rank-16, alpha-32 LoRA factors on all 196 attention/MLP
  projections: 18,464,768 trainable coefficients, zero initial B factors.
- Collect 64 ordinary missions, seeds 73,100,000 through 73,100,063; three,
  four and five ordered waypoints in rotation, at most 32 decisions per mission.
  The supplied ordinary teacher selects the documented nominal motor, reaches
  the next unvisited waypoint, then stops. It uses the public boundary only.
  Any failed teacher mission stops the entire run before fitting. No filtering
  failures away, additional draws, or automatic retries.
- Train 384 AdamW updates, batch eight (3,072 single-token supervised targets),
  learning rate 4e-5, weight decay .01, clip norm one, 16-update warmup followed
  by cosine decay to 10%. Sampling with replacement uses seed 113991. Model and
  adapter initialization seed 113. Each loss is the next action token over the
  full vocabulary; no EOS loss or prompt-token loss. Maximum context 2,048,
  no truncation. Log the exact sampling schedule and all collected rows.
- Freeze the final adapter before any reserved mission. No best-checkpoint
  selection or subsequent repair based on these scores.
- Evaluate unchanged base (the exact initial zero-B adapter) and trained parent
  on the same 16 new ordinary missions, seeds 73,200,000 through 73,200,015,
  three to five waypoints and 32 decisions. Save every prompt, token, invalid
  action, physical/decision/sensor trace and paired initial-state hash.
- Prerequisite readiness: at least 14/16 successes for the trained parent.
  Report base success, paired corrections and regressions as well. This gate
  establishes ordinary-task readiness only; it does not establish adaptation,
  language retention, WSPL superiority, or equivalence to the historical run.

Only training missions supply labels. Reserved missions are opened only after
the two evaluated conditions freeze. These deterministic panels are not hidden
from the infrastructure author; they are prospective readiness checks, not a
blinded independent benchmark. They must not be reused as the final author-study
panel. Historical raw-observation codec knowledge is deliberately reused.

## Runtime and limits

Native Windows: dedicated Python 3.10 simulation venv (NumPy 1.26.4,
robosuite 1.4.1, MuJoCo 3.1.6) and dedicated Python 3.13 learner venv
(Torch 2.12.1+cu126, Transformers 5.14.1, NumPy 2.4.4). The latter contains
independent copies of the installed dependency closure, not editable links to
unrelated projects. Windows robosuite binding/DLL changes are recorded by
`setup_fresh_sim.py`, following upstream installation instructions. All runtime
files, native libraries, simulator assets, model/tokenizer files and Python source
are inventoried before approval. GPU driver/OS identity is recorded separately.

One RTX 3090, one serial training job; no hosted author calls or associated API
budget. Eight-hour total guard; simulator RPC timeout 60 seconds. An external
launcher must terminate the process tree if the total guard expires, including
a stuck native GPU or simulator operation. No automatic retries or follow-on.
One rolling optimizer checkpoint plus a final adapter; all include source identity.
The run supports a cooperative `control.json` stop; it does not auto-resume.

Native Windows simulation does not provide the historical Linux/bubblewrap
arbitrary-author sandbox. This prerequisite executes trusted repository code
only. Author sessions must not be launched until an equivalent isolation
boundary or a separately declared restricted tool protocol is implemented and
tested. Installing the simulator does not resolve author isolation.

## Verification and launch boundary

Engineering checks include the supplied analytic motor/inverse relationship,
all four physical calibration fixtures with an explicitly privileged fixture
selector, real camera rendering, actual Qwen synthetic-token backward pass,
native/WSPL factor equality, restored factor/base identity, complete paired-result
accounting, teacher failure stopping before fitting, and a fake end-to-end runner.
Engineering fixtures are not task scores for Qwen. No task text is predicted in
the pretrained-model preflight and no optimizer update is applied there.

Concrete constructor: `skillforge.authoring_sft.AuthoringSFTStudy`.
Runner: `experiments/run_fresh_robot_parent.py`. The candidate source manifest
is prepared by `tools/prepare_fresh_robot_parent.py`. Launch only through
`experiments/source_consistency.py` after user approval of that exact SHA-256
and separate registry insertion. Historical approvals do not apply.

If readiness passes, freeze the new parent and separately prepare fresh matched
native/WSPL factor authors plus a newly trained adaptation control. If it fails,
retain and report the failure; do not automatically tune or extend the budget.
