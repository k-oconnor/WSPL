# Compact closed-loop control and local mechanism replacement

Prepared November 2026. This is stage one of a stronger modular-control study.
No training, author generation or reserved evaluation has launched.

## Claims this stage can test

1. A small supervised executive chooses complete action sequences from observed
   state on longer tasks and new combinations of disturbances.
2. Live outcome feedback improves success over a nominal plan produced by the
   same frozen executive.
3. When one known actuator mechanism changes, replacing one WSPL-authored neural
   block restores behavior while executive parameters and five other blocks stay
   unchanged.

This remains a planar kinematic environment with supplied observations, not
contact physics, perception or real robotics. The new mechanism is specified to
the block author; autonomous causal identification is not tested. The author has
seen the movement function family, but not this negative-gain parameter range in
its original training distribution. No new algorithm discovery is claimed.

## Compact interface and runtime

The executive returns one line per decision: `transfer OBJECT to POINT`,
`move_to OBJECT_OR_POINT`, `grasp OBJECT`, `release OBJECT`, or `stop`. No header,
hash, namespace, loop, explanatory prose or multi-action answer is accepted.
The host binds the same six typed neural artifacts used in the completed compact
executive study. `transfer` expands into four existing primitive calls; it does
not select a task goal or recover from a failure.

Observations contain raw goal bindings and before constraints, delivered-object
names, held-object identity, currently blocked point names, home/safety flags and
last action/outcome/failure counts. Delivered objects are recorded from observed
release geometry, not from commanded intent. The prompt does not supply a sorted
plan, enabled-goal list, future event schedule, mechanism identity or teacher
answer. Motor modules use supplied live geometric state as before.

The executive chooses the next action after success or failure. No host-generated
retry or omitted-goal insertion occurs. Stop is scored against the full mission.
An independent monitor detects delivery-order violations, regrasp of completed
work, missing goals, final position and unsafe motion. Explicit scalar checks
are not proofs about arbitrary controllers.

Bounds:160ticks per primitive,4,096ticks per episode, at most2N+8 executive
decisions (the over-budget attempt is logged and rejected). Invalid programs and
unsuccessful runs remain in all denominators. Physical violations terminate the
episode. Observed-state skills and their stopping checks are host machinery.

## Disturbances

- **Slip:** one specified object's first geometrically valid grasp attempt is
  jammed until that skill fails. It then becomes graspable. Retrying requires
  another executive decision.
- **Moved object:** immediately after a successful approach, the target moves
  0.13units. The following grasp fails contact checks; a later approach uses its
  newly observed position.
- **Blocked point:** one destination is unavailable until a different object has
  been delivered. The release condition is not included in the prompt. The scene
  admits another enabled goal. Invalid attempts fail without issuing motion.
- **Combined:** all three occur within one mission; this combination is absent
  from training. There are no permanent failures or learned fault diagnosis in
  this first stage.

Training blocked tasks use independent goals to guarantee feasibility. The
combined panel also uses independent goals; do not claim unseen dependency
families from that panel. Other tasks use chain, independent, fan-in, fan-out or
random DAG constraints. Coordinates and names vary; scene geometry is shared in
distribution. Twelve-object scenes extend beyond the training object count.

## Training and frozen comparisons

Fresh rank16/alpha32 LoRA on frozen Qwen2.5-Coder-1.5B-Instruct for each of three
seeds113,227,341. Reinitialize all adapter A/B matrices; reset optimizer and RNG.
Generate16,384 training decision snapshots and256 development snapshots from
reference trajectories on2–5object missions. Whole missions belong to one split;
training/development seeds start at7,000,000/8,000,000. Training mission mix is
40% clean and20% each slip/moved/blocked, before snapshot sampling. Reference
teachers use the same observations available to the model. They provide
supervised labels; there is no gradient through the simulator or reward learning.

Two epochs/4,096updates per seed; total12,288updates. Microbatch1/accumulation8,
completion-only loss, BF16 base, FP32 adapters and AdamW. LR0.0002,160update warmup,
cosine to10%, decay0.01, norm clip1. Context2048, no truncation. Development NLL
at initialization and epoch boundaries. Retain final epoch, with no checkpoint
selection from final mission outcomes.

All three executive adapters and all replacement blocks freeze before any final
mission evaluation. Interactive observations during an episode are allowed;
evaluation never updates weights or tunes a recipe. Generate greedy actions with
64-token cap. The historical whole-plan executive uses its original prompt and
512-token cap. Authoring uses one512-token-cap reply. All replies are archived.

Nine fresh panels,16missions each,144unique scenes. Test seeds start at9,000,000:
familiar3, familiar5, long8, long12, renamed12, slip8, moved8, blocked8, combined12.
No final scenes were scored while preparing the study.

| Comparator | What it isolates |
|---|---|
| Three trained closed-loop executives | Repeated training-seed outcomes with live feedback |
| Same executive's nominal plan | Generate its actions without unexpected slips/relocations, freeze the sequence, then replay in the disturbed world; initial blocked-point information is preserved |
| Previous best compact executive | Original one-shot model/prompt on fresh scenes, with initial blocked points included in the scene; pipeline context, not a matched retraining control |
| Untouched Qwen | Same one-action observation interface, no new fine-tuning |
| Deterministic observed-state reference | Feasibility/ceiling, with no hidden event schedule |

Nominal planning retains initial blocked-point observations and the deterministic
blocked-point release mechanism, so it does not lose information available at
the first real decision. It removes only unexpected slips/relocations. The
fixed-plan controls advance through their frozen actions after failed calls;
they receive no opportunity to add a retry. This is deliberately explicit and
can be more permissive than the earlier stop-on-first-failure whole-program
executor. Clean and blocked-only nominal results reuse the identical closed-loop
episode, because no observation discrepancy exists. They are marked reused, not new trials.
The new executive has different supervision, prompts and execution structure from
the previous model. Do not attribute that comparison solely to data volume.

## One-block adaptation

Use the16long8 scenes for paired actuator conditions. Both motion axes reverse:
physical displacement=(-vx,-vy). The executive receives no mechanism flag and
is never retrained. Keep close/open/near/holding/safe byte-identical.

- Original movement block under normal mapping: already measured baseline.
- Original block under reversed mapping: negative control.
- Saved small WSPL author's one-shot replacement: signed gain=-0.73, speed0.07.
- Same-architecture2–4–3 AdamW replacement per seed, warm-started from the existing
  movement weights:4,096CPU updates,32sampled rows/update, LR0.003, zero decay,
  128fixed calibration rows. Seeds change minibatch sampling for local gradient
  adaptation.
- Analytic replacement: supplied-rule ceiling, also checked with reference policy.

Both author and AdamW receive the same128calibration rows; the functional rule is
supplied explicitly to the author and defines the gradient targets. Half the
inputs cover[-2,2]^2 and half[-0.15,0.15]^2. Report their information/algorithm
differences and wall times; this is not compute-matched optimization. The author
has at most27compiled parameters, matching the original movement network's
capacity. No author feedback, retries, fitting tools or hidden candidate selection.
An invalid author reply leaves a failed replacement condition; no analytic fallback.

Save all generated WSPL, exact weights and provenance. Check authored and analytic
weights against the desired scalar function over the declared input box using
the rational-real equivalence checker's explicit independent-specification mode.
It preserves both source/artifact identities and does not claim the candidate
was compiled from the desired specification. The default compiler-provenance
check remains strict. EQUIVALENT is scoped to that
block/function/semantics, not FP32 execution, physical safety or whole missions.
Report DIFFERENT/UNKNOWN as returned; proof outcomes do not trigger retries.

## Predeclared descriptive progression targets

Report every panel and every training seed, including unsuccessful outcomes.
Do not treat all repeated scenes/seeds as independent observations.

For each seed, seek at least90% familiar success;85% pooled long/renamed success;
80% on each individual disturbance panel and75% on combined disturbances. Require
at least20percentage points improvement over the same executive's nominal plans
on pooled disturbed scenes, with paired improvements/regressions reported.

For replacement, require normal long8 success at least80%, at least50percentage
points loss with the unchanged block under inversion, and the authored replacement
to retain at least90% of that executive's normal successes. Report AdamW and
analytic replacement alongside it. Reference policies must succeed on all normal
and analytic-inversion scenes. Hash checks must show unchanged executive files,
base model and other neural blocks. Meeting these targets is a progression
decision, not proof of general robotics performance or statistical superiority.

## Resources and next stage

Expected8–16hours;36-hour overall guard. No hosted/API calls. Three compact final
executive adapters, one overwritten rolling checkpoint per seed every30minutes
and at epoch/clean-stop boundaries, small replacement artifacts, all decision
and losslessly compressed motor/decision traces. Scores append during execution
and are consolidated once at completion. No base checkpoint copies, automatic retries or automatic
physics expansion. Partial runs are reported incomplete; no implicit resume.

If the first stage supports the claim, prepare a separately pinned physics study
with the same compact interface: contact-based grasping, transport and placement,
one mechanism change, fresh layouts and disturbances. Physical hardware follows
only after simulator validation and a separately specified deployment protocol.
This staged program seeks strong evidence; no outcome is guaranteed in advance.
