# Reworked WSPL: original-tanh Breakout comparison

Prepared November 4, 2026. No authoring call or game evaluation has run for this
study. An exact source-manifest approval is required before launch.

## Question and scope

Can the rebuilt WSPL tensor editor author useful policies under CSAIL's original
fixed Breakout architecture, with complete replayable artifacts, at performance
comparable to fresh native editing? This is a descriptive feasibility screen.
The earlier ReLU source-compiler comparison remains a separate historical study
whose fairness/result writeup is paused. No old outcome is being overwritten.

Six fresh Astra/xhigh one-shot requests: three native and three WSPL. Order is
native/WSPL, WSPL/native, native/WSPL across three draw blocks. With an odd number
of pairs, first position cannot be exactly balanced. Draw labels pair schedule
and evaluation seeds, not the provider's uncontrolled sampling seeds.
All conversations are fresh; no reference solution, past reply, previous score,
test trajectory, tool execution or feedback is supplied. Prefix-cache usage is
reported separately. The proposer only receives the original game source/rules
and observation/action mapping plus the complete common two-interface guide.
The prompts differ only in the final selected-arm line.

Original architecture: raw 600 inputs ->32 tanh ->32 tanh ->6 logits,
20,486 FP32 parameters, all initially zero. Full six-tensor access is available
to both arms. Both deploy through the same unchanged NumPy tanh policy and
MinAtar 1.0.15 evaluator. No topology augmentation, scalar relay compiler,
learned feature adapter, hidden state, gradient training or local GPU is used.

Native uses the original compact literal-edit syntax. WSPL 0.2 offers a native
numeric compatibility profile plus exact-to-storage arithmetic, constants,
copying and bounded construction loops. Valid literal edits in the compatibility
profile reconstruct the archived three native controllers bit-for-bit. This is
a comparison of interface packages, including the optional numeric profile;
not an isolated surface-syntax effect. Both reach the same deployed function
class. Not all broader WSPL capabilities are exercised: placed circuits support
affine/ReLU, and graph surgery is reserved for a separately specified repair task.

Both arms require exactly one fenced candidate and reject any invalid statement
in full. This intentionally changes CSAIL's historical last-block/skip-invalid
policy, symmetrically. Failure-adjusted means assign invalid proposals zero as
an operational penalty, not a measured episode score. Valid zero policies run.
Resource limits and numeric differences are explicit in the
[shared guide](BREAKOUT_REWORKED_INTERFACE_V1.md). There is no automatic repair.

## Evaluation and interpretation

Every valid proposal receives the same 1,000 episode seeds `1110000+i` and action
seeds `1210000+i`, i=0..999. These differ from historical local panels. Sample
six actions using FP64 softmax of FP32 logits, sticky probability0.1, cap10,000
steps; retain horizon-truncated episodes. Log partial stopped episodes separately
without scoring them. Original upstream RNG is unavailable: this is a matched
local comparison using the original task/network, not bitwise upstream replay.

Primary endpoint: arithmetic mean of the three proposal means per arm, including
the declared invalid-proposal penalty. Report all proposal scores, valid counts,
invalid diagnostics, valid-only mean, episode standard errors, truncations,
between-proposal SD and draw-paired score differences. Three author draws are the
replication count; 3,000 episodes do not establish 3,000 independent authors.
Collection errors stop the queue; unfinished proposals are not imputed as zero.

Prospective descriptive progression rule: native failure-adjusted mean>=100;
all three WSPL proposals evaluable; WSPL failure-adjusted mean>=100 and >=90%
of native. A failing native baseline leaves feasibility unresolved. A WSPL
failure is a no-go for this configuration, not a language impossibility claim.
These thresholds are engineering decisions, not statistical noninferiority,
superiority, commercial acceptance or universal reliability claims. Kinetix
preparation may continue independently, but its replay/authoring requires its
own concrete protocol and source approval; no automatic scientific chaining.

Archive collection tokens/time, construction and evaluation time, numeric profile,
statement/loop/operation counts, nonzero coefficients, exact artifact identities
and emitted source size. Native normalization plans are post-hoc evidence and
never supplied to the proposer; construction timing includes archive preparation
and is not pure compiler or human-effort measurement. Hosted compute, energy
and dollar cost are not measured. No gradient/RL baseline is retrained here.

## Artifacts and evidence

For each attempted proposal preserve prompt/preview, exact reply, events, command,
stderr and reported usage. For valid proposals also preserve:

- Full source, immutable replayable edit plan, exact FP32 tensor artifact and the
  deployed NPZ (native format). Re-load both and check exact weight identity.
- All 70 storage-derived neuron equations and the author's original explanation.
  Semantic roles in that explanation remain unverified hypotheses.
- 32 fixed synthetic binary inputs for artifact/deployment-adapter parity, with
  fixture identity. This is an empirical integration check, not gameplay or a
  universal IEEE proof. Tanh formal status is explicitly unsupported.
- Every primary episode's seeds, reward, length, actions and truncation status.
- Full native Boolean observations, requested actions, rewards and terminal flags
  from the FIRST THREE primary episodes, with no extra simulator calls or cherry
  picking. These permit later rendering/inspection; requested actions can differ
  from sticky actions actually applied by the environment.
- Artifact index with file hashes, plus source receipt, config, dependency and
  CLI identities. Invalid/failed proposals remain in the denominator and archive.

One final artifact per proposal; no training checkpoints or redundant versions.
Canonical artifact/plan and deployment NPZ intentionally preserve distinct
replay/deployment formats. Raw bulky evidence remains in the isolated checkout;
compact replies/reports/indexes can enter Git after auditing. Do not commit
credentials, local path bindings or whole model service session stores.

## Collection and guards

Existing ChatGPT-backed Codex login, explicit `--model gpt-6-astra`, xhigh,
one response, no permitted tool events. The restricted CLI still advertises a
question tool; any use invalidates collection. Full offline prompt preflight
uses an empty authentication home. Commands and prompt previews are archived.
No model call is part of preparation. No API-key fallback, harness retries or
automatic relaunch. Bounded transport retries may occur within an invocation.

20-minute invocation guard, four-hour overall guard; expected roughly1-2hours,
depending on reasoning and episode duration. Existing Codex usage is consumed;
there is no enforceable dollar/output-token cap. CPU evaluator uses one math
thread; local GPU remains unused. Controlled stop through `control.json`.

Constructor: `skillforge.wsp_breakout.ReworkedBreakoutPolicy`.
Runner: `experiments/run_wsp_breakout.py::run`.
Preparation: `tools/prepare_wsp_breakout.py`.
Intended isolated checkout: `Skillforge-wsp-breakout-v1`.
Output: `results/wsp_breakout_v1`.
Launch only via `experiments/source_consistency.py` after exact digest approval.
