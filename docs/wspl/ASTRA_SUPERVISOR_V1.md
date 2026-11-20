# Astra-led student capability engineering

Status: implemented pilot undergoing final preparation; not scientifically approved.

Implementation: supervisor_session.py, supervisor_transport.py, supervisor_workspace.py,
supervisor_workbench.py and supervisor_evaluation.py; source-gated runner
experiments/run_astra_supervisor.py. The implemented tool transport uses a persistent
filesystem and complete visible transcript across fresh Astra requests. A general
Python-program tool executes arbitrary research code in a separate Linux sandbox;
it is not native CLI shell access or a claim of resumed hidden reasoning. Requested
camera frames are archived and attached to the next author request. This replaces
the former fixed action/fit recipe with general coding authority.

The concrete envelope allocates 20 hours to development and a separate four-hour
final evaluator guard, with eight cumulative GPU job-hours during development and
64 author responses. No automatic retry after interruption. GPU and robot runtime
is an isolated Linux environment; the previous Windows environments are unchanged.

## Question and motivation

Can Astra diagnose a weaker robot executive, choose an adaptation method, and
produce an autonomous Qwen-based system that handles calibration changes while
retaining ordinary performance? Astra owns the learning strategy, not merely the
demonstration actions. The host supplies resources and independent evaluation.

The completed teaching-v2 experiment used a host-selected imitation recipe.
Astra solved eight of eight teaching missions (82 decisions, 13 probes). The
ordinary-practice student achieved 8/8 ordinary and 0/24 adaptation successes;
the demonstration-trained student achieved 6/8 and 1/24, issuing zero deployment
probes. The unchanged executive achieved 0/32. Both continuations ran 128 updates.
That result motivates diagnosis, not a presumption that more imitation solves it.
Full local evidence remains in ../WSPL-executive-teaching-repair-v2/results/
executive_teaching_v2. The arrival repair remains: three measured samples within
4 mm; independent success requires three true samples within 5 mm. No guarantee
under unbounded sensor noise is implied.

## Astra's development authority

Provide a persistent tool-enabled Astra/xhigh engineering workspace, not the
existing restricted codex_one_shot collector. Astra can inspect source and prior
logs, execute Python, create diagnostic scripts, inspect tensors and gradients,
generate data, choose training objectives and curricula, and evaluate revisions.
Retain its working notes and visible tool results across sessions. Do not claim
that hidden reasoning survives a restart; archive the actual session transport.

Provide full access to DEVELOPMENT robotic simulation: environment creation,
reset, snapshot/replay, seeds, robot controls, true simulator state and dynamics,
camera placement, images/video, disturbances, and scenario construction. It may
inspect the actuator mechanism and change it in experiments. Diagnostic access
is privileged and logged, not artificially restricted to student telemetry.
No physical robot or unrelated user filesystem is part of this access.

Provide Qwen2.5-Coder-1.5B-Instruct and the original trained executive, the
ordinary-practice continuation and the unsuccessful taught continuation. Astra
chooses its starting point and can revert, compare, or combine artifacts. Record
that choice and all inherited adapters. No requirement to start from the failed
taught checkpoint. Existing motor modules are available as a common library.

Allowed methods include direct base-weight edits, authored or trained LoRA,
gradient optimization, synthetic examples, imitation, reward-based training,
attention/KV initialization, runtime activation hooks, neural routers, and WSPL
circuits. Astra can implement methods not anticipated by this list. Do not force
a prescribed sequence or restrict it to one small JSON operation vocabulary.
All executable additions must be exported and counted as part of deployment.

## What must be learned and what counts

Primary outcome: a Qwen-based executive improves adaptive decision making with
the SAME frozen motor library and standard observation/action boundary. It chooses
probes, controller selection, retries, goal ordering and stopping autonomously.
Prompt changes and persistent memory are permitted and explicitly reported.

Secondary outcome: Astra may alter motor modules, add an external decision
algorithm, change deployed observations or otherwise redesign the system. Do not
forbid useful solutions, but classify these as system engineering, not evidence
that Qwen learned the decisions. Submission declares every component and its role.

A KV prefix is a runtime artifact, not by itself a durable weight edit. Fixed
hooks, prefixes and routers must have explicit initialization/reset rules and
finite memory budgets. They cannot call Astra or receive a regime label, hidden
physics, scorer output or task-specific answer from the evaluator. A hook that
computes actions itself is an external policy even if inserted inside a layer.

## Deployment contract

Use fresh missions from the known four conditions: ordinary, changed-from-start,
original-to-changed, changed-to-original. Three to five ordered waypoints. Keep
the repaired observation interface, noise, motor library, physical and decision
budgets, and independent scoring from teaching-v2 for the primary comparison.
Initial module selection is explicit. Simulator development may be unlimited in
scenario variety; final deployment must honor the frozen contract.

Submit a self-contained bundle implementing reset(episode_public_spec),
act(public_observation) and close(). The loader must instantiate the declared
Qwen model and all persistent artifacts. Per-step computation, GPU memory,
latency, token use, trainable parameters and extra circuit parameters are recorded.
No remote calls or access to developer workspace, sealed evaluator, previous
test outcomes or test-dependent files during deployment. Reset per-episode state;
declare any within-episode online learning and account for its compute separately.

For attribution, evaluate the submitted system with student changes disabled
where well-defined, and with supporting learned/runtime components disabled.
Report non-equivalence honestly if an ablation cannot execute. Hashing a model
does not establish that its decisions are causally used: trace actual Qwen calls,
action derivation and added policies. Do not call a hardcoded controller student
learning merely because it resides inside a module named adapter.

The initial implementation records Qwen forward-call counts in deployment RPC
traces and archives the complete submitted source. These counts are diagnostic,
not causal evidence, and are not tamper-proof against deliberately evasive code.
The automatic gate reports SYSTEM performance only. Component attribution and
applicable ablations remain a required audit before claiming student learning;
they are not silently inferred from a successful system score. No automatic
post-test editing or additional scored ablation is included in this launch.

## Sealed evaluation and stopping

Teacher sees task families, success rules and the development environment. Final
instances, seeds and scores live outside its accessible filesystem/process
boundary. Withholding a path or asking it not to read a file is not isolation.
Use a separately mounted evaluator runtime; the development worker must fail
tests attempting to read sealed fixtures. No final instances are generated in
the teacher workspace. Development fixtures are explicitly unlimited in variety
within the resource budget and cannot silently become final test data.

Freeze a single final submission before final evaluation. Compare it and the
saved ordinary-practice executive on the same 32 NEW missions (8 per condition).
Do not reuse teaching-v2's evaluated 32 missions as fresh confirmation. Keep the
previous learned-imitation result as historical evidence; it is not necessary to
rerun every historical arm. Score the selected starting checkpoint as an additional
baseline only when it differs from ordinary practice.

Exploratory go criteria: ordinary-practice baseline succeeds on at least 6/8
ordinary missions; no regressions on its successful ordinary missions; at least
four net adaptation corrections across the 24 changed/switch/return missions;
no condition-level net loss. Report all counts, paired outcomes, partial goal
progress, failures and tool usage. This is one engineering session, not a
statistical superiority or universal safety claim. No test-driven repair loop.

## Proposed resource envelope (requires launch approval)

Up to 64 Astra response cycles in a persistent tool-enabled session, 20 minutes
per response, eight cumulative GPU job-hours, and 24 hours total study wall time.
These are proposed budgets, not spent or approved funds. Simulator jobs can use
the remaining wall budget without an artificial small episode quota. CPU/GPU jobs
must be metered by the supervisor, including child processes; cancellation must
terminate owned descendants. Record actual GPU utilization and elapsed device-job
time; do not equate reserved time with FLOPs. Concurrent GPU jobs must not evade
the aggregate cap. Model calls consume existing Codex usage; no enforceable
dollar/output-token cap or API fallback is assumed. Keep the chosen Astra model;
stop rather than silently substituting another model on capacity errors.

One rolling resumable checkpoint per active training job every 30 minutes and at
clean stop; retain the final submitted model and essential ablation artifacts.
Reuse frozen base weights where possible. Archive compact edits, programs,
datasets and their hashes, action/physics traces, requested camera bytes, visible
session transcript, metrics, time and failures. No automatic scope expansion.

## Pinning dynamic code without preventing research

Keep orchestration, budget enforcement, artifact packaging and sealed evaluation
source immutable under the approved source manifest. Astra writes candidate code
in a distinct writable workspace outside that inventory. Each executed revision
gets a content-addressed snapshot and provenance receipt. Its results bind that
snapshot, weights, data and environment identity. Launch approval explicitly
authorizes generated candidate code under this envelope, rather than approving
each experimental revision manually. It does not authorize modifying the frozen
evaluator or silently replacing the supervisor. This policy must be implemented
and tested before asking for a scientific launch.

## Existing components and required implementation

Reuse PandaBackend/ObservationSession for simulation and camera transport;
TeachingMission for a reference task implementation; repaired StudentExecutiveEpisode
for the primary action contract; authoring_sft restoration and adapter utilities;
existing source gate and compact checkpoint conventions. teaching_training.fit
is an available recipe, not a mandatory training policy. codex_one_shot is NOT
the supervisor: it explicitly disables the tools this experiment requires.

Before launch implement and test:

1. Persistent tool-enabled Astra session with isolated development filesystem,
   requested image delivery, full simulator API access and auditable interruptions.
2. Student workbench for model load/save, inference/diagnostics and arbitrary
   candidate training/hooks, with transactional export and rollback.
3. Supervisor-owned aggregate resource accounting and descendant-process cleanup.
4. Frozen deployment package loader and offline execution boundary; validate
   generated hooks, KV reset, action output and absence of undeclared services.
5. Separately inaccessible final evaluator with paired starts and independent
   physical traces; test access restrictions rather than relying on instructions.
6. Synthetic end-to-end exercise with a scripted supervisor, including mutation,
   training, interruption, freeze, reload and evaluation. No paid model calls.
7. Final concrete runtime/configuration, source/resource pin and explicit approval.

No scientific run, Astra call or training is authorized by this design document.
