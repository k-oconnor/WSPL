# Prior evidence and what it supports

## Completed finite-world work

The two antecedent experiments live in external Gravemind worktrees. Small JSON summaries are copied byte-for-byte under provenance/toy_world; their original paths/hashes are in provenance/index.json. No weights, datasets or active source pins are imported as Skillforge approvals.

First pilot: supplied Boolean programs for a three-room, two-door/switch, one-object world. Analytic authored networks and independently AdamW-trained blocks with exhaustive local supervision both achieved exact composition through 32 actions. Random blocks failed. Replacing one door mechanism restored exact changed-world predictions; a stale mechanism failed. This verified implementation/interfaces. It was not blinded LLM synthesis, autonomous planning or robotic control.

Second pilot: limited-data independent blocks versus parameter-matched monolithic and auxiliary-output monolithic AdamW learners, three paired seeds. Held-out next-state accuracy at 28 unique source transitions was 75.40%, 37.70% and 36.51%; at 252 examples, 100%, 100% and 99.60%. Every method fit observed next states. With 16 intervention examples, local modular repair retained 100% tested unchanged behavior and exact long rollouts; global fine-tuning damaged unchanged behavior.

These results support supplied factorization and constrained repair in this finite setting. Modular training had known wiring and gold intermediate inputs; local repair knew the affected block; monoliths had no rehearsal. The repaired door had only four possible Boolean inputs. Auxiliary target labels did not remove the full information/interface difference. They do not establish LLM authorship advantages, general causal discovery or intrinsic monolithic limits. A rehearsal control remains proposed.

## Earlier robotic work

Legacy source: sibling rbl_rmv_icu, research/ntst_colearn/MODEL_FRAMEWORKS.md and experiments/libero_control. Existing symbolic routines, retry guards and supplied motor primitives are separate models. Their physical experiments motivate explicit action contracts, preparation costs, independent task outcomes and failure handling. Skillforge must not relabel those supplied motors as newly generated neural skills.

Prior motor trials included cases where zero action already satisfied broad geometric tolerances. The new physical experiment must establish nontrivial development cases before final evaluation. Existing libraries and neural-motor hybrids have mixed outcomes; none constitutes a completed weight-authored RMV/LTLf system.

## RBL-RMV workstation report

Kevin reports a separate 145M RBL-RMV model trained on approximately 500M FineWeb-Edu tokens / 30,000 steps, about 90% BLiMP and increasingly coherent generations. This is user-reported and not independently verified locally. The local rbl_rmv_icu copy appears older, with no matching 145M/BLiMP run. Its RieszFFN has a degenerate initial literal setting (lambda=0.5 with identical clauses); a small CPU audit found constant initial outputs and zero input gradients, although polarity gradients exist.

Do not infer that the local old source is the successful workstation architecture, or transfer language-model competence into a robot-skill claim. Obtain and audit the functional changes before reusing that RMV implementation.

## Current Skillforge evidence

Ninety-nine engineering unit checks cover schemas, direct/reference and DFA trace semantics, neural and scalar RMV arithmetic/fidelity, input/parameter gradients, strong planning, hierarchy fixtures, motor/observer contracts and source gating. A supplied finite pickup fixture executes a synthesized timeout/failure recovery strategy. A separate fresh headless LIBERO integration check reads twelve channels over five motor ticks with zero setpoint increments; see SIM_ADAPTER.md. The unblinded approach-only physical/imitation screen in APPROACH_V1.md is prepared but unapproved/unrun. There is no scientific Skillforge physical result, admitted robot skill, LLM task translator or physical robot synthesis yet. Abstract guarantees depend on declared effect models and assumptions; the unit fixtures and simulator wiring check are not learned-skill results.


November 3 launch evidence: the original pilot was approved and attempted. One reference pregrasp positioning trial succeeded (25.3 cm initial distance, 6.4 mm final, 41 control calls); the paired zero control then hit a runner native-horizon bug. No fitting or full evaluation ran. The failed attempt and untrained checkpoint are preserved, and the minimal horizon repair awaits a new exact-source approval. All100 unit checks pass. See APPROACH_V1_LAUNCH.md.


November 3 horizon-fix update: user approved75e33ff7d7d55a1500cfd9521a95f26dc1443f5a6354c3cde595749b87f6931b; registered and launched PID18736. All4 reference development trials succeeded, all4 zero controls timed out correctly. Collected4 teacher trajectories/154rows; MLP and RMV each completed2000AdamW updates (final training MSE0.000213251/0.003921668). First evaluation seed1001: zero timeout, reference+analyticMLP+analyticRMV success43calls/~5.918mm. Run then failed on NumPy aperture scalar rejected by strict scalar reference; no learned-model physical evaluation or remaining7seeds ran. Environment unchanged, artifacts/checkpoint saved; no active process. Read-only audit additionally found FP32/scalar error up3.48e-6 (2.19e-6 with matching input ports), above2e-6 tolerance. Prepared explicit repair canonicalizingFP32 checker inputs and usingFP64 RMV arithmetic withFP32 parameters/optimizer, unchanged architecture/budgets/tolerance. Frozen saved-input audit now <=2.99e-8 with no new simulator/training calls;102unit checks pass. Candidate d770313a57ec51094e3147e6876c975835e5ccd55950e057d7bde2c3e247aaa4 unapproved, not queued. See Skillforge/docs/APPROACH_V1_FIDELITY_REPAIR.md. Evaluation seed1001 is now partly inspected; this is an engineering panel, not sealed transfer evidence.


## Completed Skillforge pregrasp positioning pilot (November 3)

Kevin approved fidelity-fix digest d770313a57ec51094e3147e6876c975835e5ccd55950e057d7bde2c3e247aaa4, registered and launched through the unchanged gate. The run and receipt both completed: 60 episode resets, 1,200 preparation calls, 4,037 approach calls, 56.266 recorded CPU wall seconds, no APIs/GPU. All four development reference controls succeeded and zero controls failed. Both models fit 154 rows from four demonstrations for 2,000 AdamW updates. Eight-start same-scene evaluation: zero 0/8; reference, analytic MLP and analytic RMV each 8/8; learned MLP 5/8; learned RMV 3/8. Learned failures missed the position criterion (MLP 41-43 mm, RMV 56-86 mm), not just speed stability. The architectures represent a working controller, but this small imitation regime did not reliably acquire it. Analytic weights encode the known P teacher; this is unblinded engineering positioning evidence, not a blinded LLM advantage, contact-ready grasp, pickup, hierarchy or general transfer result. RMV visited-input scalar fidelity <=2.99e-8; analytic states bitwise unchanged, all evaluation starts paired, external environment unchanged. Final checkpoint SHA-256 a6ed5542ffacab0324bae3a2f105787bb4dd883a0ceab6aa7c24bb6ad6053146. Output sibling Skillforge/results/approach_v1_fidelityfix; audited report docs/APPROACH_V1_RESULTS.md. Prior failures retained; across three physical attempts 90.562 recorded seconds and 6,783 attempted calls, excluding unmeasured engineering/design/install costs. No process or next experiment active. Recommend testing state coverage/on-policy teacher corrections with fixed architectures before expansion.


## Diverse robot task catalog (November 3)

At Kevin's request, added sixteen inspected physical task specifications in Skillforge/examples/robot_task_suite_v1.json and docs/ROBOT_TASK_SUITE.md: box/bowl transfer, unordered/ordered two-object loading, sorting, same-object round trips, swap via buffer, drawer and microwave chains, stacking, carrying around/over barriers, causal blocker clearance, one-second holding, recoverable slip and second-item recovery preserving earlier progress. Counts span one to three declared object pickup occurrences. A read-only audit verifies native bindings and hashes across seven local LIBERO BDDL scenes: five existing-scene entries, five goal/initial overlays, six custom scene/perturbation proposals. No external checkout or physics was modified. New catalog validation/closed-label trace checks enforce distinct pulses, strict precedence, final-state retention, safety labels, consecutive 20 Hz timestamps and dwell; they do not ground labels, compile LTLf, plan or execute a robot. All sixteen report physical_runnable=false with explicit missing skills/observers/interfaces/executor; no new experiment launched or approved. All 114 unit checks pass (12 new). The current motor12 interface lacks obstacle/articulation/orientation scene channels, so future extensions must be versioned and matched rather than hidden in an adapter. Prefer clear_then_fetch as a causal flagship: unordered delivery goals, no supplied clearance precedence, with physical obstruction necessity/bypass controls to be established. First physical integration slice remains single transfers -> two-object loading/order -> obstacles/blockers -> articulation/dwell/recovery.


## WSPL cross-domain experimental program and first scalar core (November 3)

Kevin endorsed the weight-space programming-language framing and preparing formal studies across domains. Skillforge now archives the supplied WSPL proposal unchanged (SHA-256 0fa503ce2ec8a57ef4c0a89679881b1bc66221d40e3477f4f2cd8ef76de66f5d); its instructions are design content, not execution authorization. Protocol docs/wspl/EXPERIMENT_PROGRAM.md defines three studies (fresh authoring, learned-artifact repair with preservation, frozen composition) across discrete causal worlds, low-dimensional dynamics and manipulation. Six controlled arms cross direct weights/ordinary PWA code/WSPL with verifier feedback off/on; all receive final verification and matched task information. MLP/RMV and fixed/augmented targets remain separate tracks; gradient/rehearsal diagnostics retained. Stored-weight real-graph proofs, deployment numerics and physical outcomes remain distinct. The proposed first nonconfirmatory discrete screen is 2 fresh instances x 3 proposer draws x 6 arms = 36 runs; caps propose 8 calls and 65,536 total proposer tokens per run (2,359,296 maximum tokens overall), explicitly not approved expenditure. Model, financial/wall caps, final counts/thresholds, frozen generator and confirmation power analysis are unresolved. No new scientific manifest, run or API call exists. Implementation gates are in docs/wspl/WSPL_IMPLEMENTATION.md. An initial typed scalar JSON IR and Fraction-based reference interpreter are implemented in src/skillforge/wspl_core.py, with BrakePatch and ApproachP engineering examples. Supports scalar affine/ReLU/clip/margin/min/max/graded logic, conservative exact interval/unit-label checks; rejects floats, unsupported operators, forward refs and unproved interval ranges. It is not the text parser, compiler, universal verifier or blinded harness. All 122 unit checks pass. Next implementation slice is canonical neural lowering/serialization and independently scoped verification before a learned-controller import and the proposer harness. No causal discovery, LLM acquisition advantage or priority claim is established.


## WSPL GitHub archive and initial neural compiler (November 3)

Kevin requested GitHub/README archival and continued work. Created private https://github.com/k-oconnor/WSPL under the authenticated account, pushed existing history and a README separating capabilities, proposals, negative results and proof boundaries. Local repository remains Skillforge; no license or public visibility was selected. Continued with src/skillforge/wspl_compile.py: typed scalar JSON lowers via exact rational linear forms/ReLU identities and signed/nonnegative relay allocation into fresh dense ReLU MLPs. Independent wspl_artifact.py loads/evaluates non-executable JSON matrices without importing source interpreter/compiler; optional Torch emission rejects coefficient rounding and explicitly leaves IEEE execution unproved. BrakePatch reproduces exact 3->4->2->1, 29-parameter proposal weights; ApproachP compiles to 12->6->6->4, 148 parameters. Both generated tiny engineering weight graphs/source maps are tracked; no training checkpoint/dataset is published. Ten new cross-checks cover matrices, operator compositions, deep skip relays, serialization/hash mutations, capacity limits, CLI and FP32/FP64 finite probes; all 132 tests pass. This is fresh allocation only, not fixed-architecture/RMV allocation, universal equivalence proof, LLM synthesis or a new robot experiment. Next is independent scoped universal verification, then learned imports/patches and blinded harness. See Skillforge/docs/wspl/COMPILER.md. No scientific run or API call launched.


## Scoped verification and pseudo-causal execution evidence (November 3)

At Kevin's request, added a separate empirical computational intervention path for opaque networks alongside initial formal checks. wspl_verify.py uses optional Z3 5.3.3.0/QF_LRA against serialized rational-coefficient real graphs. All three BrakePatch I/O contracts and ApproachP bounds/open-gripper/direction contracts are solver-backed PROVED with non-vacuity witnesses; reduced gain and wrong input columns are REFUTED with independent Fraction replay. Hash/domain/unit/schema failures are INVALID; unsupported numerical profiles/capacity/budget are UNKNOWN. Results bind artifact/contract, verifier/evaluator source and solver/query hashes, with no independently checked proof trace. No source-equivalence, IEEE execution or robot guarantee is claimed. interventions.py logs one-input tensor-site zero/scale/replace probes on declared stateless/eval PyTorch models: baseline/no-op/intervention/repeat with matched global Python/NumPy/Torch RNG, registered-state integrity (including nonpersistent buffers), module/config/source/input/instrumentation fingerprints, invocation indices, selected features and output metrics. Replay checks identities/output digests; JSONL hash chains need an external trusted head and do not authenticate a malicious writer. Empirical statuses cannot be PROVED; no semantic/world-causal/universal interpretation is established. Stateful/tuple/non-PyTorch adapters and weight-edit lineage remain future work; unexpected registered-state mutation is detected, not automatically rolled back. Known-answer Tanh fixture effect/no-effect logs and replay are tracked; no trained model/large-model sweep/robot/API trial ran. All 151 tests pass (19 new). Docs: Skillforge/docs/wspl/VERIFICATION.md and INTERVENTION_EVIDENCE.md. Private k-oconnor/WSPL archive continues; scientific 36-run design remains not launchable and unapproved.


## Controller change review demo and customer validation (November 3)

Kevin endorsed progressing toward a narrow customer workflow. Added
PRODUCT_VALIDATION.md with a proposed controls/robotics change-review use case,
five discovery interviews, artifact/access/commitment gates and a proposed 2x
active-engineering-time hurdle conditional on matched quality. These are planning
criteria, not measured savings, customer commitments or an outreach authorization.
The known-answer tools/demo_controller_review.py compiles BrakePatch, deliberately
weakens gain, persists a replayed counterexample and checks a supplied correction;
optional opaque Tanh probes remain empirical. Full suite: 155 tests pass. Standalone
CPU demo: PROVED / REFUTED / PROVED plus effect/no-effect controls and exact replay.
No model calls, training or robot execution; no scientific run approved.


## Formal experiment preparation (November 3)

Implemented S1 finite supplied-mechanism authoring kernel, six arms at a fixed
541-parameter target, data-only sparse-weight/PWA-expression/WSPL front ends,
finite rational verification, development/verification feedback separation,
usage budgets, final selection, request export and source-gated offline transcript
grading. Preserves all 36 trial denominators and descriptive task/draw-paired
contrasts; 14 new engineering checks bring the full suite to 169 passing tests.
No scientific proposer calls or training ran. FORMAL_EXPERIMENTS_V1.md specifies
S1 authoring, S2 independently learned repair and S3 frozen composition with their
separate missing capabilities. The first screen is explicitly nonconfirmatory;
finite Boolean proofs are not continuous/deployment/robot guarantees. Narrow
sparse-matrix/single-expression front ends do not establish superiority over
unrestricted scripts/code. Model/sampling/money/task freeze and isolated live
collection with provider receipts/pre-call accounting remain pending. No new
manifest is approved, no source registry entry added, no experiment launched.


## Initial claim readiness after G3a (November 3)

Reviewed Mo's merged parser, equivalence, frozen-MLP import and relational checks.
No learned parent/patch transaction or S2 result is implied. The formal protocol
now maps C1 per-instance rational source fidelity, C2 bounded independent LLM
weight-authoring feasibility and C3 representation/feedback contrasts to explicit
evidence and exclusions. S1 retains its scalar JSON WSPL arm and 36-trial pilot;
it cannot establish text-language usability or broad superiority. Final selected
code/WSPL sources now receive Mo's independent equivalence diagnostic, never in
search feedback and separately from common finite task qualification. Three new
regressions include a ReLU tent that passes every Boolean task case while failing
continuous source equivalence. Full merged suite:225 tests pass; source receipt
updated. Provider-neutral preparation preference persists. No proposer calls,
training, robotics, source approval or scientific launch occurred. File-based
collection controls, model/budget, fresh task/access/prompt freeze and exact source
approval remain launch prerequisites.


## Text syntax selected for the authoring screen (November 3)

Kevin selected Core PWA text for both WSPL arms before scientific collection.
Candidate strings now lower through Mo's parse_module; JSON IR submissions and
candidate ensures/preserves clauses are rejected. Study I/O types/order are fixed;
all arms still produce 541 parameters and use the same primary evaluation and
budgets. Request schema v1 changes request hashes. The shared public documentation
bundle now includes PARSER.md. Submitted text remains in the candidate ledger;
final source equivalence checks the lowered IR against actual selected weights,
with the parser trusted separately. Updated matched-arm tests and two new
regressions cover text-only/contract rejection, typed lets and fixed input order.
All227 engineering tests pass. No proposer calls, paid execution, source approval
or scientific launch; Astra and proposed150USD budget remain as previously scoped.


## Breakout replay preparation (November 3)

Kevin requested proceeding with the CSAIL progression. Prepared native tanh
fixed-artifact replay and uniform zero-weight control, 1000 episodes each,
10000-step horizon, explicit per-episode environment/action seeds, CPU/one thread,
zero API calls/training, two-hour guard and controlled stop. Upstream commit,
reference files, MinAtar wheel, core runtime files and installed package versions
are pinned; source resources and NPZ remain ignored, notices/provenance retained.
Twelve focused checks pass in a fresh outside-repo MinAtar environment. Primary
suite:261 tests,260 pass and one native wiring skip covered in the isolated suite.
A read-only 32-synthetic-input frozen-weight audit differs from scalar math
reference by at most1.4862e-5; no released-policy episodes scored. Original
RNG/evaluator not released, so comparison is statistical and scope is explicit.
Protocol docs/wspl/BREAKOUT_REPLAY_V1.md. Source manifest is prepared unapproved;
registry unchanged. This does not approve fresh Astra, PPO training, WSPL
comparison or the proposed150USD authoring budget.


## Completed CSAIL Breakout artifact replay (November 3)

Run and source receipt complete; released and uniform arms each1000episodes,
2checkedconstructor calls, no stderr, workers exited. Released mean511.828
SE7.60482 vs upstream517.411 SE7.41695; uniform0.551 SE0.02478. Difference-5.583,
about0.526 combinedSE, descriptive agreement only. All2000 episode/seed/action/
termination/horizon accounting and recomputed statistics passed an independent
read-only audit. Source59Python files and pinned reference/auxiliary resources
unchanged. Active183.969s,CPUonly,0APIcalls/updates/GPU.497 released episodes
horizon-truncated and retained. Exactsourcepin46379f57b9ebd29f36390be998f89352249246f917263afecb2c4b5e2d2d963c,
isolatedcheckout8488733. Docs/wspl/BREAKOUT_REPLAY_RESULTS.md and compact
breakout_replay_results.json archive result/audit/raw-file identities. This is
fixed released-artifact replay, no fresh Astra authorship, WSPL advantage,
formal tanh proof, PPO reproduction or product win. No next run authorized.


## Completed fresh Astra native authoring screen (November 3)

Run/results/source receipt complete at approved digest73bf70a24e92b978831aeca4998c44d049b6b2f39b4dda9544c5409b02b7fc46,isolated Skillforge-astra-breakout-v2 checkoutfeae13d. Three separate fresh one-response Astra/xhigh requests produced native20,486-parameter tanh Breakout policies.1000paired episodes each:484.756(SE4.469),541.818(SE5.000),510.042(SE7.755);policy-mean512.205,across-policy sampleSD28.592. All3000episodes retained;804/775/521horizon truncations.193/118/91edits,zero skips;3339/1032/1153nonzero parameters. No toolcalls,scorefeedback or optimizer updates.

Read-only completion audit verified65source files,bound resources/executable/dependencies,three constructor calls,unique threads/one turn,prompt/preview identity,all episode/seed/action/termination accounting,independently recomputed statistics,and bitwise reply-to-NPZ reconstruction. Local raw receipts/episodes/weights preserved;raw replies also byte-preserved in Git under artifacts/astra_breakout_native_v2. Reproducible auditor tools/audit_astra_breakout_native_v2.py. Report docs/wspl/ASTRA_BREAKOUT_NATIVE_RESULTS.md;full audit/results docs/wspl/astra_native_v2_results.json;README updated.

Elapsed2468.609s;collection1779.844s. Usage10020input(3200cached),65775outputincluding55638reasoning. Prefix caching does not imply shared conversation;provider sampling seeds not controlled. Prior aborted invocation remains in history with unknown usage:four total harness invocations,three completed replies. LocalCPU evaluation only;hostedmodelcompute/dollar cost unmeasured. Released replay511.828 useddifferent seeds,contextonly. This establishes supplied-rule native weight authoring in one task;no WSPL superiority,gradient comparison,formal controller proof,robot transfer orRTI result. No further experiment authorized or queued.
