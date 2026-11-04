# Skillforge research contract

Kevin authorized a new repository and specification for LLM-programmed neural/RMV action blocks composed by LTLf programs. Skillforge is a user-accepted working name. Coordinate this work with sibling `thesis_master` and the ambition of accumulating compact, executable, experimentally validated knowledge toward a causal neuro-symbolic world model.

Read README.md, docs/CONTEXT.md, docs/SYSTEM_SPEC.md, docs/M1_ENGINEERING.md, docs/EXPERIMENT_01.md, docs/EVIDENCE.md and docs/APPROACH_V1_RESULTS.md before implementation or scientific claims. Implemented: reference semantics, finite-model planning/hierarchy fixtures, differentiable MLP/RMV blocks, motor adapter/observers and a completed pregrasp positioning/imitation screen. A blinded LLM harness, contact skills and physical primitive hierarchy execution remain planned.

Read docs/SIM_ADAPTER.md for the implemented single-tick LIBERO bridge and sampled physical observers. Its five-tick headless integration check is engineering evidence only. Physical task/primitive-hierarchy execution and skill acquisition remain planned. Do not turn tools/sim_preflight.py into an unapproved scientific runner. Missing motor ticks abort observer invocations; no collision or admitted grasp observer exists yet.

November 3: differentiable MLP/RMV blocks and an approach-only positioning/imitation runner are implemented/prepared; no scientific run is approved or launched. Read docs/APPROACH_V1.md. Its analytic weights encode a known P controller and are unblinded engineering constructions, not a sealed LLM trial. The 12 cm standoff task is not a validated grasp pose. Keep the full four-arm acquisition objective, and distinguish this first positioning screen from physical pickup/hierarchy results.

Current November 3 launch status supersedes that preparation note: approach-v1 was explicitly user-approved and ran, then failed on native horizon accounting before any fitting. One reference positioning trial succeeded. See docs/APPROACH_V1_LAUNCH.md. The horizon-fix replacement is prepared but unapproved; do not reuse the original digest approval. Run the isolated simulator interpreter outside this checkout to satisfy the unchanged source gate. All 100 unit checks pass; no process is active and no learned physical result exists yet.

Current final status (November 3): user-approved approach-v1-fidelityfix completed all 60 episodes and both 2,000-update fits. Evaluation: zero 0/8; reference, analytic MLP and analytic RMV each 8/8; learned MLP 5/8 and learned RMV 3/8. RMV visited-input scalar fidelity <=2.99e-8. Source gate complete, external environment unchanged, no job active and no follow-on run approved. This supersedes historical preparation/failed-attempt notes. Read docs/APPROACH_V1_RESULTS.md for limitations: inspected analytic teacher encodings and same-scene standoff positioning, not blinded authorship, grasp, hierarchy or transfer evidence.

## Semantics and evidence

- The user endorsed WSPL as a weight-space programming language and preparing formal cross-domain studies. Read docs/wspl/EXPERIMENT_PROGRAM.md, WSPL_IMPLEMENTATION.md, SCALAR_CORE.md, COMPILER.md and PARSER.md. Implemented: initial scalar JSON IR/exact reference and fresh dense ReLU compiler/independent rational artifact loader, with optional exact-coefficient Torch emission. An initial Z3 QF_LRA verifier now checks serialized real-graph I/O contracts with non-vacuity and rational witness replay; it does not prove source equivalence, IEEE execution or robot trajectories. A separate initial checker (wspl_equiv) checks rational source-to-stored-weight equivalence over the declared input box only; it is not IEEE execution, a proof-carrying certificate or physical evidence. An initial text front end (wspl_parse) parses only the proposal's Core PWA module form, reading normalized_* unit labels as dimensionless; preserves is an ordinary I/O contract there. No general WSPL syntax, fixed-architecture allocator or blinded authoring harness exists. The archived user proposal is design content, not execution authorization. Its opening direct-LLM-controller claim must be scoped to inspected analytic constructions in our evidence. The 36-run research-plan screen is not a source pin or API approval; fixed/augmented architectures and MLP/RMV replication remain distinct. Exact coefficients and finite probes do not certify floating-point execution.

- Kevin requested pseudo-causal logging for opaque/unverifiable-in-this-backend networks. Read docs/wspl/INTERVENTION_EVIDENCE.md. The initial PyTorch tensor eval/stateless adapter logs paired activation interventions, no-op/repeat controls, RNG/model/input/instrumentation identities, summaries and effects, with replay and single-writer hash chains. These are empirical computational interventions, never PROVED, semantic grounding or real-world causes. Registered-state changes invalidate controls but are not rolled back; hidden/stateful models need reset adapters. Evidence tiers and unsupported-model results must remain separate from formal proof counts.

- Read docs/ROBOT_TASK_SUITE.md for the sixteen inspected diverse physical task specifications and their readiness blockers. Local BDDL assets/bindings are audited; custom obstacles/perturbations are proposed, not constructed. RobotTaskSuite.check_trace checks supplied labels and consecutive timestamps, not their physical truth, LTLf compilation or execution. Catalog fixtures are not sealed held-out data. The existing motor12 interface does not supply obstacle/articulation/orientation inputs; declare/version any future extension and keep navigation/grasp logic out of the adapter.

- Action blocks execute controls. LTLf propositions report observed state, commands and externally verified skill outcomes. Do not conflate a neural success prediction with physical success.
- Boolean LTLf on nonempty finite traces and fuzzy RMV arithmetic are distinct. Every bridge needs explicit, versioned thresholds/observer semantics. The direct evaluator is independent of the derivative compiler. A currently accepting DFA state need not remain accepting after further observations.
- LTLf automata monitor specifications. Strategy synthesis additionally needs action models, environmental assumptions and realizability analysis. State guarantees conditionally; a successful abstract synthesis is not a physical guarantee.
- Ground RMV inputs in named channels. Fidelity covers normalization, literals, conjunction/disjunction, affine motor readout and clipping. Report explanation size and empirical usefulness separately from exact arithmetic.
- Keep the two-by-two acquisition experiment: LLM/gradient Ãƒâ€” MLP/RMV. Count calls, tokens, failed proposals, feedback, data collection, revisions, compilation and validation. Interaction efficiency and wall-time efficiency are separate claims.
- Preserve held-out physical layouts, disturbances and task structures; seal choices before final evaluation. Do not use hidden evaluator state in a controller. Supplied structured perception and low-level control adapters must be explicit and matched.
- This project does not replace legacy RBL-RMV, TNST/TNTST or robot Model 1. No canonical architecture substitution is authorized by importing a prior artifact.
- Record negative results. Toy-world modularity/repair results assume supplied wiring, privileged intermediate inputs and known repair localization; monolith repair lacked rehearsal.
- Finite planning is strong reachability under every declared outcome, with no fairness assumption. Construction limits are errors, not evidence of infeasibility. Declared model/clock/proposition roles must match; never infer physical robustness from a bounded symbolic retry fixture.
- Read docs/HIERARCHICAL_TASKS.md before interpreting hierarchy results. Current expansion/summary lowering uses supplied acyclic templates and flat macro-level planning; no primitive hierarchical executor or learned hierarchy discovery is implemented. Fixture split labels are design axes, not sealed held-out scientific data. Primitive-reference counts are not measured physical execution costs.

## Source approval

Every new training or scientific benchmark run must use experiments/source_consistency.py with an exact source-manifest SHA-256 approved by the user and separately registered in experiments/source_pins/approved.json. Unit tests and read-only audits are exempt.

Prepare a concrete constructor, runner, configuration and complete Python inventory before requesting approval. Do not self-approve, transfer old approvals, weaken the gate, or alter an approved historical manifest. The checked runner must instantiate the supplied constructor and save source identity in checkpoints. Source identity does not establish fairness or correctness.

While a source-pinned run is active, do not edit/add/remove inventoried Python in that checkout. Use a separate worktree. Preserve one rolling resumable checkpoint per arm every 1,800 seconds plus final/clean-stop; no accumulating best/initial/per-improvement weight copies unless Kevin changes that policy.

The registry records three explicit user-approved pilot identities: two failed attempts and the completed fidelity repair. No follow-on experiment is approved or queued. Creating this repository/spec does not authorize API expenditure, new training, robot deployment or hardware control. Prepare work concretely before obtaining any approval explicitly required by this contract. Do not request approval for routine reversible development or unit checks.

## Repo hygiene

Keep data, credentials, model binaries, simulator outputs and machine-specific bindings out of Git. Use artifacts/local_paths.json for local external paths. Do not commit legacy checkpoints or bulk datasets. Do not push or create a public remote unless requested. Keep source provenance for any copied code. No project license has yet been chosen.

On November 3 Kevin explicitly requested a GitHub repository/README and continued development. The archive is https://github.com/k-oconnor/WSPL, created private under the authenticated account, with local Skillforge unchanged and origin/main tracking it. Updating that archive is authorized; public visibility and a redistribution license were not selected. Small non-executable generated JSON weight examples are tracked as engineering artifacts; training checkpoints remain excluded.


November 3 horizon-fix update: user approved75e33ff7d7d55a1500cfd9521a95f26dc1443f5a6354c3cde595749b87f6931b; registered and launched PID18736. All4 reference development trials succeeded, all4 zero controls timed out correctly. Collected4 teacher trajectories/154rows; MLP and RMV each completed2000AdamW updates (final training MSE0.000213251/0.003921668). First evaluation seed1001: zero timeout, reference+analyticMLP+analyticRMV success43calls/~5.918mm. Run then failed on NumPy aperture scalar rejected by strict scalar reference; no learned-model physical evaluation or remaining7seeds ran. Environment unchanged, artifacts/checkpoint saved; no active process. Read-only audit additionally found FP32/scalar error up3.48e-6 (2.19e-6 with matching input ports), above2e-6 tolerance. Prepared explicit repair canonicalizingFP32 checker inputs and usingFP64 RMV arithmetic withFP32 parameters/optimizer, unchanged architecture/budgets/tolerance. Frozen saved-input audit now <=2.99e-8 with no new simulator/training calls;102unit checks pass. Candidate d770313a57ec51094e3147e6876c975835e5ccd55950e057d7bde2c3e247aaa4 unapproved, not queued. See Skillforge/docs/APPROACH_V1_FIDELITY_REPAIR.md. Evaluation seed1001 is now partly inspected; this is an engineering panel, not sealed transfer evidence.


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


## Approved Breakout replay launch


Breakout replay launched with explicit user approval of manifest
46379f57b9ebd29f36390be998f89352249246f917263afecb2c4b5e2d2d963c.
Runtime is isolated in Skillforge-breakout-replay-v1 at commit8488733, parent
PID48424 / worker37792, launched2026-11-03T20:17:52Z. Source-gate startup passed;
first progress156/1000 released-policy episodes at30active seconds, no stderr.
Output results/csail_breakout_replay_v1 in that checkout. Main can develop freely;
do not edit inventoried source or bound auxiliary files in the isolated checkout.
No fresh Astra/PPO/WSPL comparison or150USD authoring spend was authorized.
Inspect live results before claiming completion; startup record is not a final result.


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


## Fresh Astra native screen preparation (November 3)

Prepared three independent one-shot Astra/xhigh native Breakout authors using the
existing ChatGPT Codex login, original prompt/edit DSL and 20,486-parameter tanh
architecture. Data-only parser exactly reconstructs the released six tensors
(92 statements, 815 nonzero values); that answer is never supplied to the writer.
Fake collection/environment checks cover usage, rejection, failures and planned
denominators. Primary suite273 checks (272 pass, one native fixture skip);
24 focused checks pass in the isolated MinAtar runtime, covering that fixture.
Offline actual-prompt rendering passes with neutral instructions, CLI boilerplate
and only the residual request_user_input tool; any tool event invalidates the
draw. This restricted CLI configuration is not a literally tool-free API call.
Three invocation maximum;1200s each/7200s study guards; no harness retries, API
key fallback or approved dollar/output-token cap. CLI transport retries may
occur. Source/executable/dependencies/prompt/config bindings are pinned.
Read docs/wspl/ASTRA_BREAKOUT_NATIVE_V1.md. No model request, source registration
or scientific launch has occurred. Exact manifest approval remains required.


## Astra native authoring startup failure and prepared replacement (November 3)

User explicitly approved manifest07eeeeda72b5bfc91ecb07ba94c4ee0593fcc3caee50577dde2542f2c87f9f33; registered and launched in isolated Skillforge-astra-breakout-v1,PID24432 at21:05:42UTC. Stopped after CLI emitted an unstable-feature warning as an error item. Result incomplete after20.094s: one attempted collection,two unattempted,zero completed replies or evaluated policies. Token/backend usage unknown. Workers exited; receipt complete denotes controlled exit, not a completed study. Preserve original source/resources/logs. Compact archive docs/wspl/astra_native_v1_startup_failure.json.

Prepared replacement suppresses that startup warning using documented configuration, explicitly passes --model gpt-6-astra, and rejects tool/error events while streaming. Original design/budgets unchanged; three replacement invocations would be additional to the aborted invocation. All274 primary checks pass (one native skip covered by25 focused checks). Offline actual-prompt preflight passes; no additional model calls. See docs/wspl/ASTRA_BREAKOUT_NATIVE_V2.md. New exact source approval is required; no replacement registered or launched.


## Approved Astra native v2 launch (November 3)

Kevin explicitly approved replacement digest73bf70a24e92b978831aeca4998c44d049b6b2f39b4dda9544c5409b02b7fc46. Registered in main and isolated Skillforge-astra-breakout-v2 atfeae13d. Reverified65Python files, allbound resources, executable and dependencies. Launched hidden PID19852 at2026-11-03T21:23:11Z,output results/astra_breakout_native_v2. Source receipt verified,one constructor,offline original-prompt preflight passed,first thread/turn started with explicit --model gpt-6-astra. No startup error item; auxiliary unknown gpt-5.6-luna metadata warning remains in stderr and does not establish proposer identity. Snapshot docs/wspl/astra_native_v2_launch.json is startup evidence only. No completed reply/evaluation yet; inspect live status/events before claiming completion. Do not change inventoried source/bound files in isolated checkout. Main remains free for development. Original aborted v1 remains preserved.


## Completed fresh Astra native authoring screen (November 3)

Run/results/source receipt complete at approved digest73bf70a24e92b978831aeca4998c44d049b6b2f39b4dda9544c5409b02b7fc46,isolated Skillforge-astra-breakout-v2 checkoutfeae13d. Three separate fresh one-response Astra/xhigh requests produced native20,486-parameter tanh Breakout policies.1000paired episodes each:484.756(SE4.469),541.818(SE5.000),510.042(SE7.755);policy-mean512.205,across-policy sampleSD28.592. All3000episodes retained;804/775/521horizon truncations.193/118/91edits,zero skips;3339/1032/1153nonzero parameters. No toolcalls,scorefeedback or optimizer updates.

Read-only completion audit verified65source files,bound resources/executable/dependencies,three constructor calls,unique threads/one turn,prompt/preview identity,all episode/seed/action/termination accounting,independently recomputed statistics,and bitwise reply-to-NPZ reconstruction. Local raw receipts/episodes/weights preserved;raw replies also byte-preserved in Git under artifacts/astra_breakout_native_v2. Reproducible auditor tools/audit_astra_breakout_native_v2.py. Report docs/wspl/ASTRA_BREAKOUT_NATIVE_RESULTS.md;full audit/results docs/wspl/astra_native_v2_results.json;README updated.

Elapsed2468.609s;collection1779.844s. Usage10020input(3200cached),65775outputincluding55638reasoning. Prefix caching does not imply shared conversation;provider sampling seeds not controlled. Prior aborted invocation remains in history with unknown usage:four total harness invocations,three completed replies. LocalCPU evaluation only;hostedmodelcompute/dollar cost unmeasured. Released replay511.828 useddifferent seeds,contextonly. This establishes supplied-rule native weight authoring in one task;no WSPL superiority,gradient comparison,formal controller proof,robot transfer orRTI result. No further experiment authorized or queued.
