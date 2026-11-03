# Working context

Kevin selected WSPL text syntax for the initial authoring comparison. Both WSPL
arms now accept Core PWA module strings through Mo's parser, with JSON IR internal
only. Candidate contracts are rejected, fixed input order/types remain enforced,
and final equivalence checks compare parsed IR to selected stored weights.
Request schema v1 invalidates old request hashes. All arms receive the same text
grammar documentation. Earlier notes describing a JSON-IR submission arm are
historical. Astra selection, 541 parameters and all budgets remain unchanged;
the proposed $150 is still not spending authorization.

Astra selected: Kevin explicitly chose gpt-6-astra for S1. The provider-neutral
file preparation path stays in place. Standard direct-API planning estimate is
$20-70 for 36 trials, with a $117.9648 token-only bound at the entire 2,359,296-token
allowance charged at $50/M output; reasoning tokens are included. Proposed total
cap $150 includes room for separately budgeted engineering calibration. This is
an estimate, not approval: monetary_cap stays null; no paid call or source pin
is authorized. Model pricing/assumptions are saved in the screen plan.

Initial-claim readiness update: merged frozen-tensor import and relational checks
are G3a engineering support, not a learned S2 baseline. S1 final grading now records
Mo's source-to-weight equivalence diagnostic for selected code/JSON-WSPL sources,
separately from the shared finite Boolean task score and never in search feedback.
FORMAL_EXPERIMENTS_V1.md maps C1 source fidelity, C2 bounded LLM authoring and C3
representation/feedback effects to their evidence. The 36-trial screen remains
nonconfirmatory and unlaunched. Budget/sampling selection and isolated file-based
collection are still prerequisites; no provider integration was selected.

Kevin explicitly selected provider-neutral file interface first for the S1
authoring screen. Keep public request export and transcript grading independent
of a provider; choose model and budget before launch. This selects a preparation
path, not a model, expenditure or scientific source approval.

November 3 formal-experiment preparation: FORMAL_EXPERIMENTS_V1.md separates S1
supplied-mechanism authoring, S2 independently learned controller repair and S3
frozen composition. S1 now has a six-arm finite Boolean kernel, public request
export and source-gated offline transcript grader, all at fixed 541 parameters.
The first screen is 2 fresh tasks x 3 draws x 6 arms = 36 synthesis trials.
No scientific proposer calls ran; task/model/cost freeze, isolated collection and
pre-call accounting remain pending. Sparse weights and one parsed PWA expression
are the pilot front ends; full scripted construction is not implemented. S2/S3
are specified, not runnable. See experiments/plans/authoring_screen_v1.json.

November 3 product validation: Kevin endorsed testing a narrow customer problem
after the PI's startup assessment. Proposed first workflow is controller change
review and repair, with discovery/paired-cost gates in PRODUCT_VALIDATION.md.
tools/demo_controller_review.py is a known-answer engineering demonstration:
compile, deliberate braking-gain regression, replayed counterexample, supplied
correction and optional empirical opaque-network probes. It performs no automatic
repair, training, API calls or robot execution. Customer value is unvalidated;
no outreach or scientific launch is authorized by the plan.

Kevin requested this new repo/spec on November 2, 2026 and accepted Skillforge as a working name to revisit later. The research idea is an LLM authoring small neural/RMV action implementations and temporal task programs, with validation, persistent reuse, frozen composition and local repair.

Current organizing direction: Kevin endorsed WSPL, a typed neural weight-space programming language for LLM-authored executable modules, and a formal cross-domain experimental program. The three studies are fresh authoring, repair of independently trained artifacts and frozen composition; domains are discrete causal worlds, low-dimensional dynamics and robot manipulation. All use direct weights/ordinary code/WSPL with and without verifier feedback, with shared final verification. The structured plan is an unregistered design draft, not launchable or approved for model expenditure. The scalar JSON IR/exact interpreter implement partial G0, and fresh dense ReLU compilation with independent rational artifact loading implements an initial G1. All 227 engineering checks pass. Initial Z3 real-graph I/O verification, rational source-to-artifact equivalence, a text front end for the proposal's module form and empirical opaque-NN activation intervention logging now exist. General WSPL syntax, fixed allocation, IEEE/closed-loop proof and blinded harness remain pending. The private archive is https://github.com/k-oconnor/WSPL; local Skillforge is its robotics testbed. See docs/wspl/EXPERIMENT_PROGRAM.md and SCALAR_CORE.md. Earlier positioning evidence remains scoped as below.

The task is broader than the preceding Gravemind toy-world program tests. Those were fixed compositions of Boolean state-transition predictors; they did not control a robot, synthesize a plan or evaluate blinded LLM weight generation. Their summaries/plot are copied with provenance, not relabelled as this project's results.

Current status: the user-approved fidelity-repaired pregrasp positioning pilot completed all 60 episodes and both 2,000-update fits. Evaluation: zero 0/8; reference, analytic MLP and analytic RMV each 8/8; AdamW MLP 5/8 and AdamW RMV 3/8. Scalar/neural RMV discrepancies on recorded visited inputs were <=2.99e-8. Source gate complete and environment unchanged. All 102 engineering unit checks passed; no source changed during the run. See APPROACH_V1_RESULTS.md. No process or follow-on experiment is active.

The first motor experiment is LLM versus gradient acquisition crossed with MLP versus grounded RMV. AdamW is acceptable; Kevin used SGD colloquially, not as an optimizer requirement. Compare acquisition cost, physical reliability, fidelity, frozen composition and repair; include rehearsal for global repair. Every skill must expose physically checked status propositions, not self-declared success. Fixed positive/negative literal rails are proposed for RMV to avoid shared-polarity cancellation and allow independent clause orientations.

Kevin selected MuJoCo -> robosuite -> LIBERO as the simulation stack. A fresh isolated Windows environment now passes an opt-in five-tick headless integration unit check: MuJoCo 3.2.7, robosuite 1.4.1, LIBERO 8f1084e3132a39270c3a13ebe37270a43ece2a01. The twelve channels include geometry-derived finger-pad aperture and object-specific contacts. Four commands map to OSC_POSE without a scripted task routine. Isolated Windows binding fixes and the initial missing-dependency failure are recorded. No scientific run is approved.

M1 engineering fixture: the synthesized strategy handles one approach timeout and one grasp failure, then observes a grasp, lifts, and stops on an accepted finite trace. Infinite adversarial retry/fatal failure models are rejected. The compiler is checked against direct semantics on fixed and seeded mixed short-trace cases; the planner checks every declared outcome and a strictly decreasing winning rank. This is a symbolic supplied-model result, not a learned robot capability.

Next recommended experiment: keep architectures/observer fixed and test richer source-state coverage or logged on-policy teacher corrections. The analytic blocks prove a working feedback controller fits these architectures; four demonstration trajectories did not give either gradient arm complete physical reliability. This does not establish LLM acquisition advantage: analytic weights encode the known teacher and prior/design cost is unmeasured. Actual grasp/contact, physical hierarchy execution and a blinded writer test remain future work. No follow-on run is approved or queued. The repository remains local Git only.

Hierarchical task fixtures now cover transfer, transfer/park, two-object loading, blocker clearing with an unordered root, repeated object transfers, second-object grasp recovery and an immovable-blocker negative control. Pickup/put_down -> transfer -> load_pair templates reuse the same definitions/bindings. The planner infers the blocker prerequisite from the supplied world, but solves a flat product at the macro-summary cut. These inspected fixtures are not a sealed benchmark or a physical hierarchy result. Future comparisons must include matched flat execution and distinguish supplied versus learned decomposition.


November 3 launch update: original approach-v1 was user-approved, registered and launched. The environment was moved outside the checkout (identical content) after the gate rejected in-repository third-party startup imports. The reference reached the supplied standoff from 25.3 cm to 6.4 mm in 41 calls on seed 101. Zero control then hit a native-horizon bug: 20 settling calls were not added to the 120-call horizon. Run/gate failed before any fitting; models verified exactly untrained. A minimal horizon-fix candidate is prepared with unchanged settings and 100 passing tests. It requires a new exact-source approval; no job is active. See APPROACH_V1_LAUNCH.md.


November 3 horizon-fix update: user approved75e33ff7d7d55a1500cfd9521a95f26dc1443f5a6354c3cde595749b87f6931b; registered and launched PID18736. All4 reference development trials succeeded, all4 zero controls timed out correctly. Collected4 teacher trajectories/154rows; MLP and RMV each completed2000AdamW updates (final training MSE0.000213251/0.003921668). First evaluation seed1001: zero timeout, reference+analyticMLP+analyticRMV success43calls/~5.918mm. Run then failed on NumPy aperture scalar rejected by strict scalar reference; no learned-model physical evaluation or remaining7seeds ran. Environment unchanged, artifacts/checkpoint saved; no active process. Read-only audit additionally found FP32/scalar error up3.48e-6 (2.19e-6 with matching input ports), above2e-6 tolerance. Prepared explicit repair canonicalizingFP32 checker inputs and usingFP64 RMV arithmetic withFP32 parameters/optimizer, unchanged architecture/budgets/tolerance. Frozen saved-input audit now <=2.99e-8 with no new simulator/training calls;102unit checks pass. Candidate d770313a57ec51094e3147e6876c975835e5ccd55950e057d7bde2c3e247aaa4 unapproved, not queued. See Skillforge/docs/APPROACH_V1_FIDELITY_REPAIR.md. Evaluation seed1001 is now partly inspected; this is an engineering panel, not sealed transfer evidence.


## WSPL text front end (November 3)

Mohammad Askari took the proposal's M0 text parser (issue #3, branch feature/wspl-text-parser). skillforge.wspl_parse lowers the proposal's module form to the scalar JSON IR and contract suites; see docs/wspl/PARSER.md. The verbatim proposal BrakePatch compiles to layers identical to the committed artifact, including the proposal's stated first-layer weights, and its three parsed contracts PROVE. ApproachP text reproduces the committed IR and contract suite exactly. Open decision for Kevin: normalized_* unit labels are read as dimensionless (IR unit "1"), following the proposal's "explicit normalization" wording. preserves is parsed as an ordinary I/O contract; tensors, functions, stateful modules and requires domains are rejected. 20 new engineering checks. No training, simulator, API call or source-gated run; the S1 study design is unchanged.


## Diverse robot task catalog (November 3)

At Kevin's request, added sixteen inspected physical task specifications in Skillforge/examples/robot_task_suite_v1.json and docs/ROBOT_TASK_SUITE.md: box/bowl transfer, unordered/ordered two-object loading, sorting, same-object round trips, swap via buffer, drawer and microwave chains, stacking, carrying around/over barriers, causal blocker clearance, one-second holding, recoverable slip and second-item recovery preserving earlier progress. Counts span one to three declared object pickup occurrences. A read-only audit verifies native bindings and hashes across seven local LIBERO BDDL scenes: five existing-scene entries, five goal/initial overlays, six custom scene/perturbation proposals. No external checkout or physics was modified. New catalog validation/closed-label trace checks enforce distinct pulses, strict precedence, final-state retention, safety labels, consecutive 20 Hz timestamps and dwell; they do not ground labels, compile LTLf, plan or execute a robot. All sixteen report physical_runnable=false with explicit missing skills/observers/interfaces/executor; no new experiment launched or approved. All 114 unit checks pass (12 new). The current motor12 interface lacks obstacle/articulation/orientation scene channels, so future extensions must be versioned and matched rather than hidden in an adapter. Prefer clear_then_fetch as a causal flagship: unordered delivery goals, no supplied clearance precedence, with physical obstruction necessity/bypass controls to be established. First physical integration slice remains single transfers -> two-object loading/order -> obstacles/blockers -> articulation/dwell/recovery.


## Source-to-artifact equivalence (November 3)

Mohammad Askari took the G2 source-equivalence gap (issue #1, branch feature/wspl-source-equivalence). skillforge.wspl_equiv encodes the validated scalar IR directly in Z3 without importing the compiler, encodes the stored ReLU layers over the same inputs, and asks whether any output can differ on the declared box. BrakePatch and ApproachP are EQUIVALENT to their committed artifacts; gain, column-order, 1/10^6 bias and swapped-operator regressions are DIFFERENT with counterexamples replayed through both rational interpreters. Identity-binding and port mismatches are INVALID; budget/capacity limits UNKNOWN. This is rational real-arithmetic equivalence under a trusted solver and encoders, not IEEE execution, a proof-carrying certificate or physical evidence. wspl_verify.py is unchanged, so recorded contract fixture hashes remain valid; validation.json/unit_tests.txt receipts were not regenerated. 14 new engineering checks; 183 total pass with the formal-experiment checks. No training, simulator, API call or source-gated run.


## Frozen MLP import and relational contracts (November 3)

Mohammad Askari took G3a (issue #6, branch feature/wspl-mlp-import), the import and relational-query part of the S2 prerequisites. skillforge.wspl_import reads a wspl.mlp_import.v0 bundle (explicit dtype and tensor layout, 0x hex-float values, ports with domains, provenance; identity pre/postprocessing only) into a ReLUArtifact with allocation mode imported, whose layers are exactly the tensors and whose source_sha256 is the bundle hash. skillforge.wspl_relational checks parent/edited contracts over input.*, parent.* and edited.* ports in QF_LRA with witnesses replayed through both artifacts. Fixtures: BrakePatch and ApproachP compiled weights re-expressed as bundles (import fidelity only, not learned, never S2 baselines) and a deterministic pseudo-random untrained float32 MLP. wspl_artifact.py now accepts mode imported, which changes its recorded evaluator hash. 19 new engineering checks; 222 total pass. G3b (patch and preprocessing composition, transactions) and any learned parent remain; no training, simulator, API call or source-gated run.
