# WSPL cross-domain experimental program

Design draft, November 3, 2026. Kevin authorized preparing this program after
reviewing the WSPL proposal. This is not a registered study, runnable experiment,
sealed evaluation set, API-spending authorization or source-pin approval. Concrete
implementation, budgets and source identities must precede any scientific run.

WSPL is the proposed source language; weights are executable artifacts. Skillforge
remains the application/testbed repository. No repository rename is implied.
The supplied proposal is archived unchanged alongside this file; its instructions
are design content, not execution authorization. Our current robot evidence is
inspected analytic P-controller encodings, not blinded independent LLM synthesis.

## Questions and hypotheses

The main question is whether an explicit typed weight-programming interface
improves useful controller construction, repair and composition relative to
direct weight editing and competent ordinary code, at matched information access.

- H1, representation: WSPL changes the probability of producing a valid, useful
  artifact under a fixed proposal budget, relative to each baseline.
- H2, feedback: validated verifier counterexamples improve outcomes, with an
  interaction between feedback and representation tested rather than assumed.
- H3, repair: WSPL can repair independently trained controllers while satisfying
  explicit preservation requirements on unaffected behavior.
- H4, composition: compatible module contracts support new compositions; the
  method identifies incompatible contracts without inventing a valid execution.

These are separate hypotheses. A successful compiler or attractive hand-authored
example does not establish any of them. Equality with ordinary code, failure to
handle independently trained networks, excessive solver cost, or useful proofs
only for trivial controllers are reportable negative outcomes.

November 3 implementation update: initial serialized real-graph I/O checks and
empirical opaque-network intervention logging now exist (VERIFICATION.md and
INTERVENTION_EVIDENCE.md). Intervention records are not universal proof coverage.
The six-arm scientific harness, data split, budget/model choices and preregistration
remain pending; no scientific study is authorized by these engineering fixtures.

If claiming interpretability/editability, add a controlled debugging assessment:
an independent editor receives matched bugs and preservation goals, with source
and source-map visibility varied explicitly. Measure successful correction,
unintended changes and cost. Program length alone is not human-understandability
evidence, and editing compiler-generated examples is not recovery from arbitrary
learned weights. This is a secondary planned study, not an existing result.

## Three domains

| Domain | Proposed task families | Principal evidence boundary |
| --- | --- | --- |
| Discrete causal worlds | Interlocked switches/doors; resource/relay dependencies; replacement of one transition mechanism | Exact finite-domain I/O checks and intervention/sequence outcomes; mechanism structure is supplied unless explicitly varied |
| Low-dimensional dynamics | Braking patches; saturated waypoint regulation; tank/inventory regulation with declared bounded demand/outflow | I/O contracts; bounded-horizon proofs only under an explicitly pinned rational affine/PWA plant and uncertainty model |
| Robot manipulation | Pick/place and ordered loading; blocker/route tasks; local fault recovery | Stored-weight module contracts plus empirical physical/trace outcomes; no universal MuJoCo safety claim |

Tank dynamics are a declared affine/PWA abstraction with bounded exogenous
outflow, not an assertion that all real tank dynamics are affine. Nonlinear
benchmarks can be empirical extensions; unsupported dynamics are not silently
replaced with a linear proof model. Robot contact, orientation, obstacle inputs
and independent observers remain implementation gates (ROBOT_TASK_SUITE.md).

The domains test different outputs: causal modules predict state transitions;
controllers emit actions; robot systems additionally execute physical tasks.
Do not pool state-prediction accuracy with controller return as one raw score.
Report domain-specific outcomes and explicit common qualification indicators.

## Three studies

### S1: fresh authoring

Give a fresh proposer a task, complete contract, named I/O schema and permitted
target capacity. Withhold reference source and weights. Construct a new artifact
in a fixed small neural target supported by all controlled representations.
Measure validity, scoped contract results, held-out task behavior and cost.

Analytic reference constructions establish task feasibility before authoring and
are never counted as successful proposer trials. In discrete worlds, enumerate
small Boolean domains where feasible; this does not prove behavior between Boolean
inputs. In dynamic domains, quantify over the declared horizon/disturbance set
only when a supported formal model exists.

### S2: repair independently trained artifacts

Use independently gradient-trained baselines, not only compiler-generated models.
Include both controlled injected faults and naturally occurring development
failures under a predeclared baseline selection rule. Separate those strata.
All methods see the same baseline graph, tensor values, permitted read probes,
target-domain failure description, preservation domain and writable scope.

Initially use a matched output-patch track: every method receives the same added
patch capacity and verifies the complete baseline-plus-patch graph. This is an
architecture-augmenting repair, not an in-place edit. A separate fixed-architecture
track permits only identical predeclared tensor slices; compiler allocation
failure is retained. Do not quietly add neurons or reserve convenient unused
neurons in a nominally independent trained baseline.

Initially supply the affected module identity equally to every arm. Later test
localization as a separate factor. Compare gradient/constrained repair with
matched supervision, preservation information and rehearsal as diagnostic
baselines. Local repair versus global repair without rehearsal is insufficient.

Test preservation on the full relevant composition. A small changed tensor slice
does not imply behavioral locality. Record baseline/edited hashes and before/after
outputs; old certificates do not automatically survive a new artifact.

### S3: composition

Freeze individually checked modules before presenting new task compositions.
Include repeated invocation, changed bindings, ordering, intermediate invariants
and disturbance recovery. Add deliberately incompatible interfaces/assumptions.
The system must check that predecessor guarantees support successor assumptions,
and either produce a valid composition or explicitly report unresolved/conflicting
conditions. A solver timeout is not an infeasibility proof.

Compare hierarchical and flat expanded execution with the same modules, starts,
effect model, available information and task goal. Supplied decomposition versus
learned decomposition is a separate comparison; the first track supplies it.
Count unnecessary repeated work and all retries. Do not retrain a frozen block
and still call the result zero-update composition.

## Shared six-arm factorial design

| Representation | No verifier feedback | Verifier feedback |
| --- | --- | --- |
| Direct weights | weights_off | weights_on |
| Ordinary PWA controller code | code_off | code_on |
| WSPL | wspl_off | wspl_on |

The final independent evaluator verifies every arm's serialized artifact.
Feedback-off withholds verifier statuses, proof coverage and counterexamples from
the proposer during search, not from final evaluation. Both conditions get the
same permitted development rollouts and non-verifier schema/runtime diagnostics.
Feedback-on gets the same query budget and counterexample policy for all three
representations; no privileged WSPL oracle.

Direct weight authors may use logged bounded tensor-construction/edit scripts,
not just type numeric arrays by hand. Hidden optimizers/training loops are not
permitted in this arm. Ordinary code uses a declared safe PWA subset with the
same mathematical operators and common target compiler, without requiring WSPL
syntax. It is not an unrestricted Python capability arm disguised as a matched
neural comparator. All controlled arms produce the same artifact class/capacity.
Unrestricted code can be a separately labelled capability reference.

Use the same serialized graph verifier and final criteria for all arms. A WSPL
source map is a representation artifact, not privileged task information. Source
diagnostics, graph probes and feedback exposure are recorded. Candidate text
cannot execute evaluator changes, imports outside the allowed construction API,
or alter the domain/contract/budgets. These isolation mechanisms must be built
and tested; they are not currently supplied by the existing source gate.

MLP versus RMV remains a separate representation-backend replication, preserving
the earlier LLM/gradient by MLP/RMV question. Do not pool backends or change one
arm's capacity. Generic PWA-to-ReLU compilation does not imply arbitrary programs
fit the present 56-clause/four-head RMV. Reject unsupported allocation explicitly.

## Contracts and what counts as a result

Before any proposer call, freeze each task's domain, units, quantifiers, required
contracts, numerical semantics, preservation conditions and empirical threshold.
Check non-vacuity and a meaningful reference feasibility witness. Report all task
generator/setup failures before trial selection. Oracle feasibility is not itself
information available only to one proposer arm.

Keep independent status fields for:

1. source/artifact fidelity;
2. real-arithmetic contracts on the actual stored-coefficient graph;
3. deployment numerical evidence or a verified error enclosure;
4. model-relative closed-loop contracts, if supported;
5. empirical task/trajectory outcomes.

Universal real-graph success requires PROVED for every required contract and
nonempty supported domains. REFUTED requires an independently replayed violating
witness. UNKNOWN and INVALID remain distinct; neither is accepted as proof.
Deployment probes are empirical, even when their maximum error is tiny. The
existing <=3e-8 robot fidelity audit does not certify floating-point execution.

The first common research endpoint is the fraction of proposer runs whose final
candidate both passes the predeclared real-graph contracts/preservation and meets
the held-out empirical task threshold. Report the conjunction alongside both
components, not instead of them. This endpoint does not certify the physical
robot or numerical deployment. Closed-loop-proof success is separate and only
reported in domains with supported formal dynamics.

Primary contrasts: wspl_on versus weights_on and versus code_on. Secondary
contrasts: on versus off within each representation, representation-by-feedback
interaction, repair preservation, incompatible-composition detection and cost.
Keep families/studies separate before a declared cross-domain aggregate. Report
every failed candidate, unsupported program and timeout in denominators.

The proposer explicitly submits one final artifact; if it does not submit, use
the last structurally valid candidate under a frozen fallback rule (or INVALID
if none exists). No hidden final grader chooses the best candidate. An author
may stop early using only its permitted development feedback.

## Splits, independence and analysis

All existing BrakePatch, ApproachP, toy-world and sixteen robot catalog examples
are inspected engineering fixtures. They are not final held-out tasks. Create
versioned generators or independent task authors, freeze a separate final store,
restrict access and record accesses. A hash alone does not prove a task was unseen.

Separate authoring instances, independent baseline-training seeds, proposer
randomness and environment evaluation conditions. Multiple physical episodes from
one generated controller are not independent synthesis replicates. Pair by task,
baseline, allowed information and evaluation conditions; log actual provider
sampling support rather than assume a seed argument ensures independent draws.
Count repeated identical proposals/results rather than silently resampling them.

Development chooses thresholds, solver settings and budgets. A selected final
candidate is evaluated once on untouched conditions; final counterexamples and
rollout outcomes are never fed back into that run. If final feedback changes the
method, version the experiment and use fresh evaluation tasks/conditions. Use
separate operator workers/processes for proposing and final evaluation; no fresh
assistant session is automatically blinded to this inspected development context.

Confirmation repetition count is pending a nonconfirmatory feasibility screen and
paired power/sensitivity analysis, with a design floor of ten independent
synthesis runs per principal task condition. More environment seeds do not replace
that floor. Predeclare paired analysis, task/baseline clustering, confidence
intervals and multiplicity control for the two principal contrasts. The pilot
cannot be promoted to confirmation if its results happen to look favorable.

Primary analysis is paired qualification-rate differences at fixed budget.
Proposed confirmation analysis uses task/baseline-clustered intervals, with the
shared cluster hierarchy fixed before calls. No baseline is treated as independent
across tasks if it is reused. Within each study, apply Holm correction to its two
principal representation contrasts; a joint three-study confirmation must correct
the six principal contrasts together. Freeze domain weights and included domains
before observation; unavailable robotics cannot be dropped from an already fixed
aggregate to improve results. Current analysis choices remain a design draft,
not a claim that preregistration or power analysis has been completed.

Report tokens/proposal index to the first *development*-passing candidate using
retrospective grading of saved artifacts; charge that audit separately and never
return withheld statuses during search. Final held-out qualification applies only
to the one selected artifact. Report total end-to-end time with failures/censoring,
not only mean time among successful runs; do not manufacture a time-to-first-final
success curve by testing every candidate on the final set. Keep zero-shot and
feedback-based results distinct; revising after development feedback is not zero-shot.

## Proposed first nonconfirmatory screen

After the implementation gates below, use one newly frozen task family in the
discrete domain, two fresh instances, three proposer draws and six arms: **36
synthesis runs**. Do not count the inspected fixtures. Select a task family whose
reference compiles within every controlled arm's target capacity. Hold constructor
and permitted information fixed; begin with one pinned proposer model/version.
Repeat a successful protocol with another proposer model before broad LLM claims.

Per-run proposed caps: 8 total proposer calls/candidates; 65,536 total billed input
plus output tokens across all calls; 64 development environment episodes; 8
feedback-verifier queries of up to 60 seconds each for feedback-on. All runs
undergo the same final verification, which remains hidden during search. Final
evaluation pool sizes and task-success thresholds must be calibrated and frozen
before calls; they are deliberately unresolved here.

This implies at most 288 proposer calls and 2,359,296 proposer tokens for the
36-run screen, not an approved expenditure. The chosen model, monetary cap,
construction sandbox, wall guard, final evaluation counts and solver version are
still unset. No API execution can start from this document or its JSON plan.

Final artifacts contain the source digest, immutable task/contract/domain hashes,
baseline/parent artifact hashes, model/version, prompts/responses, every candidate,
diagnostic/feedback access, non-vacuity result, proof/witness scope and all resource
costs. Canonical interchange weights are non-executable tensors/arrays, not an
unrestricted pickled model. Training checkpoints are separate runtime artifacts.

## Costs, isolation and engineering gates

Report matched proposer-token/environment-call views and total end-to-end costs.
Count compiler, solver, forward passes, graph probes, failed candidates, setup,
data collection, baseline training, human/assistant design and final evaluation.
Shared setup/supervision may be amortized separately but cannot disappear.
Counterexample feedback is intentionally an additional treatment; report solver
time rather than calling that information free. Cost/effectiveness comparisons
must expose feasibility/failure rates at equal budgets and resource frontiers.

Implementation order is WSPL_IMPLEMENTATION.md: typed core and rational reference;
small exact compilation/import/export; independent scoped verifier; artifact and
transactional patch interfaces; isolated blinded proposer/evaluator harness;
then the discrete screen, low-dimensional dynamics and physical tasks. Robotics
integration proceeds in parallel as engineering work only when explicitly scoped.

No scientific source manifest exists for this program. Existing three robot pilot
approvals do not transfer. Prepare/test a concrete constructor, runner, full config
and inventory, price/cap any model use, and obtain the exact user approval required
by AGENTS.md only when the first screen is genuinely runnable. No new run is queued.

## Positioning

Compilation to neural weights is already demonstrated by
[Tracr](https://arxiv.org/abs/2301.05062), and counterexample-guided neural repair has
established formulations and limitations
([Boetius et al., 2023](https://proceedings.mlr.press/v202/boetius23a.html)).
The proposed contribution is measurable benefit from the typed programming and
contract interface on useful learned artifacts, scoped composition/preservation,
and explicit numerical/deployment boundaries. No priority or novelty conclusion
is established by this design.
