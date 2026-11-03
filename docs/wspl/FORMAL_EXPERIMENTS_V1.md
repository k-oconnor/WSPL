# Formal experiments for neural program authoring repair and composition

November 3, 2026. Three studies separate whether WSPL helps an LLM construct
weights, repair a learned controller and compose frozen modules. S1 now has an
implemented finite-domain experiment kernel and offline transcript grader. S2
and S3 are protocols with explicit implementation prerequisites. This is a
nonconfirmatory feasibility program; positive pilot results do not become
confirmation or establish commercial savings.

## Study overview

| Study | Question | First domain | Primary outcome | Readiness |
| --- | --- | --- | --- | --- |
| S1 | Does the interface help implement a supplied mechanism? | Boolean switch and door transition rule | Final fixed-size graph exactly matches every Boolean input | Kernel, request exporter and transcript grader implemented; model, task freeze and live controller pending |
| S2 | Can a learned policy be repaired while preserving unaffected behavior? | Small PWA dynamical controller | Required repair plus preservation plus empirical rollout thresholds | Frozen import and relational checks implemented; learned baseline, patch composition and transactions needed |
| S3 | Do frozen modules support new compositions and reject incompatible ones? | Finite causal mechanisms, then dynamics | Correct frozen composition and incompatible-contract detection | General composition checker and independent task split needed |

Robotics follows successful supported-domain tests. The prior standoff pilot and
sixteen task specifications remain engineering background, not held-out trials
in these studies. No grasp, obstacle or physical hierarchy capability is assumed.

## Initial claims and evidence requirements

| Claim | Required evidence | What the first screen can establish |
| --- | --- | --- |
| C1 A submitted supported scalar program and its stored neural weights compute the same function | Final selected source/artifact pair returns EQUIVALENT over the declared real input box, with source/graph/checker hashes and solver scope | Per-instance rational-real source fidelity, not correctness of the compiler for every program or textual parsing equivalence |
| C2 An independent LLM can author neural weights that implement a supplied mechanism | Frozen model/prompt/response provenance, every attempt retained, final graph correct on the complete Boolean domain under budget | Bounded authoring feasibility and measured success/cost on these task instances |
| C3 Representation and verification feedback change authoring outcomes | Six matched arms, independent trial state, paired task/draw comparisons and full cost/denominator accounting | Descriptive feasibility contrasts; two task instances cannot support a broad superiority claim |

These claims are distinct. C1 is a mathematical per-instance result under a
trusted encoder/solver; C2 and C3 require actual independently collected proposer
trials. Engineering fixtures and this assistant's inspected constructions cannot
stand in for those trials. The text parser is available, but this version's
WSPL arm remains scalar JSON IR: it does not measure the text language's usability.
Introduce a text arm as a separately versioned comparison before collecting data
if the desired claim concerns that syntax.

S2 repair of a genuinely learned parent and S3 composition remain subsequent
claims. Mo's import fixtures establish lossless tensor import and scoped
relational queries, not learned-model repair or a physical controller benefit.
Runtime safety, causal discovery, unrestricted-code superiority and commercial
time savings are outside this initial screen.

## S1 hypotheses and comparison

H1: at a fixed proposal budget, WSPL changes final qualification probability
relative to direct weights and ordinary code. H2: verification feedback changes
qualification, potentially differently by representation. Test both directions;
code equality or superiority is a meaningful result.

| Representation | No search verification | Search verification |
| --- | --- | --- |
| Sparse direct weights | weights_off | weights_on |
| Restricted ordinary PWA expression | code_off | code_on |
| Typed scalar JSON WSPL | wspl_off | wspl_on |

Every arm receives the same supplied mechanism, input names and domains, output
requirement, fixed target architecture, allowed development inputs, operator
documentation and budget. Feedback-on may receive full finite verification and
the first exact violating input. Feedback-off gets development feedback and
structural/runtime diagnostics only. All selected final artifacts receive the
same independent artifact-runtime evaluation.

This version operationalizes direct weights as sparse matrices and code as one
parsed PWA expression. It executes no candidate Python. The broader research
proposal's isolated tensor-construction scripts and general ordinary-code subset
are not implemented. This narrower comparison cannot establish superiority over
competent unrestricted code or scripted weight construction. Add those controls
before making that stronger claim. Record this as an explicit pilot restriction,
not an undocumented change to EXPERIMENT_PROGRAM.md.

## S1 tasks target and qualification

Each task specifies a Boolean mechanism for next_open. Four inputs are named
switch_a, switch_b, power and lock. The output is the OR of two distinct clauses;
each clause is an AND of two signed input literals. The complete rule is provided
equally to every proposer. Causal wiring is supplied; no causal identification
or discovery is tested. Tasks specify transition predictions, not robot actions.

Use two fresh task instances, three proposer draws per task, six arms: **36
synthesis trials**. The task generator is inspected infrastructure. Seeds 1
through 8 used by unit tests are development fixtures. Select different seeds,
check distinct nontrivial truth tables and record all setup/access before calls.
New seeds alone do not establish blindness. Feasibility references and final
evaluation must stay outside the proposer's process and files.

All arms emit **4 -> 12 -> 12 -> 12 -> 12 -> 1**, four ReLU hidden layers and
linear output: **541 stored parameters**. Zero weights still count. Code and
WSPL use the same compiler followed by zero padding and nonnegative hidden-state
identity relays; excess allocation is rejected. Direct weights use the same
matrix dimensions. Stored coefficients have absolute value at most 64 and
denominator at most 1024. Candidate payloads are at most 24,000 canonical JSON
bytes. Larger or unsupported candidates are INVALID, not silently expanded.

Four shared development assignments (binary indices 0, 5, 10 and 15 in declared
input order) are evaluated after each structurally valid proposal. Final grading
evaluates all 16 Boolean assignments by exact rational execution of stored
weights against an independent Boolean oracle. Qualification requires 16/16 exact
matches, the fixed architecture and budget compliance. Fractional near-correct
outputs do not pass. All 16 inputs are valid, so the domain is non-vacuous.

The other twelve inputs are reported as non-development inputs. They are not a
held-out set for feedback-on: full verification may disclose violations there.
Final qualification is full-specification satisfaction, not an estimate of
generalization to a hidden Boolean distribution. No claim covers real-valued
interpolations, IEEE arithmetic, closed-loop behavior or semantic sensor grounding.
The finite-domain PROVED status is explicitly distinct from Z3 real-graph proof.

For the selected code/WSPL source, final grading now also runs the independent
source-to-stored-weight equivalence checker over the whole declared real input
box, with a five-second query budget. It is final-only and absent from proposer
feedback. Direct weights report NOT_APPLICABLE because there is no submitted
scalar source. Missing Z3 or a timeout remains UNKNOWN, never EQUIVALENT.
This diagnostic is separate from the common 16/16 task endpoint; it cannot confer
extra task success on source-based arms. Report task qualification and source
equivalence jointly and separately. A compiler mismatch needs investigation and
a versioned repair, not quiet removal of the affected trial.

A regression injects a ReLU tent between the Boolean power values: the modified
graph still passes all 16 Boolean cases but fails continuous source equivalence.
This distinguishes the two evidence requirements directly.

## S1 budgets selection and cost records

Each trial permits at most eight proposer turns/candidates and 65,536 total
input plus output tokens. Input counts include documentation and all repeated
feedback context. Maximum screen exposure is 288 turns and 2,359,296 tokens;
these are ceilings, not approved expenditure. The provider controller must impose
output limits and reserve/check remaining budget before each call. The offline
grader can detect reported overspend but cannot prevent earlier API spending.

Each attempted candidate is retained and charged, including schema/allocation
failures. Every transcript is bound to its public request hash and each turn to
the preceding permitted feedback hash. Explicit final selection wins; without
selection use the last structurally valid candidate, never the candidate with
the best final score. An explicitly selected invalid candidate fails. No valid
candidate or zero returned turns remains a failed trial in the denominator.
Final grading is one-shot; revised proposals require a new trial/version.

Record full provider messages/responses and immutable model/version, generation
settings, sampling/seed support, tokens, elapsed time, monetary cost and failed
requests. The current kernel records submitted candidates and externally
reported usage; it does not independently attest provider identity, billed tokens
or that an upstream collector retained every failure. A live controller and
receipt audit are prerequisites for a scientific launch. Separate evaluator
time from proposer time; shared authoring/setup costs do not disappear.

## S1 implementation and execution boundary

- src/skillforge/authoring_study.py: mechanism generator and separate Boolean
  oracle, restricted AST parser, common fixed target, finite verifier, trial
  budgets, feedback firewall and final selection.
- tools/prepare_authoring_requests.py: public request/doc export, no answers,
  candidate generation or model calls. Refuses existing output directories.
- experiments/run_authoring_screen.py: source-gated transcript replay, complete
  pairing checks, per-trial ledgers/artifacts and descriptive summaries.
- experiments/plans/authoring_screen_v1.json: machine-readable draft; deliberately
  unset model, task seeds, sampling, monetary and wall caps.

After filling and freezing a separate study JSON, request export is:

```sh
python tools/prepare_authoring_requests.py --study work/frozen_study.json --output work/public_requests
```

There is no automatic provider execution command yet. Candidate code is parsed
as data, but this kernel is not an operating-system sandbox. A proposer with
repository access could inspect reference helpers and final evaluation. Isolate
its process, credentials, filesystem and conversation state; never call the
current context "blinded". Freeze identical documentation snapshots and a
balanced/randomized execution order before collection.

The transcript bundle maps each trial ID to request_sha256, turns and
final_candidate. A turn contains candidate, usage and previous_feedback_sha256.
Usage contains input_tokens, output_tokens, elapsed_seconds and cost. The first
preceding-feedback hash is null. Preserve raw provider receipts separately.
The grader constructor is run_authoring_screen.AuthoringScreen; the runner is
run_authoring_screen.run. Its approved configuration must bind the frozen study,
transcript path and SHA-256. It requires an existing verified source receipt and
uses the constructor supplied by the source gate. No source pin is fabricated
while model/data/cost choices remain unresolved. Candidate collection also needs
its own approved, bounded execution controller before scientific model calls.

## S1 analysis and progression

Report qualification numerator/denominator for every arm and paired differences
wspl_on minus weights_on and wspl_on minus code_on, pairing by task/draw. The
grader emits these descriptive contrasts. Report feedback effects, invalid rates,
duplicates, tokens and cost separately; do not silently resample repeated answers.
Three draws per task are not three independent task instances. With only two
tasks, do not claim task-general statistical significance or compute a naive
36-independent-observation confidence interval.

Use the pilot to choose an independent confirmation design and conduct a paired
power/sensitivity analysis. Preserve the existing design floor of ten independent
synthesis runs per principal condition, expand task coverage, and freeze task
clustering and multiplicity correction before confirmation. Confirmation must use
fresh tasks; do not promote a favorable pilot. A ceiling across every arm motivates
harder composition/repair tasks, not a claim that WSPL is superior.

## S2 learned controller repair

Train independent small MLP baselines with recorded seeds/data, then freeze and
import their actual tensors, preprocessing and numeric profile. Parent training
requires its own approved source identity. Do not substitute an analytic compiled
controller for a learned baseline. Prefer a small braking or saturated-regulator
plant with explicit model assumptions before robotics.

For each repair request define a target region, an unchanged region and a
transition band where behavior may change. Establish feasibility; continuity
must not force conflicting boundary requirements. Compare the same six
representation/feedback arms at matched patch capacity and information. Begin
with output-patch augmentation, labelled as such. Include gradient repair with
rehearsal, and separate supplied localization from diagnosis of which module failed.

Primary success is the conjunction of required repaired behavior, relational
preservation against the actual frozen parent, valid domain and predeclared
empirical rollout thresholds. Verify the whole composed graph. Measure retained
behavior, patch size, failed proposals, parent-training/data cost and repair cost
separately. Baseline/repair seeds and final disturbances remain independent units.
Freeze actual plant, target regions, budgets and repetitions only after import,
transaction rollback and relational queries pass engineering checks. None of
these capabilities is inferred from the operator-restoration demo.

## S3 frozen composition and incompatibility

Freeze checked modules before revealing composition requests. Supply contracts
and bindings equally. Test new orderings, repeated calls, changed bindings,
intermediate-state requirements and mechanism replacement without retraining
unaffected modules. Include satisfiable, incompatible and resource-unknown cases;
UNKNOWN is not INFEASIBLE. Distinguish supplied decomposition from planning.

Measure correct compositions, rejected incompatibilities, false acceptance,
retention after one replacement and full rollout correctness. Preserve module
hashes before and after every trial. Use a matched flat-expansion baseline and
charge retries. First use finite transition mechanisms with independently
enumerated reference behavior; extend to PWA dynamics and robot trajectories
only with their distinct observers and assumptions. Seal repetitions/thresholds
after the composition checker exists, before exposing final requests.

## Launch checklist

Choose model and sampling, maximum monetary spend and wall guard; freeze prompt
snapshots and fresh task instances/access records; implement/audit isolated
collection and pre-call accounting; pass failure and leakage checks; inventory
sources and bind all inputs. Then obtain the exact manifest approval required by
[AGENTS.md](../../AGENTS.md): "Every new training or scientific benchmark run must
use experiments/source_consistency.py with an exact source-manifest SHA-256
approved by the user." Preparing these experiments is authorized; the draft is
not a scientific launch or API-spending approval.
