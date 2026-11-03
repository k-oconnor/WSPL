# Controller change review and repair validation

November 3, 2026. Kevin and the collaborator can use a narrow first workflow to
test the startup hypothesis: a controls engineer changes a small neural controller,
checks its required behavior, diagnoses a regression and reviews the correction.
The first deliverable is a reproducible change review with artifact identities,
counterexamples and explicit evidence scope. Customer demand and savings remain
hypotheses to test.

## First user and problem

Recruit robotics or controls teams that already maintain small policies with
structured sensor inputs and bounded outputs. Prefer a team with a recent failed
change, a simulator or recorded regression cases, and an engineer responsible for
review. The engineer is the proposed user; the engineering lead is a possible
buyer. Neither role nor budget has been validated.

Start with bounded policies or residual corrections whose requirements can be
written down. The question to ask is: "Show us the last controller change that
was difficult to validate. What evidence did the reviewer need, and how long did
that take?" A retrospective example is more useful than general enthusiasm.

The proposed value is shorter time from a behavior request to a reviewable,
qualified change. WSPL may support a fresh compiled controller, a future local
patch, or empirical investigation of an opaque policy. These are different
capabilities. The current prototype compiles fresh controllers, imports supported
frozen MLPs and composes checked output patches. Customer-model repair has not
been evaluated.

## Existing alternatives and the central risk

Neural verification already has capable tools. [Marabou](https://github.com/NeuralNetworkVerification/Marabou)
accepts neural networks and property queries and returns guarantees or
counterexamples. [alpha beta CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
supports multiple network architectures and general verification properties.
These official project descriptions were checked November 3. They are potential
backends and comparison points, not evidence of demand for WSPL.

The strongest alternative for a small explicit controller may be ordinary code
with tests, review and a verifier. A neural representation needs a concrete
advantage: integration into an existing neural runtime, useful learned behavior
that must be retained, or reusable modules that genuinely reduce engineering work.
If code wins the customer's actual workflow, do not force the task into weights.

Our commercial hypothesis concerns the complete authoring, change and evidence
workflow. Our initial Z3 verifier is a small prototype, not a claimed advantage
over existing verification systems. The language experiment must distinguish
benefits from compilation, verifier feedback and ordinary tooling integration.

## Runnable engineering demonstration

From the repository root, install the README development and verification extras.
The optional opaque path additionally needs the tested Torch/NumPy environment
(Torch 2.5.1 CPU and NumPy 1.26.4); the basic demo only needs Z3.

```sh
python tools/demo_controller_review.py --output work/controller_review
python tools/demo_controller_review.py --output work/controller_review_with_probes --include-opaque
```

Choose a new directory for each invocation. The demo creates its parent, refuses
to overwrite an existing directory and preserves partial files on errors.

| Step | What the collaborator should inspect |
| --- | --- |
| Original controller | Compiled BrakePatch graph and three non-vacuous real-graph contracts |
| Fault injection | Explicitly reduce the supplied braking gain from 2 to 1 |
| Review failure | REFUTED contract and exact rational counterexample, replayed on stored weights |
| Operator correction | Restore the original supplied program, compile and recheck all contracts |
| Opaque example | Separate Tanh fixture with activation-effect and no-effect records plus exact replay |
| Handoff | Sources, weights, bound contracts, verification results, review.json and file-hash inventory |

The original, broken and corrected statuses should be PROVED, REFUTED and PROVED.
Missing/exhausted solver support produces an incomplete outcome, not success.
CLI exit codes are 0 for expected fixture outcomes, 2 for incomplete/unexpected
outcomes and 1 for input/dependency/output errors. The unsigned manifest checks
file identity; it is not a research source approval or an authenticity signature.

This inspected fixture demonstrates the review loop. The correction is supplied
by the operator; no LLM discovers it and no gradient learner trains. The final
weights equal the original weights. Mathematical contracts cover the rational-real
I/O graph, not floating-point deployment, source equivalence or physical safety.
Activation effects establish computational dependence only. Neither path measures
customer savings or qualifies a controller for deployment.

## Customer discovery and decision rules

For the CSAIL-inspired benchmark progression, use the
[product win conditions](wspl/CSAIL_COMPARISON.md): useful performance, required
checks and a predeclared workflow benefit. Higher return than CSAIL is optional.
The initial proposed targets are at least 90% of matched baseline game return
and half the engineering/review time, subject to absolute usefulness and
customer-required constraints. Benchmark success and paid customer commitment
are separate milestones.

Kevin and the collaborator should conduct five initial interviews, using a recent
change as the unit of discussion. No outreach is authorized or sent by this plan.
Record the current toolchain, controller size/operators/state, actual failure,
reviewer, engineering hours, simulator/data availability, required evidence and
who could approve a paid pilot. Capture stated estimates as estimates; request
existing logs or issue history where sharing is permitted.

Use these proposed internal gates to control effort, not as estimates of market
size or statistical proof:

| Gate | Evidence to continue | Reason to reconsider |
| --- | --- | --- |
| Problem | At least three of five describe a concrete recent costly change | Interest is primarily novelty or no recurring review problem |
| Access | Two teams can supply permitted artifacts and a testable requirement | Required data/interfaces are unavailable or unsupported |
| Commitment | One team commits engineering time and discusses a paid pilot with a budget owner | Nobody will allocate time or identify a buyer |
| Benefit | A paired pilot meets the predeclared quality bar and a target of at least 2x less active engineering time | Ordinary code/current tooling is as fast or more reliable |

These thresholds may change after discovery and before any pilot. Keep the
original reasons and version the revised decision. Never tune the customer
success criterion after observing the comparative result.

## Matched pilot design

First establish the workflow on fresh synthetic tasks through the formal
[experimental program](wspl/EXPERIMENT_PROGRAM.md). Its six-arm representation
by verifier-feedback comparison remains the scientific design. This product plan
does not replace it or authorize its 36 proposed runs.

Then select one partner workflow with explicit permission. Freeze the change
request, reference behavior, preserved regions/tasks, information available to
each arm, budget, success conditions and final evaluator before authoring. Start
with simulation or offline review. An imported artifact outside our compiler or
stateless adapter needs an explicit supported import/reset path first.

Compare WSPL with the partner's normal workflow, an equally assisted ordinary-code
workflow and a direct-weight construction/edit workflow. Keep the same model,
feedback, task information and end-to-end budgets for controlled language
contrasts. If retaining learned behavior is essential, add a gradient repair arm
with matched supervision and rehearsal. An unassisted partner baseline measures
workflow improvement; it cannot by itself establish a language-specific benefit.

Qualification requires all predeclared supported graph contracts and empirical
task/preservation thresholds; report both components separately. Unsupported
requirements, UNKNOWN, failed candidates and incomplete runs stay in denominators.
Opaque-network probes are diagnostic evidence, not replacements for required
contracts. A safety reviewer defines acceptance for the intended use.

Measure active engineering time and elapsed time separately, including contract
authoring, integration, diagnosis, failed candidates, review and recovery. Count
API calls/tokens, money, demonstrations, simulator interactions, solver time,
training time, unchanged-task regressions and reviewer corrections. Report setup
cost separately and amortize it only over an explicit number of changes.

The proposed 2x time target is a business hurdle, not an observed result. The
quality bar must be met first. Customer anecdotes do not replace independent
synthesis replicates or the publication's held-out evaluation.

## Collaborator handoff and next build

1. Run the demo and inspect a counterexample directly on the broken artifact.
   Review assumptions and proof scope before accepting the output.
2. Bring one real change request and the team's current review procedure. Use
   the discovery questions above; keep identities/data private unless permitted.
3. Kevin and the collaborator choose a supported first task and a reviewer.
   Confirm whether the pain is authoring, regression diagnosis or evidence review.
4. Build the missing isolated proposer/evaluator and cost ledger. Keep generated
   candidate text away from evaluator code, final tasks and hidden diagnostics.
5. Prepare a runnable study with pinned sources, model identity, cost cap and
   frozen evaluation. Obtain the exact source approval required by AGENTS.md
   before scientific execution or paid model calls.

Use controller change review as the first validation target while preserving the
broader research on neural program authoring, local repair and composition.
