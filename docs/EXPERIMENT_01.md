# First experiment: acquiring interpretable motor skills and composing pickup

Status: proposed protocol, not an approved or runnable scientific experiment. Concrete simulator bindings, learner/LLM configurations and exact source identity must be prepared and approved before execution. No API calls or robot training have been launched.

## Questions and staged scope

Primary: can an LLM write weights for a reliable fixed-architecture closed-loop skill with less interaction or total acquisition time than gradient learning? Secondary: can grounded RMV match useful reliability while giving a faithful explanation of the entire motor output? Then test frozen composition and local repair.

First motor screen: approach, close/grasp and lift. Define applicability and completion independently, using fresh structured observations. Approach reaches an actual grasp pose from a nontrivial displacement; close establishes stable contact without claiming pickup; lift raises the object and verifies maintained grasp. A pickup task succeeds only after independently observed object elevation and stable contact. Placement, transport, lowering, release and retreat extend the library after this core works.

Do not let close/grasp silently invoke a scripted lift. The existing legacy robot routines supply installation/history knowledge, not a free skill implementation. Every preparation action needed to start an isolated close or lift trial is shared and counted. Separate such reference-prepared skill trials from autonomous full pickup.

## Acquisition arms

| Weight acquisition | Ordinary MLP | Grounded RMV |
| --- | --- | --- |
| LLM-authored | Fixed 1,604-parameter architecture | Fixed 1,588-parameter architecture |
| Gradient-learned | Same MLP and interface | Same RMV and interface |

Include zero/hold control and a transparent handcrafted reference controller for feasibility and task difficulty. Their interactions and implementation knowledge are reported. A direct LLM-generated Python controller may be a later comparator, but it is not part of this weight-writing test.

The writer sees architecture, tensor layout, units, task contract, actuator semantics and permitted observations. It may emit numeric tensors or a transparent tensor-construction script; the executed robot controller must remain the fixed network, not that script. No hidden optimizer, extra state, simulator access or teacher API is allowed outside the logged budget. Separate analytic, unblinded development constructions from candidates produced by a fresh writer session on sealed skill specifications.

The gradient arm starts from fresh weights with audited nondegenerate initialization. AdamW is acceptable; exact rates, update limits, clipping, loss and any RMV regularization must be selected on development data and frozen in the run manifest. Use the same architecture for both acquisition methods within a representation. General model pretraining gives the writer substantial prior knowledge; do not claim equal prior compute or information.

## Shared information and costs

Begin with supervised imitation as the gradient baseline: a declared reference teacher collects command-labelled trajectories on source layouts. The same permitted trajectories, goals, returns, violations and observations are available to the writer. Record which items it actually reads. A teacher-trained imitation baseline is not a comparison against all gradient/RL algorithms; later reward-only/RL tests need separate controls.

Proposed initial screen: one writer/training seed, three skills, four acquisition arms. Nested source budgets 0, 4, 16 and 64 demonstration/feedback episodes per skill. Maximum cumulative writer proposals at those budgets: 1, 2, 4 and 8. Feedback episodes, failed trials and candidate revisions are charged, including attempts that yield no accepted skill. These numbers are design targets to finalize before pinning; a single-seed screen supports feasibility only. Confirmation uses at least three independent paired acquisition seeds after a frozen protocol.

Pure zero-shot means no physical feedback before the proposed weights are sealed. Evaluation still costs interactions; keep that evaluation cost separate from information available to the writer. If a writer revises after seeing a validation result, that validation interaction becomes acquisition feedback, and the candidate needs a fresh untouched evaluation set.

Report all of:

- Unique simulator transitions/episodes used for teacher collection, preparation, proposals, gradient fitting, physical validation and candidate selection.
- LLM model/version, calls, input/output tokens, tool runs, latency, failures and measured API cost.
- Gradient updates, rows/timesteps presented, accelerator time, memory and initialization trials.
- Data preparation, prompt construction, weight conversion, compilation, fidelity checking and physical evaluation wall time.
- Human-provided mechanics, formula templates, interface design and reference controllers, separating development from the sealed writer evaluation.

Shared teacher episodes can be collected once but must be charged as available supervision to every arm. Shared acquisition data does not imply identical optimization effort or prior knowledge. Compare wall time and sample efficiency separately. No efficiency win may be claimed by omitting writer-side work or validator interactions.

## Development and final evaluation

Partition physical initial conditions by layout/pose families, not adjacent timesteps. Source/development/final pools are disjoint. Source includes both successes and failures. Development sets thresholds, permitted initialization and all hyperparameters. Final held-out trials include translations, grasp offsets, object placement/size or friction changes that the sealed interface can represent. Specify which changes alter a mechanism versus simply an initial condition.

Suggested screen evaluation: 16 development trials per budget/candidate and 32 fresh final trials per selected skill implementation, paired across arms. These are inadequate for a strong high-reliability claim; report denominators and binomial intervals. A later admission/confirmation panel should use at least 100 fresh trials per relevant skill/condition and a predeclared reliability criterion. All final tests are logged, including invalid starts and aborted trials, under a fixed exclusion rule.

Before final testing, calibrate task difficulty using development-only checks: ordinary zero/hold commands should fail nontrivial approach/lift trials, while the transparent reference demonstrates feasibility. Do not change final thresholds to rescue performance. If a success tolerance is so broad that no action succeeds, redesign on development and version the task rather than claiming skill acquisition.

Primary outcomes: independently measured skill success, constraint violations, completion steps, failures/timeouts, worst-condition reliability and interactions/time to a predeclared reliability target. Initial small screens report curves rather than claiming the target is statistically established. RMV also reports full-output numerical fidelity, formula complexity and grounded input sensitivity. A rule graph passing arithmetic tests is not a successful robot controller.

## RMV verification

Check exact architecture and normalization metadata, finite weights, W/V nonnegativity, literal grounding and orientation and output bounds. Compare the differentiable runtime with the independent scalar export on Boolean corners, random interior points and activation/normalization boundaries. Proposed FP32 maximum motor-output discrepancy tolerance is 2e-6 on the sealed finite probe set; this is an empirical numerical check, not a universal floating-point proof. State whether any formal construction argument additionally applies.

Every motor output includes the affine readout and clipping. Report natural-scale command differences after the shared adapter as well. Record saturation, dead/input-insensitive paths and clause diversity before training. Report dense versus simplified clause/literal counts and any performance cost of simplification. Keep evaluation of human usefulness distinct from fidelity.

## Frozen temporal composition

After skills are acquired/admitted, hash and freeze their implementations and observers. Use gold LTLf specifications first to isolate motor composition. Then evaluate LLM-generated specifications with a fixed compiler/planner. Hold out task structures/orderings and perturbation conditions; distinguish new language paraphrases from genuinely new compositions.

Concrete task families and shared hierarchy definitions are in HIERARCHICAL_TASKS.md and examples/hierarchical_tasks.json. They are inspected engineering fixtures; build and seal fresh physical/LLM evaluation instances rather than reporting their existing unit outcomes as held-out performance. Hierarchical primitive execution must be implemented and checked against the macro contracts before physical hierarchy claims.

The first composed task is pickup: approach, establish contact, then lift while maintaining it. Later tasks reuse the enlarged library for pick/place, return and constrained ordering. Include ordering, avoidance and failure/retry requests, plus infeasible requests. Ground truth is actual trace satisfaction and physical completion, not the language model's explanation or a planner's abstract plan.

Compare LLM versus gold specifications, candidate versus reference skills, and a fixed scripted order versus synthesized composition where appropriate. Verify compiled DFA acceptance against the closed-trace reference on exhaustive short traces for small alphabets and sampled/counterexample traces for larger tasks. Do not report monitoring as controller synthesis. Account for planner computation and failures to find a strategy.

## Repair and stopping decisions

Change one mechanism, such as grasp friction or gripper response, while retaining the observable interface. First supply the affected skill identity to isolate repair; diagnose it from traces only in a later experiment. Compare local repair/replacement with global fine-tuning both with and without a matched rehearsal buffer. Charge old and new examples, updated weights and all candidate selection. Verify frozen unaffected skills bitwise and retest unaffected tasks as well as the changed skill/compositions.

Advance only when implementation checks, task feasibility and measured skill behavior support the next phase. Lack of a valid skill, wrong task translation, infeasible planning and physical execution failure are different outcomes. Stop and diagnose an input-insensitive RMV initialization or a controller that cannot fit source behavior before spending a larger budget. Preserve all negative findings. No run follows automatically from this document.
