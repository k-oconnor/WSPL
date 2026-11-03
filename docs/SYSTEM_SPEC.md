# System specification

Status: design version 0.1, November 2, 2026. Only the JSON schemas, closed-trace LTLf reference evaluator, scalar RMV reference and source gate are implemented. Other modules below are proposed contracts for implementation.

## Objective

Acquire small executable skills from task specifications and experience; validate them independently; retain them in a persistent library; compose them for new finite-horizon tasks; and repair a changed mechanism locally. Test acquisition speed, physical reliability, exact computational fidelity and composition separately. This is a proposed component of Kevin's causal neuro-symbolic world-model program, not a completed world model or an established priority claim.

## Architecture and ownership

| Layer | Input / output | Responsibility |
| --- | --- | --- |
| Task author | Natural-language instruction -> typed LTLf AST + bindings | Propose intended temporal relationships; cannot invent undeclared skills/propositions |
| Task checker/compiler | AST + vocabulary -> validated specification -> DFA | Check syntax, bindings, finite-trace semantics and consistency |
| Planner/synthesizer | DFA + skill contracts + abstract environment model -> strategy or infeasible result | Account for controllable commands, observed outcomes, failures and assumptions |
| Executor | Strategy state + observations -> skill invocation/STOP | Enforce applicability, mutual exclusion, timeouts and library version identity |
| Skill | Named local observations -> bounded motor command | Execute closed-loop control; weights may be LLM-authored or gradient-learned |
| Outcome observer | Physical observations -> Boolean state/events + unknown status | Verify applicability and outcomes independently of a controller's claims |
| Registry | Candidate + evidence -> versioned admitted artifact | Preserve contracts, assumptions, provenance, cost and observed failure scope |

The LLM proposes both task programs and weight artifacts; it does not bypass the checked runtime. A controller may predict success for diagnostics, but only the observer supplies success propositions. Actions are executable policies; predicates/propositions describe their applicability, invocation and measured results.

## Skill interface

A skill has an immutable ID/version, named input channels and units, normalization ranges, output order/scales, applicability formula, declared assumptions, outcome/failure observers, timeout, candidate-weight identity and validation scope. Its reset and memory behavior are explicit. The first controller family is stateless; any later recurrent state must be named, bounded, reset on invocation and included in exported semantics and comparison budgets.

Proposed tabletop observation vector, fixed in order before any run:

1. Object relative to end effector: dx, dy, dz in meters.
2. Bound target relative to end effector: dx, dy, dz in meters.
3. End-effector velocity: vx, vy, vz in meters/second.
4. Gripper aperture in meters.
5. Left/right finger contact indicators in [0,1].

These are twelve structured channels, not learned visual perception. Goal/object bindings are supplied consistently to every arm. Simulator availability and exact definitions of velocity, aperture and contact must be audited before freezing an adapter; substitute channels require a new interface version, not an undocumented proxy.

Output is four normalized commands: dx, dy, dz, gripper. A shared, declared low-level adapter scales xyz increments, holds orientation and maps the gripper command into the simulator. It may use the simulator's ordinary servo/IK controller; it must not run an approach/grasp/place routine on behalf of the neural block. Frequency, clipping, workspace limits and hold/abort behavior are matched infrastructure and included in execution logs. Proposed initial control rate is 20 Hz, subject to simulator preflight and source pinning.

Robot adapter first target: a fresh, isolated MuJoCo/Robosuite tabletop setup, drawing on prior LIBERO integration for installation knowledge. No dependency or simulator version is selected/pinned yet. Do not import legacy task-completing primitives as if their behavior were newly learned by a skill.

## Logical interface and clock

Examples of observed propositions: holding, aligned, at_target, object_supported, pickup_done, place_done, contact_lost, collision. Invocation propositions such as invoke_release are controllable commands. Each symbol has a declared producer, physical meaning, thresholds, debounce/stability rule, scope and version. Completion events are one-tick pulses scoped to a skill invocation, rather than permanently true flags that can accidentally satisfy later tasks.

The first specification clock is a motor tick. At tick t, the observer labels the current physical state and outcomes of earlier commands. The executor chooses the current invocation/command; the combined observed and controllable valuation is checked, then the simulator advances. The terminal valuation records the final observation with no new skill invocation. Exactly one terminal valuation is appended; STOP does not manufacture an extra successful action. A later skill-boundary clock is a different interface version and must not silently drop constraints inside a skill.

Unknown/missing sensor values are not false. The observer reports an explicit unresolved status and the executor pauses/aborts under a declared policy. The Boolean reference evaluator rejects missing/non-Boolean propositions. Runtime monitor design must handle incomplete knowledge explicitly before claiming physical guarantees.

Example finite task:

`F(grasp_done & X F(place_done)) & G(invoke_release -> at_target) & G(!collision)`

This requires a grasp completion with a strictly later placement completion, release commands only at the target, and no collision over the completed trace. It does not itself define motor actions, prove applicability, or forbid every other unrequested event. Additional task restrictions belong in the formula or skill contracts, not in an undocumented interpreter.

Strong next X is false at the final position; weak next Xw is true there. U is strong until and requires a witness within the trace. The reference API uses nonempty, closed traces. A false value of F(goal) on an unfinished prefix is not a definitive failure. Online acceptance/violation and realizability require a proper automaton implementation; the reference evaluator does not provide them.

## Compilation and planning

1. Validate the generated AST against the declared proposition vocabulary and skill/object bindings.
2. Check the intended formula on positive and negative development traces, including terminal and ordering counterexamples. Parsing success is not translation correctness.
3. Compile the restricted LTLf fragment to a DFA with a sealed backend/version; cross-check acceptance against the direct reference semantics.
4. Combine the DFA with an abstract skill-effect model. Separate agent-controlled choices from environment outcomes. Include failed actions, timeouts, loss of grasp and missing observations.
5. Synthesize or plan under explicit assumptions; return infeasible when the goal cannot be supported. Compilation to a DFA alone is monitoring, not action synthesis.
6. Execute while updating the automaton from independently observed outcomes. Replan under the same declared model when allowed; log retries, aborts and LLM involvement.

Do not assume every retry eventually succeeds unless an environment assumption justifies it. Abstract guarantees are conditional on observer correctness, model assumptions and actual skill contracts. Physical reliability must be measured separately. A task translator, planner and skill learner have distinct errors; oracle-spec/oracle-skill controls isolate them.

## Grounded RMV action block

Every predicate begins from one named input channel with a fixed, declared range. Let p_i = clip((x_i - lo_i)/(hi_i - lo_i), 0, 1). One proposed weighted-DNF implementation is:

```
l_(i,+) = p_i; l_(i,-) = 1-p_i
c_j = clip(1 - sum_l W[j,l] * (1-l_l), 0, 1)
h_k = clip(sum_j V[k,j] * c_j, 0, 1)
u   = clip(A @ h + b, command_lower, command_upper)
```

W and V are nonnegative. Each clause can independently use either or both rails of each predicate; there is no shared learned polarity setting. A may be signed. Normalization, literal orientation, all logical operations, affine motor readout and final clipping are part of the exported computation. The motor readout is not an unexplained neural bypass. Any shared servo acts after this normalized command and is separately declared infrastructure.

Proposed capacity: twelve fixed calibrated inputs, 24 fixed positive/negative literal rails, 56 clauses, four concept heads and four motor outputs. This has 1,588 parameters: 1,344 AND weights, 224 OR weights, 16 readout weights and four biases. A 12-to-32-to-32-to-4 two-hidden-layer ReLU MLP has 1,604 parameters. Counts are within 1%, without padding. These are design capacities, not implemented/trained models. Report effective clause/literal use, active computation, optimizer state and inference latency separately.

The old local RieszFFN starts lambda at 0.5 with identical clauses, yielding constant initial outputs and zero input gradients. The proposed fixed dual rails avoid that shared-polarity cancellation, but do not by themselves solve clause saturation or symmetry. Before any new training pin, check asymmetric initialization, clause diversity, clipping occupancy, task gradients and actual input sensitivity. If surrogate gradients are used, label their bias and preserve the exact hard forward semantics. No surrogate choice is approved by this spec.

The exported graph is an exact computational explanation over named channels. It is not automatically a human-understandable rule set: 56 dense clauses can be opaque. Report clause/literal counts, coefficient precision, sensitivity and controller behavior. Any pruning/simplification has its own fidelity/performance evaluation, with selection confined to development data. Exact arithmetic does not establish causal or semantic correctness of a predicate.

## Candidate lifecycle

Specify -> propose weights -> validate shapes/ranges -> test input sensitivity -> check scalar/model fidelity -> measure skill behavior -> admit or retain rejected evidence -> freeze -> compose -> diagnose/repair -> revalidate affected and still-valid behavior.

A candidate includes weight-format/constructor identity, contract and predicate versions, graph/formula when applicable, assumptions, acquisition records, training/feedback data IDs, simulator version, all API usage, validation results and a cryptographic content hash. Generating arbitrary controller Python is a separate code-policy comparator; it does not count as a fixed-architecture weight-writing result.

Production skills remain unadmitted until the chosen physical reliability criteria and RMV-fidelity checks pass. Admission scope is empirical unless a genuine proof is supplied. Store rejected candidates and reasons; do not present a best-of-many candidate as a one-shot result. A new task can compose admitted skills without changing them; new weights create a new skill version and invalidate old evidence where relevant.

Local repair must identify how the affected skill was selected: supplied identity, trace diagnosis or learned localization. Compare known-location repair with an appropriate equally informed baseline, including rehearsal for global fine-tuning. Recheck the repaired skill, downstream compositions and unaffected tasks. Structural isolation is useful, but it is not evidence that the system discovered that factorization.
