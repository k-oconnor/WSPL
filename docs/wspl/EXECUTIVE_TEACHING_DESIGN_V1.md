# Teach the small executive, not a host-written selector

The implemented preparation is now specified in EXECUTIVE_TEACHING_V1.md.
This document preserves the earlier design; its remaining-work list is historical.

Research direction confirmed November 19, 2026. Design and interpreter groundwork;
NOT a launch-ready protocol or an approved scientific run. No new Astra calls,
student predictions, training or scored simulator missions have occurred.

## Primary question

Can Astra teach a weaker language model to use an acquired motor skill appropriately,
recover after an observed calibration mismatch, resume its mission, and retain old
behavior when Astra is absent? The primary comparison gives every executive the
same frozen motor modules and documentation. Extra tools alone do not establish
learning by the executive.

The completed factory pilot solved reaching through a host-owned probe, neural
selector and waypoint loop. Its 8/8 changed-regime result does not demonstrate
teaching Qwen. Its eight authoring responses contain engineering interactions,
not demonstrations of a small executive's deployment action interface. Do not
relabel those responses as an adequate executive teaching corpus.

## System ownership

| Component | Responsibility |
|---|---|
| Small Qwen executive | Decide whether/how to probe, select a controller, attempt a named goal, respond to the result, choose the next goal and stop |
| Frozen motor modules | Map position error to bounded robot commands |
| Host interpreter | Validate and execute exactly the selected operation; expose sensor feedback and enforce budgets |
| Astra teacher | Demonstrate executive decisions using the student's available observations during acquisition |
| Independent evaluator | Score physical waypoint achievement, order, completion, retention and interaction cost |

The host MUST NOT automatically probe, interpret displacement into a fault label,
invoke the saved neural selector, select/switch the controller, retry, or advance
the mission. Internal motor feedback and bounded skill execution remain normal
low-level control, not executive decision making. The small model must explicitly
select even the initial controller.

## Starting point and matched comparisons

Use the completed closed-loop Qwen2.5-Coder-1.5B rank16/alpha32 executive checkpoint,
seed113 chosen in advance, not the best outcome-selected seed. Restore its exact
LoRA weights. The base remains frozen. Reset optimizer identically for continuation
arms; do NOT call run_closed_loop.fit_seed, which resets the adapter from scratch.

All arms receive the same frozen original and factory-repaired movement networks,
module documentation, command API, context limit, decoding and observation fields:

1. **Unchanged executive + library:** no further gradient updates.
2. **Interface/rehearsal continuation + library:** learn the new command interface
   and ordinary waypoint operation; no adaptation demonstrations.
3. **Astra-taught executive + library:** same update budget, including verified
   Astra adaptation demonstrations and ordinary-task rehearsal.

Arm3 versus arm2 is the main comparison. Arm3 versus arm1 alone confounds adaptation
learning with additional training and a new action syntax. An observed-state reference
policy is an engineering feasibility check, not evidence of LLM learning.

Keep module weights and library documentation identical across all three arms.
Only the student LoRA changes. No automatic selector is available as a callable tool.
Opaque module/goal identifiers vary by mission; accompanying module documentation
is permuted consistently. Report whether the model follows the documented roles
instead of memorizing an ID or first-entry convention.

## Action interface: implemented first pass

Exactly one JSON action per decision:

```json
{"op":"probe","command":[0.1,0,0],"ticks":4}
{"op":"select","module":"tool_17"}
{"op":"reach","goal":"goal_8"}
{"op":"stop"}
```

Probe returns noisy observed displacement but never a diagnosis or controller ID.
Select changes only the active motor block. Reach executes a bounded ten-tick
chunk toward the named taught waypoint, with the selected motor network and open
gripper; further attempts require further executive decisions. Reaching a goal
does not choose the next one. Stop does not mark unfinished goals successful.

Observation contains the task-supplied waypoint order/coordinates, sensor-based
visit history, selected module, available IDs, latest action/outcome, robot telemetry
and remaining budgets. It excludes hidden dynamics, event schedules, oracle
success, suggested next action and selector scores. Public measured arrival and
independent physical success remain distinct.

Implemented in src/skillforge/student_control.py. Eight unit tests check ownership
of probing/selection/retry/goal advance/stop and finite decision budgets. This is
an interpreter test with synthetic sensors, not a student or robotics result.

## Teaching data: proposed bounded first collection

Collect new closed-loop Astra demonstrations, rather than generate labels from an
experimenter-written diagnostic policy. Initial target: eight two-waypoint missions,
balanced across ordinary calibration, initially changed calibration, a change after
the first goal, and a return to the old calibration. Up to16 responses/mission:
128 calls maximum. Final budgets must be source-pinned before collection.

Every teacher decision records the exact student-visible observation, one action,
prospective prediction separately, actual command/sensor trace and independent
mission result. A teacher failure is retained. Failed trajectories are not silently
converted into positive imitation labels. Require successful evidence for every
intended training condition before proceeding to student fitting; otherwise report
a teacher/acquisition failure without automatically expanding the budget.

Qwen2.5-Coder here is text-only. For this teaching comparison Astra must choose
demonstrated actions from the same text telemetry interface. The camera acquisition
tools remain available in the wider system, but teacher-only visual information
cannot be silently required for student decisions. A future vision student or
shared visual encoder requires its own interface and evidence.

Split by whole mission before labeling/training. Never mix states from the same
mission between acquisition and evaluation. Audit duplicate observations and keep
episode lineage. Repeated samples count as repeated exposures, not new examples.
Mechanical identifier substitutions may be evaluated separately; do not call them
new Astra demonstrations or introduce an expert-generated policy under that name.

## Proposed student continuation

Pilot one matched pair of continuation arms from the exact same saved executive.
128 updates each, microbatch1/accumulation8; completion-only action loss; AdamW
LR0.00003 with16-update warmup and cosine to10%, decay0.01, norm clip1. Rank and
base precision remain unchanged. These settings are proposed, not yet preflighted.

Arm2 draws1024 exposures from ordinary interface/rehearsal records. Arm3 draws
512 ordinary and512 successful Astra-action exposures, sampled reproducibly from
the actual collected rows. These are update/example-exposure matched, not necessarily
token or FLOP matched. Report unique records, repeats, action targets and wall time.
No gradients through physics or motor modules. Both final checkpoints retained;
no selection on final mission outcomes. Half-hour rolling saves plus final/clean-stop.

## Reserved deployment comparison

Proposed32 fresh missions, eight per condition: ordinary, changed initially,
unannounced switch between goals, return to original between goals. Fresh geometry,
noise seeds and identifiers; longer3-5waypoint chains versus two in teaching.
Same exact missions across three arms =96 executive episodes. All trained artifacts
freeze before any reserved outcome. Astra is absent for every deployment episode.

Independent scoring must check true site geometry, requested order, full completion
and physical action/decision budgets. Report individual corrections/regressions,
probe use, unnecessary controller switches, failure/recovery latency and ticks.
The experimenter may schedule disturbances; the interpreter must not reveal or
respond to them for the student. No claims of grasping, objects, contact recovery
or a new motor algorithm: this is learned executive use of supplied skill modules.

The desired evidence is better adaptation/recovery by arm3 than BOTH comparison
arms, while retaining ordinary mission performance. Numerical go thresholds must
be fixed before the final run, after engineering feasibility but before reserved
student predictions. A syntax-only gain, failure of all arms on ordinary missions,
or a benefit explained entirely by module availability does not pass the learning
claim. One student seed is exploratory; replicate only after a clear signal.

## Remaining preparation

- Bind the completed parent executive and factory motor artifacts by hash.
- Implement mission scheduling and independent physical scoring for the student API.
- Prepare teacher collection, provenance filtering and ordinary rehearsal data.
- Implement warm-start continuation without resetting the saved LoRA.
- Verify frozen base/modules, exact restoration, prompt lengths, adapter switching,
  matched data exposure, and no host-provided decisions.
- Prepare exact source pins and final budgets; obtain approval before calls,
  training or scientific evaluation. No launch is authorized by this design.
