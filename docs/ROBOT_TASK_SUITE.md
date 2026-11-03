# Diverse robot task suite v1

Prepared November 3, 2026 at Kevin's request for multiple grasps, chained actions,
obstacles and broader robot behavior. The suite contains **16 inspected task
specifications**, a strict catalog validator, local BDDL asset audit and labelled
trace checks. It does not execute robot tasks, construct new physical scenes,
admit skills or launch training. No scientific run is approved by adding it.

The previous positioning pilot remains the only completed physical capability
screen. In particular, its 12 cm standoff target is not a contact-ready grasp.
All 16 tasks currently report physical_runnable=false with explicit missing skill,
sensor, observer and executor capabilities. Existing scene assets are a starting
point, not evidence the full task can be executed by Skillforge today.

## Task ladder

| Tier | Task ID | Requested behavior | Declared object pickup occurrences | Scene work |
| --- | --- | --- | ---: | --- |
| 1 | single_box | Pick a box and place it on the bowl | 1 | Existing scene |
| 1 | single_bowl | Pick a bowl and place it on a plate | 1 | Existing scene |
| 2 | load_two | Load two objects into one basket, either order | 2 | Existing scene |
| 2 | sort_two | Route two mugs to different plates | 2 | Existing scene |
| 2 | ordered_load | Finish A before picking up B | 2 | Temporal goal overlay |
| 3 | round_trip | Same object: bowl -> plate -> bowl | 3 | Temporal goal overlay |
| 3 | swap_via_buffer | Swap two source locations using a staging surface | 3 | Custom initial/goal integration |
| 3 | drawer_cycle | Open drawer -> place bowl -> retreat -> close drawer | 1 | Goal/initial-state overlay, articulated control |
| 3 | microwave_load_close | Put mug inside an open microwave, retreat, close door | 1 | Existing scene, articulated control |
| 3 | stack_in_tray | Put the bottom bowl in the tray, then stack the top bowl | 2 | Temporal/support goal overlay |
| 3 | carry_around | Carry an object around a tall physical barrier | 1 | Custom barrier, side passage and route gate |
| 3 | carry_over | Lift/carry over a low barrier with whole-body clearance | 1 | Custom barrier and over-route gate |
| 3 | clear_then_fetch | Move a blocker, then retrieve the initially inaccessible target | 2 | Custom causal obstruction |
| 3 | hold_then_place | Pick up, maintain grasp for >=1 second, then place | 1 | Timed goal overlay |
| 4 | recover_drop | Observe a scheduled slip, reacquire and finish | 2 | Custom one-shot perturbation |
| 4 | recover_second_item | Recover B after a slip while preserving A's completed placement | 3 | Custom perturbation/progress integration |

Counts refer to specified pickup-event occurrences, not a measured minimum number
of grasps or simulator calls. Handle contacts during drawer/door manipulation are
additional interactions, not included as object pickup events. A real executor
must report every actual attempt, retry, primitive call and preparation cost.
Task tier is a proposed curriculum, not an empirically measured difficulty rank.

Five entries reuse existing scenes without scene changes; five specify goal or
initial-condition overlays; six need custom scene/perturbation integration. Native
BDDL goals often encode only final conjunctions. Our extra temporal, contact and
clearance contracts can be stronger. These are Skillforge specifications built on
LIBERO assets, not a claim to reproduce official benchmark scoring.

## Physical grounding and temporal contracts

Each task declares concrete object/fixture/region bindings, required skill and
capability IDs, distinct outcome occurrences, strict precedence edges, final-state
requirements, always-required safety predicates and optional consecutive dwell.
All proposals are in [robot_task_suite_v1.json](../examples/robot_task_suite_v1.json).

The local read-only audit resolves every native binding against seven BDDL assets
in LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`. It records original
native goal expressions and file hashes in
[robot_task_asset_audit.json](robot_task_asset_audit.json). It does not infer that
a language instruction, BDDL conjunction and our temporal contract are equivalent.
Names for unbuilt barrier/gate entities are explicitly marked planned and never
reported as present native assets. The external checkout was not modified.

Use independently measured evidence for:

- Pickup: calibrated contact, object elevation, support removal and grasp state;
  opening/closing the fingers alone is not pickup.
- Placement: supported/contained geometry, object stability, release and required
  retreat; touching a target or passing near it is not placement.
- Drawer/door: named joint state and handle transition with gripper/object
  clearance; final closed state alone does not prove an open/place/close sequence.
- Stacking: actual stable object-on-object support, including final retention.
- Route safety: robot, gripper and held-object collision geometry, allowed-contact
  policy and swept clearance at physics substeps. Point end-effector clearance
  and a route gate alone are insufficient.
- Recovery: visible physical fault consequence, invalidation of stale held state,
  verified new pickup and preservation of previously completed goals.

The generic labelled-trace checker verifies only the temporal/state contract.
It cannot certify the truth of labels or create a missing observer. Future
observers must bind object IDs and fresh invocation IDs, report unknown sensors,
and derive pulses independently of a controller's requested phase. Additional
attempts remain in raw logs even when they are not required task milestones.

Strict precedence rejects an early wrong-order event even if a later event could
provide an ordered witness. Each named occurrence must pulse exactly once; a
sticky pickup flag cannot satisfy three pickups. Final target states must still
hold at the terminal observation, so an earlier delivery followed by undoing it
does not pass. Recovery of B additionally requires preserving A after its delivery.

For hold_then_place, twenty consecutive held-state samples must occur strictly
between pickup and placement at consecutive 0.05-second timestamps. Missing ticks,
non-Boolean/missing observations and duplicate timestamps raise an unresolved
trace error. A single wait_done label cannot replace the physical held-state
history. The checker trusts timestamp provenance; it does not establish that
fabricated timestamps correspond to actual physics.

These structured constraints are **not compiled robot LTLf programs**. No physical
hierarchy refinement or planner is bound to them. The existing small-alphabet DFA
compiler has explicit proposition/state limits, and long task monitors may need
phase composition or an explicitly validated larger backend. Do not silently
turn a construction limit into an infeasible robot-task result.

## Obstacles that test something specific

carry_around needs a validated side passage and a ceiling/route constraint that
prevents an overhead shortcut. carry_over needs a low barrier and constraints
that make its overhead route meaningful. Both require clear start/goal regions,
reachable paths and clearance for the complete carried shape. Barriers must be
actual MuJoCo collision geometry; decorative clutter does not establish obstacle
avoidance. Proposed geometries are not yet constructed or physically calibrated.

clear_then_fetch is the most direct causal prerequisite task: the root contains
unordered delivery goals. It deliberately has no supplied placed_blocker ->
picked_target precedence edge. The scene must make target pickup genuinely
inaccessible until the blocker moves, across all allowed grasps/routes. A matched
cleared scene and direct/bypass controls must verify necessity. The current
catalog does not prove this condition, discover it, or run the earlier abstract
blocker model as if it were physics.

For each obstacle family include feasible-clear, obstructed-but-solvable and
unreachable/blocked controls after scene construction. Validate reachability
before treating an unsuccessful policy as evidence of planner/controller failure.
Unreachable variants need declared geometry/workspace assumptions; a failed
reference policy alone is not an infeasibility proof.

## Required interface work

The successful motor12 interface observes one bound object, one target, velocity,
aperture and contacts. Multi-object task execution needs explicit rebinding and
global progress accounting. Articulation needs handle/joint states; obstacle route
selection needs obstacle geometry and object dimensions; general grasps may need
orientation control. Those capabilities are marked planned in the manifest.

Do not hide a navigation, grasp or place policy in the four-output adapter. Decide
whether route choice belongs to a matched planner or a learned controller, then
declare its permitted scene input and count its cost. Adding scene/orientation
channels requires a new versioned model interface and a new experiment identity;
the current 1,588/1,604-parameter blocks are not automatically assumed sufficient.

## Evaluation design and first physical integration slice

Start with single_box and single_bowl, then load_two and ordered_load using the
same frozen pickup/put-down library. Integrate clear_then_fetch and carry_around
next to test geometry and causal prerequisites; then articulation, timed holding
and recovery. This gives short and long compositions with identifiable failures
rather than one large opaque end-to-end success bit.

Cross task family with object shape/size, source/target positions, ordering,
number of grasps, path clearance, friction and fault location. Vary one factor
first for diagnosis, then test combinations. Hold out layout/object families and
composition structures, not only random seeds. The current catalog and synthetic
traces are inspected design artifacts; they are not sealed held-out benchmarks.

Compare gold task programs with learned/frozen skills, authored versus gradient
weights, matched flat versus hierarchical execution, and reference versus zero
controls. Keep skills, observations, physical starts, model information and
interaction budgets matched where appropriate. Measure physical goals, observed
trace satisfaction, forbidden contacts, dropped objects, extra grasps, retries,
repeated completed work, motor/preparation calls, time and fidelity. Report task
length and hierarchy depth separately. No new trial is authorized by this design.

Before each family: verify nontrivial starts, no-op difficulty, reference
feasibility, correct observer counterexamples, actual simulator horizon including
preparation, and machine-readable readiness. Prepare and approve the complete
runner/source identity only when that concrete family is physically integrated.

## Engineering checks

Twelve new tests exercise all sixteen positive label fixtures and negative cases
for wrong order, coincident events, missing/repeated pickup, undone goals,
forbidden contact, insufficient/broken dwell, missing samples, motor budgets,
bad bindings, cyclic precedence/dwell and unsafe/missing asset paths. The BDDL
inspector checks typed grouped objects, qualified regions and original goals.
Synthetic labels are supplied test data, not measured robot outcomes.

Use `skillforge.robot_tasks.RobotTaskSuite` for schema/semantic validation,
`readiness(task_id)` for blockers, `check_trace(task_id, trace)` for closed labelled
traces and `audit_assets(suite, checkout)` for read-only scene binding inspection.
None imports or steps MuJoCo, trains a model or calls an LLM API.
