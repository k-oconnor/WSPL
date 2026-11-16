# Neural control DSL: first executable pass

November 16, 2026. Engineering prototype, not a new training or scientific result.
Kevin requested a neural DSL with chainable operator blocks and a robot action
space, synthesizing the controller author work with Gravemind composition.

## Implemented interface

```text
transfer a to tray
transfer b to buffer
move_to home
```

Object references and named planar points have separate namespaces. `move_to`
accepts either; `grasp` and `release` require an object. `transfer Object to Point`
expands into approach, grasp, transport and release. Expansion reuses the same
frozen neural weights. Invalid syntax, argument types and excessive expansion fail
before execution. This line grammar is the executive DSL; WSPL remains the language
for implementing neural operators.

The executable primitives are `move_to`, `grasp` and `release`. Each invocation
retains its tick count, returns running/success/failure and completes before the
next begins. The executive workspace contains the program counter, skill lifetime,
named targets, observations and a trace. Local skills are currently reactive;
arbitrary learned hidden memory and recursion are not implemented.

Six compiled ReLU blocks provide move, near, holding, close, open and safe functions.
The copied controller_author.py is byte-identical to the completed controller pilot
in WSPL-controller-author-v1; the demo uses supplied compiler reference programs,
not newly evaluated learned-author outputs. Frozen target-network artifacts execute
through the existing independent rational evaluator.

The two action spaces are:

- Executive: choose an operator and bind an object or target argument.
- Motor: bounded `(vx, vy, grip)` commands, issued repeatedly until termination.

The current interpreter executes explicit programs. It does not infer a task,
choose routes with a neural network, perform search or silently repair an invalid
program. A later Gravemind-style executive could choose the next typed instruction
from the current workspace. Distinguish that learned executive from the already
trained WSPL block author.

## Physical observation and failure semantics

This is a deterministic planar kinematic workcell with multiple objects, one held
object and geometric contact. There is no contact physics, perception, obstacle
collision model or real robot. Grasp attaches only a unique nearby, unjammed object;
the simulator does not receive the desired object name as a command.

Neural predicates propose completion. The executor independently checks actual
position and holding state; a false-positive predicate is a contract failure, not
task success. Skill and total tick budgets are finite. Range/nonfinite commands,
precondition errors, safety violations and timeouts have explicit failure reasons.
There is no automatic retry. Terminal ticks cannot move the robot or append traces.

The physical safety observer stops subsequent execution after a boundary violation;
this is detection, not a proof or a preventive safety shield. `success` means the
supplied instruction sequence completed. Arbitrary mission correctness requires an
external observer/LTLf specification. The unit fixture independently evaluates
always-safe and grasp-a-before-grasp-b on the recorded physical trace. It does not
claim universal verification of the controller.

Every tick records observations before/after, the active instruction, neural block
inputs/outputs, motor command, skill status and failure reason. The saved demo also
includes the program and exact compiled network artifacts.

## Reproduce the engineering fixtures

```powershell
$env:PYTHONPATH='src;tests'
python -m unittest test_control_dsl -v
python tools/demo_control_dsl.py --output work/control_dsl_success.json
python tools/demo_control_dsl.py --jam-second --output work/control_dsl_jammed.json
```

Tests cover two-object composition, macro/expanded execution identity, argument
types, failed preconditions, jammed grasp timeout, terminal idempotence, total and
expansion budgets, false arrival predicates and actuator validity. The successful
fixture transfers both objects and returns home; the jammed fixture stops on the
second object without undoing the first transfer.

## Next experiment to prepare

Keep the block bank frozen and train only a small executive to emit this DSL from
scene/task specifications. Use successful supplied-planner traces as supervision.
Split by complete mission structure, not just object-name or coordinate variants.
Reserve longer chains, new orderings and controlled grasp failures. Compare with
the unchanged executive model, a flat primitive-program author and the supplied
symbolic planner. All use the same block library and observations.

Primary outcomes: independently observed final object locations, LTLf satisfaction,
invalid instructions, unsafe steps, failure handling and execution length. Reuse
the completed author artifacts only after declaring a separate frozen-library
integration evaluation. A later study can jointly evaluate author plus executive.

No new training, learned-model inference or scientific benchmark is launched by
this implementation. A concrete dataset, model budget, isolated source manifest
and exact user approval remain necessary before that experiment.
