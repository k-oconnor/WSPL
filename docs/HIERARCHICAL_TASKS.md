# Hierarchical tabletop tasks

Status: seven checked abstract engineering fixtures and a proposed physical benchmark. No new skill training, LLM call or robot execution has been performed. The suite tests reuse of supplied parameterized hierarchies; discovery of new hierarchies is a separate question.

## Reusable library and levels

| Level | Reusable computation |
| --- | --- |
| L0: motor blocks | approach, close, lift, transport, lower, release, retreat, return_home |
| L1: subroutines | pickup(object) = approach -> close -> lift; put_down(object,target) = transport -> lower -> release -> retreat |
| L2: transfer | transfer(object,target) = pickup(object) -> put_down(object,target) |
| L3: group routine | load_pair(first,second,target) = transfer(first,target) -> transfer(second,target) |
| Task root | Compose the registered routines, add final-state goals and constraints, or leave independent subgoals unordered |

The same pickup/put_down/transfer definitions are referenced repeatedly with different bindings. Primitive network weights must eventually be shared across those calls, rather than relearned per object, target or task. The metadata counts named template calls, nominal primitive references and maximum named-template depth. Task-root operators are not counted as another named-template level. Repeated use and deeper nesting are different axes; round_trip tests reuse/length, while load_two also adds the load_pair nesting level.

The first eight neural motor blocks remain planned. These fixtures are task/compiler infrastructure for their eventual evaluation.

## Cases

| Case | Task | What it tests | Nominal primitive references |
| --- | --- | --- | ---: |
| transfer_one | Put a on the tray | Development check of one transfer | 7 |
| transfer_then_park | Transfer a, then return home | Development composition and terminal home state | 8 |
| load_two | Load a, then b, then return home | New root structure, parameter reuse and a deeper group routine | 15 |
| clear_then_retrieve | Put blocker a in the buffer and target b on the tray | Causal prerequisite: b is inaccessible until a is moved; root goals are unordered | 14 |
| round_trip | Move a to tray, buffer, tray again; return home | Three distinct executions of the same transfer/pickup templates | 22 |
| second_object_recovery | Load a and b, with one possible grasp failure on b | Resume the remaining task without unnecessarily redoing a | 15 before failures |
| blocked_forever | Clear an immovable a that blocks b | Negative control: report unrealizability under the supplied model | Not executable |

Each case also requires the stated final object locations and, when requested, the robot at home. Merely witnessing an earlier placement does not permit moving the object away and claiming success. The recovery case is a held-out-condition counterpart of load_two, not a new task-structure family. A failure may consume only part of a pickup, so static primitive-reference counts are not measured execution costs.

The intended development/structure/condition labels are recorded in examples/hierarchical_tasks.json. All current fixtures have been inspected and debugged; these labels are design axes, not a sealed scientific train/test split. A later LLM/physical benchmark needs independently sealed tasks, layouts and disturbances. These conceptual task names do not claim to match existing LIBERO task IDs; asset/task mapping is a later adapter decision.

## What is executable now

`Hierarchy` checks and expands acyclic named templates with explicit bindings. Cycles, unknown templates, missing bindings and expansion-limit violations are rejected. The full expanded tree remains inspectable. Summary-bearing L1 routines provide macro outcome propositions, e.g. picked_a and placed_a_tray. Lowering produces ordered finite-trace outcome witnesses; root-level all permits independent branches to interleave. Unordered branches nested inside sequences are explicitly unsupported rather than given an incorrect ordering interpretation.

The lowerer includes exact final-state conditions at the final valuation, plus collision avoidance. It does not prohibit every unrelated intermediate action; additional restrictions must be explicit. Independent observers must ultimately produce the real subtask outcomes, including any required retreat/termination, rather than allowing a neural policy to mint summary events.

The independent finite tabletop model tracks two object locations, one occupied hand, home state, most recent event and an explicit remaining grasp-failure budget. Its dynamics do not depend on the requested hierarchy or desired plan. A blocker at its source prevents pickup of b. Available actions are pickup/place/home macro calls with supplied effects. Source, buffer and tray have sufficient modeled capacity. There is no collision physics; collision is always false in this fixture, so its avoidance constraint is not being physically stress-tested.

The existing planner works over these macro summaries and computes contingent action choices. It infers the blocker ordering from applicability/effects when the root only specifies unordered goals. Repeating an outcome witness requires a later trace position, and event propositions arise only from actual modeled macro outcomes. The bounded failure counter makes retry assumptions explicit; unbounded adversarial failure is not presumed recoverable.

**This is not yet a hierarchical primitive executor or evidence that hierarchical planning is faster.** Lowering expands the provided task structure and the existing planner solves a flat world/DFA product at the summary cut. It trusts macro contracts instead of executing the approach/close/lift/etc. bodies. It does not establish refinement correctness, within-skill temporal safety, learned decomposition or physical task success.

## Run an engineering fixture

After the README installation step:

```sh
python -m skillforge hierarchy examples/hierarchical_tasks.json --case clear_then_retrieve --execute --output work/clear_then_retrieve.json
python -m skillforge hierarchy examples/hierarchical_tasks.json --case second_object_recovery --execute --outcome-choice last --output work/second_object_recovery.json
python -m skillforge hierarchy examples/hierarchical_tasks.json --case round_trip --execute --output work/round_trip.json
python -m skillforge hierarchy examples/hierarchical_tasks.json --case blocked_forever
```

The last command returns the ordinary unrealizable exit code 3. A construction limit remains an error, not unrealizability. Outputs include the supplied hierarchy, reuse statistics, template-library identity, actual synthesized macro policy and optional execution trace. Existing output paths are not overwritten. These commands reproduce deterministic unit fixtures and do not run a scientific benchmark or simulator.

## Physical benchmark and controls

First acquire/validate the L0 blocks and L1 subroutine contracts. Freeze their weights, observers, parameter interfaces and library identities before assessing new roots. Evaluate source layouts versus new layouts, repeated bindings, task length/depth, causal dependencies and disturbances separately. Then compare:

1. Gold hierarchy/specification with frozen reference or learned skills, isolating motor composition.
2. LLM-generated hierarchy/specification using the same fixed registry, measuring correct task translation and selection.
3. Flat expanded programs versus reusable hierarchical execution with identical goals, skills, model information and physical starts.
4. Supplied hierarchy versus learned/discovered decomposition, without quietly granting different primitive or intermediate supervision.
5. Local repair versus global repair with matched rehearsal and explicit localization information.

Primary outcomes: physical goal success, observed trace satisfaction, constraint violations, controller timeouts, skill-call counts, simulator steps, LLM cost, planning cost, hierarchy depth/reuse and unchanged-task retention. Report generation of a new root separately from invention/acquisition of a new skill or subroutine. After a fault in b, count whether the system unnecessarily repeats the already-completed a transfer. Report every failed attempt and abort.

Completion of this task-definition work is engineering evidence only. A real hierarchy advantage requires execution/refinement and the matched flat control, not simply a longer LTLf sequence. All scientific runs still require their own concrete source approval.
