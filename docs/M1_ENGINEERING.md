# M1: checked finite task compilation and execution

Implemented on November 2, 2026. This milestone is a symbolic engineering foundation, not an LLM/robotics performance study. No training, LLM API call or simulator execution was used. The scientific approval registry remains empty; all executions here are deterministic unit fixtures.

## Compiler

`skillforge.automata.compile_dfa` accepts every operator in the current AST: constants, propositions, not, and/or, implication, strong/weak next, eventual, global and strong until. It enumerates letters over the formula's referenced propositions and constructs a deterministic residual automaton. Reduced ordered BDDs canonicalize Boolean combinations over a fixed formula closure. This implementation is original project code; progression/Boolean canonicalization are established techniques, not a novelty claim. Formal background: [LTLf synthesis](https://ijcai.org/proceedings/2015/223) and [progression-based synthesis](https://www.ijcai.org/proceedings/2022/0359.pdf).

An internal nonempty-suffix marker preserves terminal semantics. After consuming X(phi), the residual is phi AND nonempty; weak next uses phi OR empty. Without that distinction, X(true) could incorrectly accept a one-position trace. An initial seen-observation flag excludes empty traces even for tautologies. Acceptance after a prefix means it is valid to terminate now, not that all possible extensions are valid.

The compiled DFA stores complete transition rows, accepting states and states from which some logical continuation can reach acceptance. Logical completability quantifies over arbitrary proposition letters; it is not physical realizability. The independent direct evaluator remains the numerical/semantic test oracle.

Default construction caps: eight referenced propositions, 512 DFA states and 50,000 BDD nodes. These are engineering limits on an explicit small-alphabet implementation. A cap raises CompilationLimit rather than returning unsatisfiable/unrealizable. The compiler is not intended as a scalable replacement for mature general-purpose tooling without further evaluation.

## Supplied finite skill model

`FiniteModel` snapshots state observations, available skill calls and all permitted next-state outcomes. Observed and controllable propositions must be disjoint; every state declares the same Boolean observation vocabulary. An unavailable call is not an option. Unknown targets and malformed outcomes are rejected. The model, clock and textual assumptions are content-hashed; execution rejects a changed model. Task/model clock and proposition roles must agree.

At product state (world, automaton), the monitor has consumed all prior valuations. A call emits current observations plus its invocation propositions, then the environment chooses a declared outcome. STOP emits exactly one final observed valuation with all invocation propositions false. It is permitted only when that final valuation makes the DFA accepting. There is no empty trace or fake completion action.

The supplied model is symbolic infrastructure. This implementation does not infer skill effects, validate physics or convert a skill's claimed success into true observations.

## Strong strategy synthesis and execution

`synthesize` explores reachable world/DFA product states. It computes the least reachability fixed point: terminal accepting nodes have rank zero; a predecessor wins when some available call sends **every** permitted outcome to a previously winning node. The selected call minimizes the largest successor rank, with deterministic action-name tie breaking. Winning rank strictly decreases regardless of the environment's declared choice.

An action with possible success and an unbounded self-loop is not enough: an adversary could choose failure forever. No fairness or probabilistic eventual-success assumption is inserted. Explicit bounded-failure protocol states can make recovery realizable. An unavoidable permanent failure can make it unrealizable. This is strong reachability in a supplied finite model, not unrestricted LTLf synthesis over an arbitrary environment.

The default product cap is 10,000 states. PlanningLimit means construction was incomplete; it is not an infeasibility proof. The abstract executor checks model identity, available calls, observed-outcome membership and decreasing rank. It distinguishes accepted execution, action-budget exhaustion, unrealizability and contract violation. Unexpected outcomes raise a contract violation rather than being projected to a convenient symbolic state.

## CLI and fixture

After the README installation step:

```sh
python -m skillforge compile examples/pick_place.task.json --output work/pick_place.dfa.json
python -m skillforge plan examples/discrete_pickup.task.json examples/discrete_pickup.model.json --initial start --execute --outcome-choice last --output work/discrete_pickup.plan.json
```

The pickup fixture has six reachable product states and six winning states. Its last-outcome execution is:

`approach -> retry_approach -> grasp -> retry_grasp -> lift -> STOP`

It reaches raised, and its observed trace satisfies the ordered grasp/pickup formula, the lift-while-holding guard and collision avoidance. The retry success is a supplied bounded-failure assumption, not measured robot robustness. The planner computes the contingent choices; it does not receive that action sequence as a plan.

CLI exit codes: 0 success, 1 invalid input/output collision, 2 construction limit, 3 unrealizable supplied model, 4 contract violation, 5 execution budget exhausted. Output artifacts are created exclusively and never overwrite a prior file. A no-execution plan command is just symbolic strategy construction.

## Validation and remaining work

The M1 validation passed 58 unit tests; the later hierarchy-summary extension brings the full suite to 69 (see HIERARCHICAL_TASKS.md). Compiler checks compare the DFA with independent direct semantics on exhaustive short traces for 16 fixed formulas, plus 32 seeded mixed formulas and their short traces. Cases include nested strong/weak next, negation, until, ordering and terminal effects. Other tests cover complete deterministic transitions, snapshots, independent schema validation, both failure branches, impossible constraints, no fairness assumption, rank decrease, STOP timing, changed contracts, clocks and distinct error codes. This is bounded test evidence, not a machine-checked universal proof.

Next: physical observation/controller preflight for MuJoCo/robosuite/LIBERO, differentiable MLP/RMV skills and independent physical observers. A future translator/robotics benchmark still requires a concrete runner and approved source identity. No new scientific run is queued.
