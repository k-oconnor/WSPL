# Roadmap and evidence milestones

## WSPL language and cross-domain studies

The user endorsed a typed weight-space programming language and a formal
authoring/repair/composition program across three domains. See
wspl/EXPERIMENT_PROGRAM.md and wspl/WSPL_IMPLEMENTATION.md. The scalar JSON IR and
exact rational reference implement an initial part of G0; 122 total engineering
unit checks pass. Compiler, universal verifier, learned import/patch transactions
and blinded six-arm proposer harness are still missing. The proposed first screen
is 36 nonconfirmatory synthesis runs, explicitly unapproved and not launchable;
model, costs, task split and final thresholds remain unset. These language gates
do not replace the physical robotics milestones below.

## M0: reference interfaces and reproducibility

Completed engineering scaffold: schemas/examples, closed finite-trace semantics, scalar RMV arithmetic, source gate, provenance and experiment design. Unit checks pass and the initial repo is committed. No physical capability is claimed.

## M1: task compilation before robot control

Implement restricted LTLf-to-DFA compilation and direct-semantics cross-checking. Define controllable versus observable propositions and a finite skill-effect model. Demonstrate a synthesized strategy on a small discrete world, including failure/timeout and infeasible requests. The old toy-world fixed sequence is not this result. Seal a task-translation development/final split before evaluating an LLM translator.

Exit evidence: acceptance equivalence on a declared finite trace suite; negative terminal/ordering cases; actual strategy execution under the abstract model; infeasible requests correctly separated from compilation errors.

Current status: implemented and checked as engineering fixtures. The compiler/planner and hierarchy-summary extension pass 69 total unit tests, including semantics agreement, bounded timeout/failure recovery, unbounded-retry rejection and construction-limit separation. See M1_ENGINEERING.md and HIERARCHICAL_TASKS.md. Broad task-translation benchmarks and LLM-generated tasks have not been evaluated.

## M2: grounded motor skills

Implement a fresh adapter for the selected MuJoCo/robosuite/LIBERO stack, independent outcome observers and differentiable MLP/RMV blocks with exported semantics. Pin compatible versions and task/controller settings after preflight. Check physical channels, no-op difficulty, reference feasibility, controller input sensitivity and RMV fidelity. Prepare the first experiment's exact source identity before new training or benchmark execution.

Current status: adapter, sampled observers and differentiable MLP/RMV blocks implemented. Headless integration, gradient and scalar-export checks pass. The approved pregrasp positioning pilot completed: analytic MLP/RMV 8/8 each, learned MLP 5/8, learned RMV 3/8 (APPROACH_V1_RESULTS.md). Contact-ready grasp calibration, collision observation and the blinded acquisition study remain pending. This does not meet M2 exit criteria.

Exit evidence: reliable isolated approach, close and lift on held-out physical conditions, acquisition cost curves for all four arms and a faithful complete RMV motor output. A successful symbolic interpreter over old supplied motors does not satisfy this milestone.

## M3: frozen temporal composition

Compose the new skills for pickup, then add transport/lower/release/retreat and test pick/place. Begin with gold task formulas, then test the LLM translator. Keep skill weights and observer definitions frozen when assessing new task structures. Explicitly distinguish monitoring, planning, physical control and task-language correctness.

Exit evidence: new temporal compositions succeed physically and satisfy their actual observed trace formulas; failures are attributed; learned skills are not silently retrained per task.

Task families and supplied reusable decompositions are now defined in HIERARCHICAL_TASKS.md. Abstract macro-summary fixtures are checked; primitive hierarchical execution, physical contracts and a matched flat/hierarchical comparison remain unimplemented.

ROBOT_TASK_SUITE.md expands the physical design to sixteen tasks over seven audited local LIBERO scenes. Start integration with single transfers, two-object loading and strict order, then add blocker/route geometry, articulation, dwell and recovery. All sixteen are specifications with labelled trace checks; no physical hierarchy capability follows from catalog validation.

## M4: local repair and stronger comparisons

Replace/adapt one changed skill and retest old/new tasks. Add global rehearsal and information-matched repair controls before claiming robust repair efficiency. Evaluate diagnosis of the affected skill separately from supplied localization. Consider wider physics changes and additional skills only after this works.

Exit evidence: repair has a measured acquisition-cost/retention benefit under stated assumptions, with verified unchanged artifacts and newly validated compositions.

## M5: research synthesis

Assess whether generated executable knowledge is retained, transferred and revised. Connect mechanism models to TNST/temporal causal work only when identification assumptions and actual intervention predictions are tested. Scope novelty against LTLf synthesis, language-conditioned skill planning, code policies, weight generation and interpretable control. Publication positioning follows evidence; the full causal neuro-symbolic world-model ambition remains broader.
