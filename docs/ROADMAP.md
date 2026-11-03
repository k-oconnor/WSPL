# Roadmap and evidence milestones

## M0: reference interfaces and reproducibility

Current deliverables: schemas/examples, closed finite-trace semantics, scalar RMV arithmetic, source gate, provenance and experiment design. Complete when unit checks pass, examples validate, implementation status is explicit and the new Git repo has a clean initial commit. No physical capability is claimed.

## M1: task compilation before robot control

Implement restricted LTLf-to-DFA compilation and direct-semantics cross-checking. Define controllable versus observable propositions and a finite skill-effect model. Demonstrate a synthesized strategy on a small discrete world, including failure/timeout and infeasible requests. The old toy-world fixed sequence is not this result. Seal a task-translation development/final split before evaluating an LLM translator.

Exit evidence: acceptance equivalence on a declared finite trace suite; negative terminal/ordering cases; actual strategy execution under the abstract model; infeasible requests correctly separated from compilation errors.

## M2: grounded motor skills

Implement a fresh adapter for the selected MuJoCo/robosuite/LIBERO stack, independent outcome observers and differentiable MLP/RMV blocks with exported semantics. Pin compatible versions and task/controller settings after preflight. Check physical channels, no-op difficulty, reference feasibility, controller input sensitivity and RMV fidelity. Prepare the first experiment's exact source identity before new training or benchmark execution.

Exit evidence: reliable isolated approach, close and lift on held-out physical conditions, acquisition cost curves for all four arms and a faithful complete RMV motor output. A successful symbolic interpreter over old supplied motors does not satisfy this milestone.

## M3: frozen temporal composition

Compose the new skills for pickup, then add transport/lower/release/retreat and test pick/place. Begin with gold task formulas, then test the LLM translator. Keep skill weights and observer definitions frozen when assessing new task structures. Explicitly distinguish monitoring, planning, physical control and task-language correctness.

Exit evidence: new temporal compositions succeed physically and satisfy their actual observed trace formulas; failures are attributed; learned skills are not silently retrained per task.

## M4: local repair and stronger comparisons

Replace/adapt one changed skill and retest old/new tasks. Add global rehearsal and information-matched repair controls before claiming robust repair efficiency. Evaluate diagnosis of the affected skill separately from supplied localization. Consider wider physics changes and additional skills only after this works.

Exit evidence: repair has a measured acquisition-cost/retention benefit under stated assumptions, with verified unchanged artifacts and newly validated compositions.

## M5: research synthesis

Assess whether generated executable knowledge is retained, transferred and revised. Connect mechanism models to TNST/temporal causal work only when identification assumptions and actual intervention predictions are tested. Scope novelty against LTLf synthesis, language-conditioned skill planning, code policies, weight generation and interpretable control. Publication positioning follows evidence; the full causal neuro-symbolic world-model ambition remains broader.
