# Working context

Kevin requested this new repo/spec on November 2, 2026 and accepted Skillforge as a working name to revisit later. The research idea is an LLM authoring small neural/RMV action implementations and temporal task programs, with validation, persistent reuse, frozen composition and local repair.

The task is broader than the preceding Gravemind toy-world program tests. Those were fixed compositions of Boolean state-transition predictors; they did not control a robot, synthesize a plan or evaluate blinded LLM weight generation. Their summaries/plot are copied with provenance, not relabelled as this project's results.

Current implemented surface: typed task/skill/RMV/finite-model schemas, independent closed-trace LTLf semantics, derivative DFA compiler, strong finite-model strategy synthesis and abstract execution, scalar RMV evaluation and source gate. Fifty-eight engineering unit tests pass. No LLM client, robot adapter, differentiable skill model or robotic experiment has been implemented/run. See M1_ENGINEERING.md for algorithm assumptions/limits.

The first motor experiment is LLM versus gradient acquisition crossed with MLP versus grounded RMV. AdamW is acceptable; Kevin used SGD colloquially, not as an optimizer requirement. Compare acquisition cost, physical reliability, fidelity, frozen composition and repair; include rehearsal for global repair. Every skill must expose physically checked status propositions, not self-declared success. Fixed positive/negative literal rails are proposed for RMV to avoid shared-polarity cancellation and allow independent clause orientations.

Kevin selected MuJoCo -> robosuite -> LIBERO as the simulation stack: physics, robot/control/environment infrastructure, then task/data/transfer evaluation. Versions, task assets and API/controller compatibility still need preflight and pinning. No installation or simulator run follows automatically from this documentation decision.

M1 engineering fixture: the synthesized strategy handles one approach timeout and one grasp failure, then observes a grasp, lifts, and stops on an accepted finite trace. Infinite adversarial retry/fatal failure models are rejected. The compiler is checked against direct semantics on fixed and seeded mixed short-trace cases; the planner checks every declared outcome and a strictly decreasing winning rank. This is a symbolic supplied-model result, not a learned robot capability.

Next engineering milestone: prepare the selected simulator/observer adapter and differentiable skill implementations, with explicit physical preflight. See ROADMAP.md. No experiment approval or API expenditure follows automatically. The run registry is empty. This repo is local Git only; remote publication and licensing have not been selected.
