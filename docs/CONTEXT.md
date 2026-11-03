# Working context

Kevin requested this new repo/spec on November 2, 2026 and accepted Skillforge as a working name to revisit later. The research idea is an LLM authoring small neural/RMV action implementations and temporal task programs, with validation, persistent reuse, frozen composition and local repair.

The task is broader than the preceding Gravemind toy-world program tests. Those were fixed compositions of Boolean state-transition predictors; they did not control a robot, synthesize a plan or evaluate blinded LLM weight generation. Their summaries/plot are copied with provenance, not relabelled as this project's results.

Current implemented surface: typed task/skill/RMV schemas and examples, direct closed finite-trace LTLf semantics, scalar complete RMV evaluation and a source-identity gate. Thirty-five engineering unit tests pass. No LLM client, DFA compiler, strategy synthesizer, robot adapter, differentiable skill model or robotic experiment has been implemented/run.

The first motor experiment is LLM versus gradient acquisition crossed with MLP versus grounded RMV. AdamW is acceptable; Kevin used SGD colloquially, not as an optimizer requirement. Compare acquisition cost, physical reliability, fidelity, frozen composition and repair; include rehearsal for global repair. Every skill must expose physically checked status propositions, not self-declared success. Fixed positive/negative literal rails are proposed for RMV to avoid shared-polarity cancellation and allow independent clause orientations.

Kevin selected MuJoCo -> robosuite -> LIBERO as the simulation stack: physics, robot/control/environment infrastructure, then task/data/transfer evaluation. Versions, task assets and API/controller compatibility still need preflight and pinning. No installation or simulator run follows automatically from this documentation decision.

Next engineering milestone: implement and cross-check the restricted LTLf compiler and contract-based planning on the discrete world, then prepare a fresh physically grounded simulator/skill implementation. See ROADMAP.md. No experiment approval or API expenditure follows automatically from repository creation. The run registry is empty. This repo is local Git only; remote publication and licensing have not been selected.
