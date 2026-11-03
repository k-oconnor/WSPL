# Skillforge research contract

Kevin authorized a new repository and specification for LLM-programmed neural/RMV action blocks composed by LTLf programs. Skillforge is a user-accepted working name. Coordinate this work with sibling `thesis_master` and the ambition of accumulating compact, executable, experimentally validated knowledge toward a causal neuro-symbolic world model.

Read README.md, docs/CONTEXT.md, docs/SYSTEM_SPEC.md, docs/M1_ENGINEERING.md, docs/EXPERIMENT_01.md and docs/EVIDENCE.md before implementation or scientific claims. Implemented: reference semantics, small-alphabet LTLf DFA compilation and strong planning/execution in supplied finite models. LLM harness, skill learners and robot execution remain planned; do not describe them as implemented.

Read docs/SIM_ADAPTER.md for the implemented single-tick LIBERO bridge and sampled physical observers. Its five-tick headless integration check is engineering evidence only. Physical task/primitive-hierarchy execution and skill acquisition remain planned. Do not turn tools/sim_preflight.py into an unapproved scientific runner. Missing motor ticks abort observer invocations; no collision or admitted grasp observer exists yet.

November 3: differentiable MLP/RMV blocks and an approach-only positioning/imitation runner are implemented/prepared; no scientific run is approved or launched. Read docs/APPROACH_V1.md. Its analytic weights encode a known P controller and are unblinded engineering constructions, not a sealed LLM trial. The 12 cm standoff task is not a validated grasp pose. Keep the full four-arm acquisition objective, and distinguish this first positioning screen from physical pickup/hierarchy results.

Current November 3 launch status supersedes that preparation note: approach-v1 was explicitly user-approved and ran, then failed on native horizon accounting before any fitting. One reference positioning trial succeeded. See docs/APPROACH_V1_LAUNCH.md. The horizon-fix replacement is prepared but unapproved; do not reuse the original digest approval. Run the isolated simulator interpreter outside this checkout to satisfy the unchanged source gate. All 100 unit checks pass; no process is active and no learned physical result exists yet.

## Semantics and evidence

- Action blocks execute controls. LTLf propositions report observed state, commands and externally verified skill outcomes. Do not conflate a neural success prediction with physical success.
- Boolean LTLf on nonempty finite traces and fuzzy RMV arithmetic are distinct. Every bridge needs explicit, versioned thresholds/observer semantics. The direct evaluator is independent of the derivative compiler. A currently accepting DFA state need not remain accepting after further observations.
- LTLf automata monitor specifications. Strategy synthesis additionally needs action models, environmental assumptions and realizability analysis. State guarantees conditionally; a successful abstract synthesis is not a physical guarantee.
- Ground RMV inputs in named channels. Fidelity covers normalization, literals, conjunction/disjunction, affine motor readout and clipping. Report explanation size and empirical usefulness separately from exact arithmetic.
- Keep the two-by-two acquisition experiment: LLM/gradient × MLP/RMV. Count calls, tokens, failed proposals, feedback, data collection, revisions, compilation and validation. Interaction efficiency and wall-time efficiency are separate claims.
- Preserve held-out physical layouts, disturbances and task structures; seal choices before final evaluation. Do not use hidden evaluator state in a controller. Supplied structured perception and low-level control adapters must be explicit and matched.
- This project does not replace legacy RBL-RMV, TNST/TNTST or robot Model 1. No canonical architecture substitution is authorized by importing a prior artifact.
- Record negative results. Toy-world modularity/repair results assume supplied wiring, privileged intermediate inputs and known repair localization; monolith repair lacked rehearsal.
- Finite planning is strong reachability under every declared outcome, with no fairness assumption. Construction limits are errors, not evidence of infeasibility. Declared model/clock/proposition roles must match; never infer physical robustness from a bounded symbolic retry fixture.
- Read docs/HIERARCHICAL_TASKS.md before interpreting hierarchy results. Current expansion/summary lowering uses supplied acyclic templates and flat macro-level planning; no primitive hierarchical executor or learned hierarchy discovery is implemented. Fixture split labels are design axes, not sealed held-out scientific data. Primitive-reference counts are not measured physical execution costs.

## Source approval

Every new training or scientific benchmark run must use experiments/source_consistency.py with an exact source-manifest SHA-256 approved by the user and separately registered in experiments/source_pins/approved.json. Unit tests and read-only audits are exempt.

Prepare a concrete constructor, runner, configuration and complete Python inventory before requesting approval. Do not self-approve, transfer old approvals, weaken the gate, or alter an approved historical manifest. The checked runner must instantiate the supplied constructor and save source identity in checkpoints. Source identity does not establish fairness or correctness.

While a source-pinned run is active, do not edit/add/remove inventoried Python in that checkout. Use a separate worktree. Preserve one rolling resumable checkpoint per arm every 1,800 seconds plus final/clean-stop; no accumulating best/initial/per-improvement weight copies unless Kevin changes that policy.

The registry was initially empty; it now records Kevin's exact original approach-v1 approval and its failed run. The horizon-fix replacement is not approved or queued. Creating this repository/spec does not authorize API expenditure, new training, robot deployment or hardware control. Prepare work concretely before obtaining any approval explicitly required by this contract. Do not request approval for routine reversible development or unit checks.

## Repo hygiene

Keep data, credentials, model binaries, simulator outputs and machine-specific bindings out of Git. Use artifacts/local_paths.json for local external paths. Do not commit legacy checkpoints or bulk datasets. Do not push or create a public remote unless requested. Keep source provenance for any copied code. No project license has yet been chosen.
