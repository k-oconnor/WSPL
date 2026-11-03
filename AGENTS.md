# Skillforge research contract

Kevin authorized a new repository and specification for LLM-programmed neural/RMV action blocks composed by LTLf programs. Skillforge is a user-accepted working name. Coordinate this work with sibling `thesis_master` and the ambition of accumulating compact, executable, experimentally validated knowledge toward a causal neuro-symbolic world model.

Read README.md, docs/CONTEXT.md, docs/SYSTEM_SPEC.md, docs/M1_ENGINEERING.md, docs/EXPERIMENT_01.md, docs/EVIDENCE.md and docs/APPROACH_V1_RESULTS.md before implementation or scientific claims. Implemented: reference semantics, finite-model planning/hierarchy fixtures, differentiable MLP/RMV blocks, motor adapter/observers and a completed pregrasp positioning/imitation screen. A blinded LLM harness, contact skills and physical primitive hierarchy execution remain planned.

Read docs/SIM_ADAPTER.md for the implemented single-tick LIBERO bridge and sampled physical observers. Its five-tick headless integration check is engineering evidence only. Physical task/primitive-hierarchy execution and skill acquisition remain planned. Do not turn tools/sim_preflight.py into an unapproved scientific runner. Missing motor ticks abort observer invocations; no collision or admitted grasp observer exists yet.

November 3: differentiable MLP/RMV blocks and an approach-only positioning/imitation runner are implemented/prepared; no scientific run is approved or launched. Read docs/APPROACH_V1.md. Its analytic weights encode a known P controller and are unblinded engineering constructions, not a sealed LLM trial. The 12 cm standoff task is not a validated grasp pose. Keep the full four-arm acquisition objective, and distinguish this first positioning screen from physical pickup/hierarchy results.

Current November 3 launch status supersedes that preparation note: approach-v1 was explicitly user-approved and ran, then failed on native horizon accounting before any fitting. One reference positioning trial succeeded. See docs/APPROACH_V1_LAUNCH.md. The horizon-fix replacement is prepared but unapproved; do not reuse the original digest approval. Run the isolated simulator interpreter outside this checkout to satisfy the unchanged source gate. All 100 unit checks pass; no process is active and no learned physical result exists yet.

Current final status (November 3): user-approved approach-v1-fidelityfix completed all 60 episodes and both 2,000-update fits. Evaluation: zero 0/8; reference, analytic MLP and analytic RMV each 8/8; learned MLP 5/8 and learned RMV 3/8. RMV visited-input scalar fidelity <=2.99e-8. Source gate complete, external environment unchanged, no job active and no follow-on run approved. This supersedes historical preparation/failed-attempt notes. Read docs/APPROACH_V1_RESULTS.md for limitations: inspected analytic teacher encodings and same-scene standoff positioning, not blinded authorship, grasp, hierarchy or transfer evidence.

## Semantics and evidence

- Action blocks execute controls. LTLf propositions report observed state, commands and externally verified skill outcomes. Do not conflate a neural success prediction with physical success.
- Boolean LTLf on nonempty finite traces and fuzzy RMV arithmetic are distinct. Every bridge needs explicit, versioned thresholds/observer semantics. The direct evaluator is independent of the derivative compiler. A currently accepting DFA state need not remain accepting after further observations.
- LTLf automata monitor specifications. Strategy synthesis additionally needs action models, environmental assumptions and realizability analysis. State guarantees conditionally; a successful abstract synthesis is not a physical guarantee.
- Ground RMV inputs in named channels. Fidelity covers normalization, literals, conjunction/disjunction, affine motor readout and clipping. Report explanation size and empirical usefulness separately from exact arithmetic.
- Keep the two-by-two acquisition experiment: LLM/gradient Ãƒâ€” MLP/RMV. Count calls, tokens, failed proposals, feedback, data collection, revisions, compilation and validation. Interaction efficiency and wall-time efficiency are separate claims.
- Preserve held-out physical layouts, disturbances and task structures; seal choices before final evaluation. Do not use hidden evaluator state in a controller. Supplied structured perception and low-level control adapters must be explicit and matched.
- This project does not replace legacy RBL-RMV, TNST/TNTST or robot Model 1. No canonical architecture substitution is authorized by importing a prior artifact.
- Record negative results. Toy-world modularity/repair results assume supplied wiring, privileged intermediate inputs and known repair localization; monolith repair lacked rehearsal.
- Finite planning is strong reachability under every declared outcome, with no fairness assumption. Construction limits are errors, not evidence of infeasibility. Declared model/clock/proposition roles must match; never infer physical robustness from a bounded symbolic retry fixture.
- Read docs/HIERARCHICAL_TASKS.md before interpreting hierarchy results. Current expansion/summary lowering uses supplied acyclic templates and flat macro-level planning; no primitive hierarchical executor or learned hierarchy discovery is implemented. Fixture split labels are design axes, not sealed held-out scientific data. Primitive-reference counts are not measured physical execution costs.

## Source approval

Every new training or scientific benchmark run must use experiments/source_consistency.py with an exact source-manifest SHA-256 approved by the user and separately registered in experiments/source_pins/approved.json. Unit tests and read-only audits are exempt.

Prepare a concrete constructor, runner, configuration and complete Python inventory before requesting approval. Do not self-approve, transfer old approvals, weaken the gate, or alter an approved historical manifest. The checked runner must instantiate the supplied constructor and save source identity in checkpoints. Source identity does not establish fairness or correctness.

While a source-pinned run is active, do not edit/add/remove inventoried Python in that checkout. Use a separate worktree. Preserve one rolling resumable checkpoint per arm every 1,800 seconds plus final/clean-stop; no accumulating best/initial/per-improvement weight copies unless Kevin changes that policy.

The registry records three explicit user-approved pilot identities: two failed attempts and the completed fidelity repair. No follow-on experiment is approved or queued. Creating this repository/spec does not authorize API expenditure, new training, robot deployment or hardware control. Prepare work concretely before obtaining any approval explicitly required by this contract. Do not request approval for routine reversible development or unit checks.

## Repo hygiene

Keep data, credentials, model binaries, simulator outputs and machine-specific bindings out of Git. Use artifacts/local_paths.json for local external paths. Do not commit legacy checkpoints or bulk datasets. Do not push or create a public remote unless requested. Keep source provenance for any copied code. No project license has yet been chosen.


November 3 horizon-fix update: user approved75e33ff7d7d55a1500cfd9521a95f26dc1443f5a6354c3cde595749b87f6931b; registered and launched PID18736. All4 reference development trials succeeded, all4 zero controls timed out correctly. Collected4 teacher trajectories/154rows; MLP and RMV each completed2000AdamW updates (final training MSE0.000213251/0.003921668). First evaluation seed1001: zero timeout, reference+analyticMLP+analyticRMV success43calls/~5.918mm. Run then failed on NumPy aperture scalar rejected by strict scalar reference; no learned-model physical evaluation or remaining7seeds ran. Environment unchanged, artifacts/checkpoint saved; no active process. Read-only audit additionally found FP32/scalar error up3.48e-6 (2.19e-6 with matching input ports), above2e-6 tolerance. Prepared explicit repair canonicalizingFP32 checker inputs and usingFP64 RMV arithmetic withFP32 parameters/optimizer, unchanged architecture/budgets/tolerance. Frozen saved-input audit now <=2.99e-8 with no new simulator/training calls;102unit checks pass. Candidate d770313a57ec51094e3147e6876c975835e5ccd55950e057d7bde2c3e247aaa4 unapproved, not queued. See Skillforge/docs/APPROACH_V1_FIDELITY_REPAIR.md. Evaluation seed1001 is now partly inspected; this is an engineering panel, not sealed transfer evidence.
