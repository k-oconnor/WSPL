# Prior evidence and what it supports

## Completed finite-world work

The two antecedent experiments live in external Gravemind worktrees. Small JSON summaries are copied byte-for-byte under provenance/toy_world; their original paths/hashes are in provenance/index.json. No weights, datasets or active source pins are imported as Skillforge approvals.

First pilot: supplied Boolean programs for a three-room, two-door/switch, one-object world. Analytic authored networks and independently AdamW-trained blocks with exhaustive local supervision both achieved exact composition through 32 actions. Random blocks failed. Replacing one door mechanism restored exact changed-world predictions; a stale mechanism failed. This verified implementation/interfaces. It was not blinded LLM synthesis, autonomous planning or robotic control.

Second pilot: limited-data independent blocks versus parameter-matched monolithic and auxiliary-output monolithic AdamW learners, three paired seeds. Held-out next-state accuracy at 28 unique source transitions was 75.40%, 37.70% and 36.51%; at 252 examples, 100%, 100% and 99.60%. Every method fit observed next states. With 16 intervention examples, local modular repair retained 100% tested unchanged behavior and exact long rollouts; global fine-tuning damaged unchanged behavior.

These results support supplied factorization and constrained repair in this finite setting. Modular training had known wiring and gold intermediate inputs; local repair knew the affected block; monoliths had no rehearsal. The repaired door had only four possible Boolean inputs. Auxiliary target labels did not remove the full information/interface difference. They do not establish LLM authorship advantages, general causal discovery or intrinsic monolithic limits. A rehearsal control remains proposed.

## Earlier robotic work

Legacy source: sibling rbl_rmv_icu, research/ntst_colearn/MODEL_FRAMEWORKS.md and experiments/libero_control. Existing symbolic routines, retry guards and supplied motor primitives are separate models. Their physical experiments motivate explicit action contracts, preparation costs, independent task outcomes and failure handling. Skillforge must not relabel those supplied motors as newly generated neural skills.

Prior motor trials included cases where zero action already satisfied broad geometric tolerances. The new physical experiment must establish nontrivial development cases before final evaluation. Existing libraries and neural-motor hybrids have mixed outcomes; none constitutes a completed weight-authored RMV/LTLf system.

## RBL-RMV workstation report

Kevin reports a separate 145M RBL-RMV model trained on approximately 500M FineWeb-Edu tokens / 30,000 steps, about 90% BLiMP and increasingly coherent generations. This is user-reported and not independently verified locally. The local rbl_rmv_icu copy appears older, with no matching 145M/BLiMP run. Its RieszFFN has a degenerate initial literal setting (lambda=0.5 with identical clauses); a small CPU audit found constant initial outputs and zero input gradients, although polarity gradients exist.

Do not infer that the local old source is the successful workstation architecture, or transfer language-model competence into a robot-skill claim. Obtain and audit the functional changes before reusing that RMV implementation.

## Current Skillforge evidence

Ninety-nine engineering unit checks cover schemas, direct/reference and DFA trace semantics, neural and scalar RMV arithmetic/fidelity, input/parameter gradients, strong planning, hierarchy fixtures, motor/observer contracts and source gating. A supplied finite pickup fixture executes a synthesized timeout/failure recovery strategy. A separate fresh headless LIBERO integration check reads twelve channels over five motor ticks with zero setpoint increments; see SIM_ADAPTER.md. The unblinded approach-only physical/imitation screen in APPROACH_V1.md is prepared but unapproved/unrun. There is no scientific Skillforge physical result, admitted robot skill, LLM task translator or physical robot synthesis yet. Abstract guarantees depend on declared effect models and assumptions; the unit fixtures and simulator wiring check are not learned-skill results.


November 3 launch evidence: the original pilot was approved and attempted. One reference pregrasp positioning trial succeeded (25.3 cm initial distance, 6.4 mm final, 41 control calls); the paired zero control then hit a runner native-horizon bug. No fitting or full evaluation ran. The failed attempt and untrained checkpoint are preserved, and the minimal horizon repair awaits a new exact-source approval. All100 unit checks pass. See APPROACH_V1_LAUNCH.md.
