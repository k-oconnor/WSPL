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

Schema validation and reference-semantics unit checks are engineering evidence only. There is no Skillforge physical run, admitted robot skill, learned task translator, DFA compiler or demonstrated robot synthesis yet. Keep this distinction current as implementation advances.
