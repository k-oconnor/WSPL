# Neural controller authoring and task composition

Completed studies, archived November 17, 2026.

## Architecture and task

A Qwen2.5-Coder-1.5B-Instruct author is adapted with rank16 LoRA to emit WSPL
modules from supplied numerical functional specifications. WSPL compiles the
modules into ReLU weights for move, near, holding, close, open and safe operators.
An independent rational artifact evaluator executes the neural computations.

A separate executive LoRA learns to generate task programs from structured scene
descriptions, placement goals and before constraints. `transfer` expands into four
stateful skills. Motor and predicate blocks are shared across objects and calls.
The executive's generation is completed before execution. The interpreter provides
termination, contract checks and bounded failure handling.

The environment is a planar kinematic simulator with geometric grasp contact and
one held object. Final geometry and observed delivery events determine mission
success. A program that terminates while skipping a goal or violating ordering
fails. The reported scope covers synthetic composition with supplied semantics,
planner supervision and state observation. Contact dynamics and perception require
additional evaluation.

## Block author pilot

1,536 training specifications, two epochs,384updates,96reserved block specifications.
All six functional families appear in every split with disjoint full parameter
specifications. The final author compiles96/96blocks, passes96/96numerical checks
and supports32/32composed missions. The untouched base compiles0/96, producing
0/32mission successes. The comparison is dominated by syntax validity.

Training reaches its final update at approximately14minutes. Total elapsed time
is70.7minutes, with3.44GiB peak allocated GPU memory. Teacher targets come from
a compiler. This establishes parameter generalization within supplied families.

[Protocol](CONTROLLER_AUTHOR_V1.md) ·
[Archived results](../../artifacts/neural_controllers/controller_author_v1/results.json)

## Initial executive pilot

4,096training missions with2–3objects, two epochs,1,024updates. All160trained
programs compile. Success is32/32on familiar3objects,25/32on4objects,19/32on5objects,
and9/32on renamed5objects. Safe failure on jammed4objects is21/32. The reference
passes all relevant checks. The five-object progression threshold was20/32 and
was missed by one mission. Total elapsed55.1minutes, peak3.30GiB.

[Protocol](EXECUTIVE_PILOT_V1.md) ·
[Archived results](../../artifacts/neural_controllers/neural_executive_v1/results.json)

## Expanded executive curriculum

Fresh executive LoRA,16,384training missions with2–5objects, randomized identifiers,
five partial-order generators, two epochs and4,096updates. The first saved six-block
library from the author pilot remains frozen. The previous executive is evaluated
on the same new scenes as the new model, with a common512-token generation limit.
All768model replies freeze before1,024executions including the reference.

| Panel | Previous | Expanded curriculum | Reference |
|---|---:|---:|---:|
| Familiar3 | 32/32 | 32/32 | 32/32 |
| Familiar4 | 19/32 | 32/32 | 32/32 |
| Familiar5 | 18/32 | 32/32 | 32/32 |
| Long6 | 14/32 | 32/32 | 32/32 |
| Long7 | 13/32 | 28/32 | 32/32 |
| Long8 | 10/32 | 28/32 | 32/32 |
| Renamed8 | 18/32 | 26/32 | 32/32 |
| Jammed8, safe failure | 13/32 | 28/32 | 32/32 |

Ordinary mission success increases124/224to210/224, or55.4%to93.8%. There are86paired
improvements and zero regressions. Familiar3–5missions reach96/96. Unseen lengths6–8
reach88/96. The jammed panel adds15improvements with zero regressions, measured as
bounded failure after preserving all earlier deliveries and their required order.

All256new programs compile. Every ordinary failure omits at least one delivery.
Of those14failures, five also violate ordering. Four jammed cases fail the safe
failure criterion. The reference gate passes. All predeclared progression targets
are met. Untouched Qwen compiles1/256programs and achieves zero mission successes.

Final training update occurs at9,180.7seconds (2h33m). Total elapsed15,350.6seconds
(4h16m). Peak allocation3.39GiB. Final development completion NLL is0.00044924.
Generated tokens total24,756for the new model,23,430for the previous model and
65,759for the base. Generation times are1,118.5s,1,097.6s and3,788.0s respectively.

The experiment uses a single training seed. Data volume, optimization budget,
identifier distribution, graph coverage and trained lengths change together.
The results support this combined curriculum intervention. The reference uses the
same topological algorithm that supplies training labels, so this evaluates learning
and generalization of a supplied procedure. Longer missions share primitives and
substructures with training. The explicit interpreter supplies failure handling.

[Protocol](EXECUTIVE_CURRICULUM_V2.md) ·
[Full metrics](../../artifacts/neural_controllers/neural_executive_v2/results.json)

## Evidence and reproduction

The compact archive includes all generated replies, all reserved task packets,
per-case scores, validation losses, six deployed neural artifacts, and the first
reserved execution trace per arm. Each archive_identity.json records file hashes,
source-manifest identity and completion receipt metadata. All original executive
trace hashes are recorded, while full traces and model checkpoints remain in the
local experiment archives. The v1 executive's source snapshot is included because
the supported runner now prepares v2.

Full approved manifests and registries contain machine-specific runtime bindings
and are retained locally. Running a new scientific evaluation requires preparation
and explicit approval under AGENTS.md. The public source and compact artifacts
support inspection of the method and outcomes. Local training paths and dependency
environments must be configured for a new machine.

Engineering validation in the publication checkout covers the controller author,
DSL, independent executive scoring, parser/compiler, planner and source gate.
Run with the repository's development environment using
`python -m unittest discover -s tests -p test_control_dsl.py` for the DSL fixtures.
