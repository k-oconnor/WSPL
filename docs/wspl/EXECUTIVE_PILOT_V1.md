# Substantial neural DSL executive pilot

Prepared November 16, 2026; awaiting exact manifest approval. No study launched.

## Question and separation of components

Can a small trained model write executable multi-object programs that generalize
to longer missions, while using a frozen library of neural motor/predicate blocks?
This tests the executive, not new motor learning. The frozen library is the first
reserved six-block set produced by the successfully trained controller author in
controller_author_v1. Selection is by saved record order, before this study's
execution scores. Its weights, specifications and source records are hash-bound.
No API authoring calls and no motor-weight updates occur.

The executive receives object/target coordinates, placement goals and acyclic
before constraints. It writes DSL text: transfer Object to Point, or explicit
move_to/grasp/release calls. An interpreter executes the program using the frozen
blocks. This is program-level learned selection; it is not a Gravemind neural router
choosing a new block after each observation. There is no learned recovery policy.

## Data and models

Generate 4,096 training missions and 128 development missions with two or three
objects. Each has randomized object names, distinct source/target layouts, a
shuffled goal map and a sampled partial order. A deterministic topological teacher
chooses the alphabetically first currently available goal, then returns home.
The prompt explicitly states this tie-break rule. All scenes are planar and contain
no obstacles. Generation seeds are fixed in executive_study.py.

Training labels are supplied planner programs; no neural simulation outcomes are
used as labels. This is supervised program synthesis from structured tasks, not
learning manipulation from demonstrations or discovering an algorithm unaided.

Train a fresh rank16/alpha32 LoRA on frozen Qwen2.5-Coder-1.5B-Instruct, all seven
attention/MLP projection types (18,464,768 trainable parameters). Two passes over
training data, 1,024 updates, microbatch1/accumulation8, completion-only loss.
BF16 frozen base, FP32 adapters/AdamW, LR0.0002, forty-update warmup, cosine to10%,
weight decay0.01, clipnorm1, context2048 without truncation. Fixed ordering seeds.
Development NLL at initialization and both epoch boundaries; retain final epoch.
No test-dependent selection or tuning. Compare the untouched base and final trained
executive, with identical greedy generation and a256-token output limit.

## Reserved panels

160 missions,32 per panel:

| Panel | Purpose |
|---|---|
| Familiar three-object | New sampled layouts, goals and order constraints |
| Four-object | Longer compositions than any training mission |
| Five-object | Further task-length/partial-order generalization |
| Five-object, new names | Length plus unseen object-name vocabulary |
| Four-object, jammed final grasp | Safe bounded failure and preservation of prior deliveries |

Training/development/test prompts are disjoint. Functional primitives and scene
generation rules are shared. Four/five-object goal structures cannot occur in the
two/three-object training set, but their smaller substructures can. These are
synthetic generalization tests, not a natural-language or physical-robot benchmark.
Jamming is hidden from the prompt and affects the final object in the reference
order. There is no retry/replanning; report safe failure separately from success.

Freeze all320 model replies before any simulation evaluation. Execute each reply
and the deterministic reference on the same160 scenes:480 candidate executions.
Malformed replies remain failures. Strip at most a single text Markdown fence;
perform no program repair. The reference scores expose unreachable fixtures; if
any reference fails its declared success/safe-failure criterion, mark the reference
gate false and withhold interpretation of progression rather than drop that case.

## Execution and independent scoring

Each transfer expands to four stateful primitives using the same shared frozen
blocks. Instruction limit64, skill limit160ticks, episode limit2,048ticks. The
workcell has a single gripper and geometric contact; no contact dynamics or vision.
Neural completion proposals are checked against observed geometry/holding.

Task success requires every object at its assigned final target, all deliveries
in the required temporal order, no subsequent regrasp or repeated delivery, no
physical boundary violation, empty gripper and return home. Interpreter completion
alone is insufficient. These independent event-order/final-state checks implement
the task properties directly; the executive does not emit LTLf in this pilot.
The existing LTLf unit fixture remains an engineering cross-check, not a universal
controller certificate. Boundary violations are detected after a step, not prevented
by a verified shield.

The jammed panel's separate safe-failure criterion requires a bounded timeout,
all unjammed objects delivered and still correctly placed, no recorded violations,
and an empty gripper. It never counts as successful task completion. Safe failure
is largely provided by interpreter contracts; this panel tests the learned program's
interaction with that mechanism, not learned fault handling.

Report every panel separately: construction rate, physical task success, safe
failure, violations, completed instructions, tick counts, output tokens and timing.
Save all prompts/teacher programs/replies, model adapter, frozen block identities,
per-tick block inputs/outputs and physical observations. Report exact accuracy
counts, not a broad statistical superiority claim from one training seed.

Descriptive progression target: at least28/32 familiar successes, at least20/32
successes on each four/five-object panel, and improvement over base on their pooled
missions. New names and jamming remain separately reported stress panels. No
automatic follow-on, even if this target is met. A flat-only author/control and
learned stepwise routing are subsequent experiments, not hidden claims here.

## Operations and approval

Expected1–3hours on the3090, six-hour wall guard. One rolling checkpoint overwritten
every30minutes and at epoch/final/clean-stop boundaries, plus final deployment
adapter. A stop control is honored between updates/generations/executions. No
automatic retry and no tested full-study resume. Partial optimizer steps do not
replace a valid checkpoint. The initial optimizer starts fresh; the prior motor
author's adapter is not used to initialize the executive.

Synthetic full-size preflight checks zero-adapter parity, frozen base identity and
exact next-update restoration. Unit tests cover independent scoring, wrong order,
skipped tasks, repeated deliveries, prompt split identity, jams and DSL contracts.
Model/runtime/library/protocol resources and all project Python sources are pinned.
AGENTS.md requires an exact user-approved source manifest before corpus construction,
training and reserved scientific evaluation. Preparation/unit fixtures are exempt.
