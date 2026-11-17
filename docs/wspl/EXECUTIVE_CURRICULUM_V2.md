# Expanded neural DSL executive curriculum

Prepared November 16, 2026. No new training or reserved evaluation launched.

## Motivation and scope

The first executive scored32/32 on familiar three-object missions,25/32 on four,
19/32 on five,9/32 on five with new names, and21/32 safe failures on jammed four-object
missions. All trained programs parsed. Missing deliveries and ordering errors were
the main failures. This pilot expands examples, length coverage, names and partial
orders together; it does not isolate which curriculum change causes improvement.

Keep Qwen2.5-Coder-1.5B-Instruct, rank16/alpha32 LoRA, compiler, interpreter and the
same six frozen author-generated neural blocks. Train a fresh executive adapter;
do not continue the previous optimizer. The previous final executive is a frozen
evaluation comparator on the exact new panels. No API calls or motor updates.

This is supervised scene/task-to-program synthesis. It is not learned recurrent
block routing, fault recovery, perception, obstacle avoidance or contact physics.
Coordinates vary within feasible workcell layouts; the supplied motor skills
handle trajectories. The main learned problem is complete goal ordering/binding.

## Curriculum

16,384 training missions, evenly spanning2–5objects;256development missions.
Fresh randomized lowercase identifiers with four letters and two digits, multiple
prefixes and randomly assigned numeric target labels. Five partial-order generators:
chain, independent goals, fan-out, fan-in and random DAG. Each occurs across all
training lengths. A supplied topological teacher emits transfers in an order that
satisfies prerequisites, breaking available-goal ties alphabetically, then returns
home. The prompt declares this rule. Names do not determine the constraints.

Eight spaced source locations and eight spaced destinations support the larger
workcell, with bounded per-scene jitter. Training uses subsets of these same sites.
There is one held object, geometric contact and no obstacles. Full prompts are
disjoint between splits; holdout structure claims are limited to lengths6–8 and
the declared new identifier format. Smaller substructures are shared.

Two training passes,4,096updates, microbatch1/accumulation8. BF16 frozen base, FP32
LoRA/AdamW, LR0.0002,160-update warmup followed by cosine decay to10%, weight decay
0.01, gradient clip1, completion-only loss, context2048 and no truncation.
Development NLL at initialization and epoch boundaries; retain final epoch.
More data entails four times the previous update budget; this is not compute matched.

## Evaluation frozen before execution

256reserved missions:32each for familiar3,4,5objects; long6,7,8objects;8objects with
unseen mixed-case identifier format; and8objects with the reference-final grasp
jammed. Jamming is not revealed in prompts. No recovery code is added.

Three model conditions: untouched base, previous final executive, new final
executive. All use identical prompts and greedy generation capped at512new tokens.
Freeze all768replies before simulation. This larger cap is shared by all arms and
differs from v1; compare the previous executive on the new panels rather than
equating historical totals. Add256deterministic reference programs, giving1,024
execution slots. Every failed program stays in its denominator. At most one text
Markdown fence is stripped, with no repair feedback or resampling.

Independent scoring checks final geometry, all required deliveries, before
constraints, no repeated delivery or later regrasp, empty gripper, return home and
no physical boundary violation. Interpreter completion is not enough. Jammed
missions report safe failure separately: bounded timeout after all unjammed goals,
preserved placements and ordering, no violations, empty gripper. This is interpreter
failure handling, not learned adaptation. Boundary checks detect rather than
universally prevent unsafe actions.

The runtime limits remain64expanded instructions,160ticks per skill and2,048ticks
per episode. All32reference missions in every panel must achieve the relevant
success/safe-failure outcome. Otherwise report reference gate false and withhold
progression interpretation; do not silently discard cases.

Report per-panel construction, task success and safe failure; paired improvements
and regressions against the previous executive; omitted goals and ordering errors;
trace/tick/instruction counts, token use and elapsed time. Programs, prompts,
weights/adapter identity and complete physical/block-call traces are saved.
Descriptive targets: at least90% pooled familiar3–5 success,60% pooled long6–8
success, and a positive paired gain over the previous executive on pooled nonjammed
missions. Renamed and jammed panels are separate stress tests. One training seed;
no broad superiority or real-robot claims and no automatic follow-on.

## Operations and authorization

Expected3–5hours including generation/evaluation;12-hour wall guard. One rolling
checkpoint overwritten every30minutes and at epoch/final/clean-stop boundaries,
plus final deployment adapter. No automatic retry; no tested full-study resume.
Stop requests are checked between updates/generations/executions. Partial optimizer
steps cannot replace a valid checkpoint.

The full-size synthetic preflight checks zero-adapter parity, frozen-base identity,
exact next-update restoration and exact loading of the prior executive adapter.
The model/runtime, prior adapter, frozen library and this protocol are hash-bound,
as are all project Python sources. Exact source-manifest approval under AGENTS.md
is required before generating the scientific corpus, training or reserved scoring.
Unit/synthetic preparation fixtures are exempt. Neither pin registration nor
launch is authorized by preparation alone.
