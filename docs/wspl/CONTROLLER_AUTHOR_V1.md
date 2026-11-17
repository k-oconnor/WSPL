# Modular neural controller author pilot

Prepared November 12, 2026. Kevin selected finite-trace LTL over state predicates
with separately typed neural action blocks. This pilot trains a small author model
to emit WSPL programs that compile into executable ReLU blocks. A supplied LTLf
compiler and finite-effect planner compose those blocks into pick-and-place tasks.

## Concrete deliverable

A Qwen2.5-Coder-1.5B-Instruct LoRA adapter produces typed WSPL modules from supplied
functional specifications. Six kinds cover clipped planar motion, proximity margins,
holding margins, gripper close/open commands and workspace safety margins. The
existing WSPL compiler constructs their neural weights. Execution evaluates those
weights through the independent rational artifact evaluator. This first pilot uses
a deterministic planar kinematic workcell with one object and a binary grasp state.

Inputs to the motion block are goal-minus-position errors supplied by the task
executor. State predicates consume those errors, world position or a holding sensor.
Predicate truth is a nonnegative network output. The planner selects approach,
grasp, optional waypoint, transport and release using an LTLf automaton over a
supplied finite skill-effect model. The physical simulator updates the object only
when the gripper actually attaches to it. Success is checked on an independently
observed physical trace using direct LTLf semantics, so a false neural predicate
cannot manufacture a successful task merely by advancing the abstract planner.

Direct mission requires always-safe execution and holding followed by a later
delivery. The waypoint mission additionally requires a visit between those events.
The formula changes the synthesized action sequence. This tests a compositional
pipeline with a supplied symbolic planner and teacher, and establishes the scope
for later robot simulation and richer skill families.

## Corpus and splits

After approval, construct 1,536 training records,96 development records and96 test
records, evenly covering six block types. Each prompt supplies ports, units,
functional semantics and numerical parameters. Each target is a compiler-authored
WSPL module. Complete specifications are unique and disjoint across splits.
Functional families and syntax patterns are shared, so evaluation measures new
parameter specifications and recombination within these families. It does not
establish generalization to unseen algorithms or inferred dynamics.

Every reference program must parse, compile within limits, and pass numerical
checks against separately implemented functions before training starts. Check
boundary inputs as well as fixed random points. All32 reference mission combinations
must succeed. These corpus validation results are saved. A failure stops execution.

## Author training

Start fresh rank16/alpha32 LoRA adapters on the local frozen Qwen2.5-Coder-1.5B-
Instruct checkpoint. All seven attention/MLP projection kinds are adapted, yielding
18,464,768 trainable parameters and1,543,714,304 frozen base parameters. BF16 base
computation with FP32 adapters, gradient checkpointing, microbatch1, accumulation8.
Train two passes,384 optimizer updates, completion-only loss. AdamW LR0.0002,
weight decay0.01, twenty-update warmup and cosine decay to10%, gradient norm cap1.
Maximum context2048, reject overlength examples without truncation. Validation
NLL at initialization and each epoch. Retain final epoch regardless of dev score.

Use one rolling training checkpoint every30minutes, plus epoch-boundary and final
saves to that same path. Save one final deployment adapter. A control.json stop
request stops cleanly at an update/phase boundary. No automatic recovery or follow-on.
Checkpoints record the approved source identity and training counters. They preserve
weights and optimizer state; a fully tested interrupted-study resume path is future
work. A partially interrupted optimizer step must not replace a valid checkpoint.

## Reserved execution evaluation included in this run

Freeze the final adapter before reserved generation. Generate all96 test modules
with the trained adapter and the untouched base,192 greedy outputs total, maximum
768new tokens per program. Identical prompts, one attempt each, no feedback. A single
Markdown fence may be removed mechanically. Other syntax/construction failures count
against the denominator. Freeze all replies before compilation and scoring.

Report construction rate, numerical behavior correctness at1e-6 maximum error,
source/token length and learned-model timing. Compile at most8,192 target-network
parameters,16layers and256units per program. Store the source and resulting weights.
Assemble16six-block libraries from reserved specifications and execute each under
direct and waypoint LTLf formulas,32missions per arm. Two predefined starting/goal
layouts alternate. Failures retain their status and complete recorded trace.
Report actuator violations, predicate stops, budget failures and actual LTLf success.
All actions are bounded and each skill has at most160simulation ticks. These are
kinematic simulations; contact physics, obstacles, perception and real robot control
are outside this pilot's implemented environment.

Primary report includes all96block slots and32mission slots per arm. A descriptive
progression target is at least90% behavior-correct blocks and80% successful missions,
with improvement over the untouched author. This pilot estimates feasibility in
shared function families. Comparison with gradient-trained target controllers and
broader task families belongs to a subsequent study with its own budget.

## Runtime, validation and launch

Use C:/Users/kevin/skillforge-runtime/venv-smollm/Scripts/python.exe on the3090.
Expected roughly1–2hours including generation, six-hour whole-study guard. Local
training and inference only. No author API calls. Hash model files, key installed
runtime packages, protocol and all inventoried project Python sources. The runner
verifies bound resources before corpus construction and model loading.

Nine initial controller checks passed, covering compiler/oracle agreement,
parameter-split identity, temporal order, false-positive predicates, wrong gripper,
unsafe commands and both mission routes. Full-size synthetic preflight verifies
zero-adapter logits, frozen parameter identity and exact next-update restoration.
Peak preflight allocation6,317,266,432bytes. This preparation has made no task
training updates or reserved author predictions.

The existing source approval rule requires an exact independently approved manifest
before scientific execution. Prepared entry point is
experiments/run_controller_author.py::run with
skillforge.authoring_sft.AuthoringSFTStudy as the checked constructor. Isolated
checkout is WSPL-controller-author-v1 on research/ltlf-controller-author-v1.
