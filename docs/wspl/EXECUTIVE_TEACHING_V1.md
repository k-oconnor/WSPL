# Astra-to-Qwen executive teaching: matched continuation pilot

Prepared November19,2026. Exact approval pending; no new author calls, student
training or reserved task predictions. This protocol supersedes the proposed
budgets in EXECUTIVE_TEACHING_DESIGN_V1.md.

## Question

Does teaching change the small LLM's ability to use an acquired motor skill?
Hold motor weights, documentation, observation fields and task interface fixed.
Compare the unchanged executive with interface/rehearsal continuation and
Astra-taught continuation. The main contrast is taught versus rehearsal, while
also requiring an advantage over the unchanged model. Tool addition alone or
host-written diagnosis cannot establish the learning claim.

Starting student: Qwen2.5-Coder-1.5B-Instruct with the completed closed-loop
rank16/alpha32 executive LoRA, seed113 selected in advance. Parent source
c5dacefd62c61ebcd9fcd439be9dca8e51ebda4620a65a7516f1aed6aefd90e3;
parent adapter tensor SHA-256
6f1e52d326fc6795658d6d24f8fc475cf9896aede293bd0774829b6c5450b58a.
Base stays frozen; learned LoRA weights are restored exactly, never reinitialized.
Both continuation arms reset optimizer/RNG identically.

Motor bank: original and repaired motor networks from the completed factory
pilot artifact, source3af5323cd31a9b7babc5bd012c1ce9896e9010f5acae4473fd8f5bdae5a2af48.
Both are45-parameter compiled networks. The artifact's automatic neural SELECTOR
and probe settings are NOT used. Bank weights stay unchanged throughout.

## Executive decisions and physical interface

One JSON action per decision: probe(command,ticks), select(module), reach(goal),
or stop. The host does not probe, identify faults, select a controller, retry,
advance goals or insert missing actions. No-controller reach is invalid. Explicit
select changes only the active motor module, with no physical action or weights.
Probe executes1..20ticks, translation bounded to[-0.2,0.2], gripper held open.
Reach uses the chosen network for at most10ticks; another chunk requires a new
executive decision. A sensor-based three-tick arrival check can end a chunk early.

Panda/MuJoCo/robosuite robot and20Hz OSC interface match the factory pilot. Encoder
noise SD0.0001rad, resolution0.0001rad; velocity resolution0.001rad/s; gripper
resolution0.0001m. Tool pose is nominal FK from the same noisy encoder sample.
Repeated reads at a tick share the sample. No camera inputs to this text-only
teaching comparison. Astra demonstrations and Qwen use the SAME telemetry and
module documentation. Camera tooling remains available in the broader system,
but privileged teacher-only image evidence is not required for student decisions.

Observation includes requested waypoint order/coordinates, sensor-based visits,
module IDs, selected module, latest action/result, initial/final measured error
for a reach chunk, robot telemetry and remaining budgets. No hidden mapping,
condition label, future schedule, oracle success, sorted solver action, or saved
selector score. Each decision uses this state, not an unbounded chat history.

Each mission randomly assigns opaque module/goal IDs and randomizes module-list
order. Documentation identifies nominal-calibration and cross-axis-compensation
roles consistently; documentation is identical across student arms. IDs may repeat
across missions; this is randomization, not a claim of unseen vocabulary.

Each reset has20setup settling ticks, reported separately. Physical budget is
60*N+80ticks for N goals, including probes. All invalid answers consume a decision.
Ordinary/evaluation decision budget4*N+12; teacher episodes have16. No repair of
syntax, automatic retry or evaluation-time weight update.

## Missions and independent scoring

Goals are taught absolute Cartesian waypoints derived from the initial reported
tool pose plus small offsets and jitter. This is reaching with an open gripper,
not picking objects or manipulating contacts. Original dynamics are identity;
changed command calibration is [[0,-0.8,0],[1.2,0,0],[0,0,0.9]], as in the completed
factory experiment. Neither teacher nor student receives this matrix.

Four conditions: ordinary; changed from start; original->changed after the first
physically reached waypoint; changed->original after that waypoint. Changes occur
at executive decision boundaries, without moving/resetting the robot or announcing
the event. Conditions are evaluator metadata, not model inputs.

An independent monitor observes true end-effector site position after each physics
tick. It counts a requested waypoint after three consecutive ticks within5mm,
checks order, and requires an explicit stop after all goals. Sensor-reported arrival
and physical success can disagree. Transit through an unrequested waypoint is not
counted as a completed command. The event scheduler uses the first physically
achieved waypoint but neither diagnoses nor repairs anything for the executive.

## Phase0: establish a student deficit before spending teacher calls

The unchanged student first attempts four acquisition missions, one per condition,
seeds34000000/34000002/34000004/34000006, two goals and16decisions each. These are
the first acquisition mission of each condition, NOT reserved evaluation scenes.
If it succeeds on all three nonordinary cases, finish with
no_acquisition_adaptation_deficit: no Astra calls, training or reserved evaluation.
Otherwise proceed under the fixed design; do not pick harder tasks or tune settings
from these results. These student outcomes are archived, not supplied as extra
private evidence to the teacher.

## Phase1: new Astra demonstrations

Eight two-waypoint missions, two per condition, seeds34000000..34000007. At most
16Astra/xhigh responses per mission =128calls maximum. Each call has the current
student-visible prompt plus a teacher wrapper requesting a prospective prediction
and one action. No hidden reasoning resume, extra observation history or private
simulator result. Prediction text is archived but NOT a student training target.

All responses, invalid actions, physical traces and failed missions are retained.
Only accepted actions from physically successful completed missions become positive
imitation records. Successful trajectories may still contain inefficient actions;
this is not an optimality certificate. Require at least one successful mission in
EVERY condition and nonempty usable labels. Otherwise finish with teacher_gate_failed,
without training, student evaluation or automatic budget expansion.

Eight trajectories are a small exploratory teaching set; repeated action exposures
are not new demonstrations. Do not substitute the old factory authoring transcript:
it used a different engineering interface and did not demonstrate this executive.

## Phase2: ordinary rehearsal and matched continuation

Collect whole ordinary missions from seed35000000 onward, cycling2..5goals until
at least512decision records exist. A supplied observed-state reference chooses the
documented nominal module and completes waypoints. This is explicitly reference-
generated interface/rehearsal data, not Astra data. Any reference mission failure
stops the study; do not silently discard it. Preserve mission lineage and all data.

Restore the same parent adapter separately for two arms; fresh identical AdamW
states. Each receives128updates, microbatch1/accumulation8, completion-only action
loss, LR0.00003 with16-update warmup and cosine to10%, decay0.01, clip1. BF16 frozen
base, FP32 LoRA and optimizer, rank16/alpha32 (18,464,768trainable parameters).
Context4096 with no truncation. No model selection or tuning on deployment results.

- Rehearsal arm:1024ordinary example exposures.
- Taught arm:512ordinary +512Astra-action exposures.

The first four ordinary examples of every update are identical across arms.
Sampling with replacement is fixed-seed and archived. Report unique records and
repeated exposures separately. Budgets are update/example matched, not token or
FLOP matched. No gradient through physics or motor networks. Save one rolling
checkpoint per arm every half-hour and at finish, plus final LoRA artifacts.

## Phase3: frozen student deployment

Freeze both final adapters, original parent, motor bank and corpus hashes BEFORE
any reserved prediction. Evaluate unchanged, rehearsal and taught executives on
the same32missions:8per condition, seeds36000000..36000031,3..5waypoints, fresh
geometry/noise/IDs. 96episodes total. Longer than the two-waypoint Astra teaching
missions, but3..5goal lengths ARE present in ordinary rehearsal: no globally unseen
length claim. No Astra calls occur during this phase.

Greedy one-action generation,96new-token cap,4096total context; preserve every reply
and cap hit. Invalid output remains a failure/consumed decision; no format repair.

Predeclared exploratory learning gate:

- Rehearsal control must succeed on at least6/8ordinary missions, establishing
  usable interface competence.
- Taught model loses zero baseline-successful ordinary missions against EACH
  comparison arm.
- Across24nonordinary missions, taught gains at least four net successes against
  EACH comparison arm; no nonordinary condition has a net loss versus rehearsal.

Report all paired corrections/regressions, per-condition success, probes, module
selection counts and physical ticks. Full timestamped traces allow recovery analysis.
Single teacher collection and one student parent/training seed are exploratory;
the gate is not a statistical superiority test or a general-capability claim.

## Engineering validation and execution

Synthetic physics fixtures cover all four conditions and explicit controller
switching. Full-size synthetic student checks verify exact parent restoration,
bit-identical next-update restoration, unchanged base and4096-token backward.
Peak synthetic allocation7,094,400,512bytes (~6.61GiB). No task prediction is part
of these preflights.

Expected2-5hours,12-hour overall guard and20minutes per Astra call. Existing Codex
usage, no hard dollar/output-token cap or API fallback. Local GPU student training
and inference; CPU robot physics in a separately bound runtime. No hardware control.
No automatic retry or further study. A model may remain GPU-resident during
teacher collection. Final/source receipts distinguish acquisition gate failure
from successful completion of all student phases.

Exact manifest approval is required before any teacher call, training or scientific
evaluation. Historical approvals do not transfer.
