# Active visual and telemetry acquisition for robot controller authoring

Prepared November 19, 2026. Unapproved: no author calls or benchmark executions.

## Question and scope

Can Astra acquire enough information through a robot interface to author a compact
motor repair and an applicability selector, retaining the original controller?
Camera observations are requested by Astra, not selected afterward by the analyst.
This is a single-session feasibility pilot, not a comparison of interfaces, a
demonstration that vision is necessary, or a physical-hardware result.

The prior planar pilot supplied two controllers and perfectly measured displacement.
Here Astra receives only the original controller, sensor observations, and its chosen
camera frames. It must construct the repaired controller itself. The environment is
MuJoCo 3.2.7 / robosuite 1.4.1 Panda with OSC_POSE and the Lift tabletop scene.
The task is reaching taught Cartesian waypoints with the gripper open, NOT lifting.
Native Lift rewards and object coordinates are not author/controller observations.

The planar Qwen executive and its six blocks are NOT transferred to Panda here.
The host supplies waypoint execution; Astra writes the low-level error-to-action
module and selector. Reconnecting a trained executive is a later, separately tested
stage. No local language-model inference or gradient fitting occurs in this pilot.

## Robot interface

20 Hz command interface: normalized OSC translation xyz and gripper command.
Translation components are bounded to [-0.2,0.2], gripper to [-1,1]. Nominal OSC
translation scale is 0.05 m at input1. These are target increments, not guaranteed
physical displacements. Orientation increments are zero. MuJoCo simulates joint
dynamics and the existing impedance controller tracks the targets.

Available telemetry: timestamp, joint encoder positions, joint velocities, gripper
joint positions, nominal tool pose, last command, and remaining interaction budget.
Joint positions have independent Gaussian noise SD0.0001rad per sampled tick and
0.0001rad quantization. Velocity resolution0.001rad/s, gripper resolution0.0001m.
Nominal tool pose is computed from the SAME noisy encoders using a separate FK
state; it is not a second exact position sensor. Repeated reads at one physical
tick reuse the sensor sample. No injected timestamp latency in v1.

Controller status reports only simulation_running. Force/torque sensing, certified
safety state and hardware fault codes are not implemented and are not fabricated.
Action bounds are an engineering envelope, not a safety certificate.

Two fixed 256x256 RGB cameras: frontview and robot0_eye_in_hand. The author may
request a capture now or schedule command_start, command_end, and
gripper_command_change captures. These are command events, not inferred grasp or
contact success. Optional clip_stride yields a chronological sampled frame sequence,
not encoded video. Captures do not advance time; command execution does. Camera
pose/calibration is supplied by the scene; images are noise-free rendered RGB in v1.
No retrospective buffer. Every image has camera, timestamp, sequence and SHA-256.

PNG bytes are attached to Astra calls through the CLI image interface. Offline
preflight verifies the actual encoded image bytes in the model-visible prompt;
each invocation archives copies and attachment order. This is local transport
verification, not a claim to inspect the provider's internal vision processing.
All visible history is carried forward; prior pixels are not automatically
reattached. Text records identify which frames were attached to which response.

## Hidden configuration and acquisition

Calibration bench A uses identity translation mapping. Bench B uses
[[0,-0.8,0],[1.2,0,0],[0,0,0.9]] between requested translation and the internal OSC
command. This represents a software command-frame/gain miscalibration, not a change
in payload, friction, or motor physics. The matrix, simulator code and ground-truth
state are not in the author packet. It is a known experimenter-selected fault;
generalization to unseen fault families is not claimed.

The investigator can address A/B to collect evidence. Neither bench label nor
regime identifier enters the deployed neural artifact. Two independent sessions
use reset seed331 and sensor seeds441/442. Each allows160 physical action ticks,
48 frames, at most20 ticks per requested action and12 attached frames per response.
All exploratory movement remains in the scene. Twenty setup/settling ticks after
each environment reset are logged separately and excluded from action budgets.

One Astra/xhigh session, at most12 JSON responses. Each includes a prospective
prediction. Operations: observe, act, install, evaluate, finish. No arbitrary
Python, shell, external lookup or executable fitting. Invalid requests consume a
response. Bounds and capture budgets are checked before movement. No automatic
retry of collection or failed study execution. Last accepted artifact is retained.

At most two evaluate operations, each one new episode per regime at seed551 and
offset [0.035,-0.025,0.02]m. Feedback contains robot-measured remaining error,
telemetry, steps and requested start/end images. Oracle physical success labels
and true position errors are stored privately and are NOT calibration feedback.

## Deployment artifact and hierarchy

Preserve original clipped proportional motor network: command=clip(10*error,+/-0.2).
Author a repaired WSPL motor network (inputs xyz goal-minus-nominal-tool-position,
outputs vx/vy/vz) and WSPL selector (inputs issued probe uvw and measured dx/dy/dz,
outputs original/repaired). Each is limited to1024parameters,4layers,64units.
Host compilation maps affine/ReLU/min/max/clip expressions to neural artifacts.
Execution uses the existing rational-real artifact evaluator; this is not evidence
about FP32 deployment robustness. Motor output clipping is shared across all arms.

The author also selects one bounded translation probe and1..10 probe ticks. At
mission start the probe is physically applied, followed by five zero-translation
settling ticks. Selector reads only command and noisy net displacement; larger
score selects a controller, ties choose original. Selection is fixed for the mission.
No observation is reset, no task/regime ID is supplied, and all probe/settling ticks
count toward the120-tick mission budget. Cameras help the engineer during acquisition;
neither Astra nor a visual encoder runs in the deployed controller. There is no
mid-mission switching in v1.

## Frozen evaluation

Freeze all sources/weights/probe settings before evaluation. Eight fresh reset/sensor
seeds771..778 with taught offsets [0.04,0,0.02],[-0.04,0,0.02],[0,0.04,0.02],
[0,-0.04,0.02],[0.03,0.03,0.02],[-0.03,0.03,0.02],
[0.03,-0.03,0.02],[-0.03,-0.03,0.02]m. Goal is the initial reported tool position
plus the taught offset, identical for paired arms. Success is true tool-site
distance <=5mm for three consecutive ticks, within120 ticks; physical scorer is
independent of the observed error fed to the controller. Report both errors.

Original-before -> changed -> original-return; same installed artifact throughout,
new physical reset for each episode. Original-return repeats scenes and noise and
is a paired return check, not additional independent evidence.

Four arms: original/no probe, original/author probe, selected/author probe, and
oracle inverse-mapping controller/author probe. The oracle gets the true matrix
only for feasibility; it is not a learned competitor. 3phases x4arms x8missions
=96final episodes. All outcomes, failures and physical traces retained.

Descriptive gate: original baseline >=6/8 successes; zero losses of baseline
successes in both original phases; at least two net corrections on changed dynamics;
all24reference episodes succeed. Report stage failures without expansion. Single
author draw and eight layouts do not establish statistical superiority.

Also report action/observation use, requested versus actually attached images,
model tokens, per-call wall time, mission ticks, probe cost, and calibration history.
No claim about visual usefulness unless the resulting trace supports a narrower
descriptive statement; an image ablation would be a separate experiment.

## Execution and limits

Expected30-90minutes;20minute per-call and four-hour overall guards. Existing
Codex account usage; no hard dollar/output-token cap or API fallback. CPU physics
and small-network evaluation; offscreen OpenGL rendering may use the GPU, but no
large local model or training. Compact last/final artifacts, full author transcripts,
frame PNGs and compressed physical traces. No automatic follow-on.

Exact source approval through experiments/source_consistency.py is required before
any author calls or scientific evaluation. Engineering fixtures and offline image
transport checks are completed separately without exposing reserved outcomes.
