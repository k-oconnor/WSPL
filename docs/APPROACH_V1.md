# Approach v1: unblinded positioning and imitation screen

Prepared November 3, 2026. **Unapproved; no scientific fitting or physical
comparison has launched.** This is a deliberately small first physical motor
screen, not the full blinded LLM/gradient acquisition study in EXPERIMENT_01.md.

## Question and scope

Can the declared fixed neural architectures execute a useful positional
controller through the real simulated motor interface? Can AdamW imitation of
that same controller learn useful positioning from four trajectories? Does the
complete RMV scalar export faithfully reproduce its neural motor outputs?

The target is a supplied fixed world point **12 cm above the settled cream-cheese
object's origin**, using LIBERO's cream-cheese/bowl scene. This is pregrasp
standoff positioning, not a validated contact-ready grasp pose or pickup. Target
selection, object perception, orientation holding and the ordinary OSC servo are
supplied infrastructure shared by all arms. No planner, LTLf hierarchy, grasp
policy, collision observer or hidden approach routine completes this task.

This narrower task establishes positioning before contact/grasp calibration.
No physical feasibility or learning result is known yet. The initial source gate
will stop before fitting if the reference fails or zero translation succeeds.

## Methods

| Method | Implementation |
| --- | --- |
| zero | Zero translation/rotation, open-gripper command |
| reference | Per-axis clip(20 * target error in meters, -1, 1), open gripper |
| mlp_authored | Analytic weights encoding that controller in the ordinary 12-32-32-4 ReLU MLP; 1,604 parameters |
| rmv_authored | Analytic weights encoding that controller in the ordinary 12-input, 24-literal, 56-clause, four-head RMV; 1,588 parameters |
| mlp_adamw | Fresh MLP, imitation on the fixed four reference trajectories |
| rmv_adamw | Fresh RMV, identical imitation examples and update budget |

The two analytic arms are assistant-authored, inspected constructions encoding
the known teacher. No simulator feedback was used to revise them before launch.
They are **not a blinded independent LLM weight-writing trial**. There are no
external API calls or measured writer acquisition costs; conversation/design
cost is not captured. Do not use this screen to claim an LLM efficiency advantage.

Analytic MLP uses six signed-error ReLU paths. Analytic RMV uses three one-literal
clauses and three heads; unused capacity remains included in its parameter count.
Each forward pass still evaluates the declared full architecture. No code policy
or motor-action bypass replaces the network. Scalar teacher equivalence was unit
checked on positional boundary/interior probes before launch.

## Frozen interface and initialization

All twelve channels use the names in `skillforge.motor.CHANNELS`. Object and
target deltas normalize from [-0.5,0.5] meters, velocities from [-1,1] m/s,
aperture from [0,0.08] meters, and contact indicators from [0,1]. Normalization is
clipped. MLP consumes 2p-1; RMV consumes fixed positive/negative rails p,1-p.
The earlier illustrative lift skill JSON used some different names/ranges; that
unvalidated example is not this experiment's contract.

RMV W/V are direct nonnegative parameters, projected to nonnegative values after
every optimizer update. Hard forward clamps use ordinary PyTorch derivatives;
there is no straight-through/surrogate gradient. Initial asymmetric AND rows
sum to randomly chosen values in [1.2,1.8]; OR rows sum to 0.8. Readout weights
are small signed random values. Unit probes establish input sensitivity, distinct
clause rows, interior gates and finite parameter gradients. The runner repeats
an input/parameter-gradient audit on the actual source examples before fitting.
An inactive/nonfinite parameter group or input-insensitive initialization aborts
the screen and retains the failure, rather than choosing another seed.

Every model seed is 7. Each arm has independent tensors; authored and fresh models
share normalization/architecture, not optimizer state or parameters.

## Physical episodes and checks

One Panda, 20 Hz, headless OSC_POSE; see SIM_ADAPTER.md for action scaling and
sensor definitions. Twenty zero-translation open-gripper motor calls settle the
scene before fixing the target. Those preparation calls are charged for every
episode. Reset errors are recorded and not silently retried or excluded.

Each attempted approach gets at most 120 motor calls (six simulated seconds).
Initial end-effector distance must be at least 5 cm. Completion requires the
independent observer's 1 cm position tolerance and 0.03 m/s speed tolerance for
three consecutive 20 Hz samples. Missing ticks abort. Native task success/reward
is diagnostic only. This screen does not measure collision avoidance, and makes
no physical safety/admission claim.

1. Development: seeds 101-104, reference and zero controls. All four references
   must succeed and all four zeros fail; otherwise stop before source collection
   or fitting. Exact starting-frame hashes must match within each pair.
2. Source: seeds 11-14, reference only. Retain all four full trajectories;
   stop on any teacher failure. The two analytic models do not read the examples;
   they already encode the teacher. Both gradient arms see every source row.
3. Fit: 2,000 full-batch AdamW updates per learned model, LR 0.003, default betas
   (0.9,0.999), epsilon 1e-8, zero decay, clip global gradient norm to 1.0.
   Fixed final endpoint, CPU FP32, two Torch threads. No best-checkpoint selection,
   curriculum, extra seeds, LR search or automatic retry.
4. Evaluation: seeds 1001-1008, all six methods, paired exact start hashes. Use
   all eight attempted starts in reported denominators. Retain all trajectories,
   failures/timeouts and motor outputs. No evaluation-driven tuning or selection.

These are disjoint random-seed starts in the same scene family. They are not
held-out object/layout families and do not establish broad transfer. All seed
lists and task choices are inspected engineering choices. Eight evaluation starts
per method support a descriptive feasibility screen, not a high-reliability claim.

Maximum if every stage proceeds: 60 episode attempts (8 development, 4 source,
48 evaluation), 1,200 preparation motor calls and at most 7,200 approach motor
calls. Constructor/reset internal physics is separate from those command counts;
reset counts and total run wall time are reported. Failed control attempts are
charged. Learned arms report updates, source rows and row presentations. API calls
are zero. Initial environment validation and model construction count in run wall
time; some early environment-creation failures are also recorded by the source
gate as failed launches rather than completed study episodes.

## Fidelity and artifacts

The unit suite compares independent scalar RMV exports with neural outputs on
all 4,096 twelve-predicate Boolean corners, 128 random interior points and the
midpoint for both fresh and analytic blocks; maximum discrepancy must be <=2e-6.
The runner checks each fitted RMV on source rows and each evaluated RMV on its
actual visited input rows. This is finite numerical evidence, not a universal
floating-point proof or a claim that a dense rule graph is easy to read.

Artifacts include episode JSONs, source command/observation pairs, initialization
audits, loss samples, success/attempt/error denominators by method, motor/reset
counts, complete RMV exports and one final/clean-stop checkpoint containing all
models, trained optimizer states, Torch RNG, training progress and source digest.
Run configuration/results carry deterministic episode seed lists. The 20-minute
wall guard is shorter than the half-hour rolling interval; no periodic weight
copies are expected. A control.json stop request (`{"stop":true}`) stops at an
episode/update boundary. Checkpoints support later approved continuation work;
this runner does not automatically resume an interrupted episode or study.

## Source/environment identity and launch

The prepared spec and manifest are under experiments/source_pins/approach_v1.*.
Candidate manifest SHA-256:
`e05119fd34b2c77973773f5abcebb50858f45bc523287ea8eeddbcac3659bba9`.
The registry remains empty until Kevin approves the exact manifest SHA-256.
Project inventory covers src, experiments, tests and tools. The configuration
also pins Python/platform, installed distribution versions and content hashes
for all 12,805 inventoried files in LIBERO, MuJoCo, robosuite and Torch, including
simulator assets/binaries and the already-applied isolated Windows fixes.
Other dependencies are version-pinned; they are not all content-hashed.
External identities are checked before and after execution. Local filesystem
bindings are ignored Git config and do not change the expected content.

After exact user approval and registration, launch only through:

```powershell
./.venv-sim/Scripts/python.exe experiments/source_consistency.py run --model-id approach-v1 --output results/approach_v1
```

Expected runtime is several minutes, with a 20-minute wall guard checked between
episodes/updates. That guard cannot interrupt a hung native simulator call. The
existing five-tick integration check and synthetic rollout unit fixtures are not
this experiment's physical feasibility results. Nothing launches automatically.
