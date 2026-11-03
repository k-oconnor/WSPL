# M2 engineering: physical sensor and command adapter

Implemented November 2, 2026. `skillforge.motor` defines the versioned interface
and independent sampled outcome observers. `skillforge.libero_adapter` bridges
one already-reset LIBERO environment to those interfaces. It performs **one native
motor step per command**, without any approach/grasp/lift/place policy inside it.
No legacy task-completing motor routine is imported.

The fresh isolated `.venv-sim` passed a headless integration unit check on LIBERO's
`put_the_cream_cheese_in_the_bowl` asset: six measured frames spanning five
20 Hz motor ticks, using zero translational/rotational increments and gripper
internal-command hold. This is sensor/control wiring evidence, not a skill,
no-op difficulty test, successful pickup or physical planning result. Constructor
and reset physics are outside the five counted command calls.

See [the machine-readable receipt](simulator_preflight.json). The first attempt
failed before stepping because BDDL's undeclared `future` dependency was missing;
that dependency was added to the isolated environment. All attempt receipts remain
under ignored `work/sim_preflight_v*`; the initial failure is also retained in
[simulator_preflight_initial_failure.json](simulator_preflight_initial_failure.json).

## Stack and isolation

Audited versions: MuJoCo 3.2.7, robosuite 1.4.1, NumPy 1.26.4, Gym 0.25.2,
BDDL 1.0.1; LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` (clean
checkout). A complete installed-version snapshot is in
[sim-win-py311.lock.txt](../requirements/sim-win-py311.lock.txt). This is a Windows
Python 3.11 development environment snapshot, not a universal compatibility claim
or a scientific source approval. It intentionally follows the older LIBERO API;
modern robosuite composite-controller APIs are not silently substituted.

The headless environment uses one Panda with the standard PandaGripper and fixed
impedance OSC_POSE at 20 Hz. Cameras, onscreen and offscreen rendering are disabled.
The adapter validates action dimension, controller type, delta/orientation mode,
input range and output scaling. Gains, assets and remaining controller parameters
must also be frozen in the first scientific experiment.

Two narrow robosuite Windows binding fixes locate `mujoco.dll` in the MuJoCo
package and exclude the Linux EGL branch on Windows. The preflight applies them
only when explicitly requested, only to robosuite 1.4.1 inside the active virtual
environment, checking the expected source text. Before/after SHA-256 values are
recorded. Installation knowledge came from the old
`rbl_rmv_icu/experiments/libero_control/patch_runtime.py`; this helper is freshly
implemented, not imported. Neither that environment nor the LIBERO checkout was
modified. The preflight writes an isolated LIBERO config under its output path;
it never uses or changes `~/.libero`. Run it in a fresh process because LIBERO
captures configuration at import time.

## Interface: skillforge.tabletop12.osc4.v1

| Channels | Measurement |
| --- | --- |
| object_dx/dy/dz | Named bound object's reported world origin minus end-effector grip-site position, meters |
| target_dx/dy/dz | Supplied fixed world target minus end-effector grip-site position, meters |
| eef_vx/vy/vz | MuJoCo grip-site world translational velocity, meters/second |
| aperture | Nonnegative projected opening between opposing oriented box finger-pad faces, meters; computed from world pad geometry and box support radii, not joint travel |
| left/right_object_contact | Penetrating/touching MuJoCo contact pair between the corresponding finger collision geoms and the bound object's collision geoms; unrelated table/object contacts are excluded |

The object/target bindings and geometry are supplied structured perception.
The target has no built-in task-solving behavior: approach must bind a properly
calibrated grasp pose; it is not enough to pass the object origin. The integration
check uses the object origin solely to inspect sensor wiring. There is no visual
perception, grasp-pose generator or independently learned target selection here.

Four finite outputs in [-1,1] map to seven native OSC_POSE commands:
`[0.4*dx, 0.4*dy, 0.4*dz, 0, 0, 0, gripper]`.
Native translation scale is 0.05 meters, so magnitude-one neural translation
requests a 0.02-meter setpoint increment. It is not an achieved displacement.
Zero rotation preserves the OSC orientation goal; orientation is not controlled
by the neural skill. PandaGripper maps command sign to incremental internal
gripper actuation: -1 opens, +1 closes, zero retains the internal command. It is
not an absolute aperture command. The adapter rejects out-of-range/nonfinite
commands rather than silently clipping malformed candidates.

One successful observation arms one step. Missing/nonfinite required sensors
disarm motion. Native reward/done/info remain diagnostics, never skill success
propositions. A native terminal state requires reset/rebinding. An independent
positive step budget is enforced because LIBERO's BDDL step implementation
overrides the ordinary horizon `done` with its task-success result. There is no
workspace safety shield, collision observer or hardware integration yet.

## Independent observer version 1

The observer accepts immutable physical frames, not network predictions. Every
invocation has a fresh ID, reset baseline, fixed target and exactly consecutive
sample times (default 0.05 seconds, numerical tolerance 1e-6 seconds). A missing,
duplicate or malformed tick, or a changed target, aborts that invocation. A new
invocation must explicitly restart observation. Default thresholds are engineering
choices awaiting task calibration, not admission criteria:

| Event | Definition, sustained for three consecutive samples |
| --- | --- |
| approach_done | End effector within 1 cm of bound target and translational speed at most 0.03 m/s |
| close_done | Both fingers contact the bound object; does not assert pickup or force closure |
| lift_done | Bound object's origin at least 5 cm above invocation baseline, with bilateral contact at invocation start and every sampled tick |

Completion emits once per invocation. Losing contact emits a one-tick
`contact_lost` edge; reacquisition cannot erase that lift invocation's failure.
This checks contact at sampled ticks, not continuously between them. It does not
exclude external support or prove a stable grasp; later admission must calibrate
those assumptions and disturbance tests. Calling close after contact is already
established is allowed; nontrivial initial conditions are an experiment concern.
The observer exposes aligned/contact/elevation states but no invented collision,
holding, placement or task-success labels. The existing abstract planner is not
yet connected to physical execution or these predicates.

## Reproduce engineering checks

The ordinary unit suite imports no simulator, trains no network and performs no
physical steps. Adapter fault fixtures require optional NumPy and skip explicitly
when it is unavailable. The separate integration check is opt-in:

```powershell
python -m venv .venv-sim
./.venv-sim/Scripts/python.exe -m pip install -r requirements/sim-win-py311.lock.txt
# Create ignored artifacts/local_paths.json with a libero_checkout absolute path.
./.venv-sim/Scripts/python.exe tools/sim_preflight.py --output work/new_preflight --fix-windows-bindings
```

The output directory must be new. This engineering helper has no trainable model,
candidate selection or skill score. Do not extend it into an unapproved benchmark;
scientific/training runners require their own exact source pin and must record
dependency fixes and external asset identities as well as project source.

Next: implement differentiable MLP/RMV motor blocks and scalar-export fidelity;
calibrate nontrivial grasp poses, no-op difficulty and a transparent reference
controller on development-only layouts; then prepare a source-pinned acquisition
screen. Physical primitive hierarchy execution, safe outcome contracts and the
four acquisition arms remain unfinished.
