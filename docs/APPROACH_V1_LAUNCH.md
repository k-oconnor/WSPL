# November 3 launch, horizon diagnosis and prepared repair

Kevin approved original manifest
`e05119fd34b2c77973773f5abcebb50858f45bc523287ea8eeddbcac3659bba9`.
The digest was registered with user approval provenance. The original manifest
and spec are preserved unchanged; output is results/approach_v1.

## Launch infrastructure

Attempt 1 (PID 35568) was rejected before model construction because the isolated
virtual environment lived inside the repository. The source gate treats packages
under the checkout as repository imports and rejected preloaded _distutils_hack.
No model or simulator work occurred. Both launch logs are retained.

The unchanged environment was moved outside the repository. Python identity,
installed versions, all 12,805 external file hashes and all 24 project Python
files were reverified. The ignored artifacts/local_paths.json records sim_python.
The source gate was not changed or bypassed. Attempt 2 (PID 7624) ran through the
gate, instantiated the pinned constructor once and reached physical development.

## Observed result and failure

On development seed 101 the reference started 0.2527825 meters from its supplied
standoff target, reached 0.0063554 meters, and satisfied the independent position,
speed and three-tick stability observer in 41 approach motor calls. Twenty scene
settling calls preceded it. This is one successful reference positioning trial,
not a grasp, learned controller result or reliability estimate.

The paired zero-translation control had the identical starting-frame hash. It
then hit `ValueError: executing action in terminated episode`. Its 20 preparation
calls consumed part of the native horizon, which was incorrectly set to 120
without adding the 120-call skill budget. After 100 actual approach steps, the
101st attempted call was rejected. LIBERO overwrites the returned horizon done
flag with task success, while robosuite retains its internal termination state.
The adapter's independent skill timeout therefore did not prevent this earlier
native expiry. This is a runner bug, not a failed learning result.

Run result and gate receipt both say failed. Recorded wall time: 11 seconds;
40 preparation calls and 142 attempted approach calls across two episodes (one
rejected native call included). External content was unchanged after execution.
No source demonstrations, fitting or evaluation panel ran. The 39,584-byte final
checkpoint contains no optimizer history/training updates, and a read-only audit
confirmed all four model states exactly match fresh/analytic initialization.
No background process remains active.

## Replacement prepared, not approved or launched

Native horizon is now `settling_steps + max_steps + 1`, or 141, preserving the
independent 120-call skill timeout. A synthetic regression reproduces the old
20+100-step failure and checks all 20+120 calls under the corrected horizon.
All 100 unit tests pass. Only three inventoried Python files changed:
src/skillforge/approach.py, experiments/run_approach.py and tests/test_approach.py.
All constructor and runner configuration values remain identical to the original
approved study. The replacement starts fresh; it does not restore the failed run.

Candidate: experiments/source_pins/approach_v1_horizonfix.manifest.json

SHA-256:
`75e33ff7d7d55a1500cfd9521a95f26dc1443f5a6354c3cde595749b87f6931b`

Model ID: approach-v1-horizonfix; proposed output:
results/approach_v1_horizonfix. Expected several minutes; same CPU-only execution,
20-minute wall guard, no APIs and final/clean-stop checkpoint. Exact source
approval is required under AGENTS.md before registration or launch. The original
approval remains historical and is not transferred to this replacement.
