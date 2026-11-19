# Approved closed-loop study launch

Kevin explicitly approved manifest `c5dacefd62c61ebcd9fcd439be9dca8e51ebda4620a65a7516f1aed6aefd90e3` and selected “Approve this manifest and launch the closed-loop study.”

Before registration, all 265 inventoried Python sources and 10,315 bound resources were reverified. The output directory did not exist. The approved digest was separately registered in the local approval registry.

Launched through `experiments/source_consistency.py run --model-id compact-closed-loop-v1 --output results/compact_closed_loop_v1` on November 17, 2026 at 18:23 PST. Launcher PID 49844; Python worker PID 48896. The source receipt exists and the live status reports training-snapshot construction progressing. This is startup evidence, not completion or evidence of trained performance.

The initial external PID write raced creation of the results directory and was subsequently saved successfully. External stdout/stderr redirection files were not created; use the runner's durable status, source receipt, training logs and artifacts. The worker was not restarted or duplicated.

Live status: `results/compact_closed_loop_v1/status.json`. Preserve inventoried source and bound inputs while running. Expected duration 8–16 hours; 36-hour guard. No automatic retry or follow-on experiment.
