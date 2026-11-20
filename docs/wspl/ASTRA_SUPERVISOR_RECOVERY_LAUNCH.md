# Recovery prepared; not launched

## Approved launch update — November 20, 2026

User explicitly approved the exact digest below and selected “Approve this
manifest and launch the recovery.” Registered approval, reverified 326 sources,
and launched hidden at 01:00 PST / 09:00 UTC, launcher PID 85240. Output is
`results/astra_supervisor_resume_v1`, with adjacent persistent launch logs/PID.
The source gate passed; bound-resource verification is underway before recovery
or new author calls. This supersedes the preparation status below, not evidence
of completed restoration or research. Inspect live status and recovery receipt;
do not duplicate or retry automatically.

Manifest SHA-256: `49836cec374b8b73e933ce045dd7a95206043840ac67bff8d5c6ab81c24706fc`.
Model ID: `astra-supervisor-resume-v1`.
Output after approval: `results/astra_supervisor_resume_v1`.

53 focused checks pass. A read-only restoration of the real parent reproduced
the interrupted prompt exactly and copied all eight workspace files unchanged.
137 parent artifact files are bound. All 326 source files and 28,416 host resource
bindings verify; 35,406 Linux runtime files verified natively. Offline rendered
prompt audit passed. No new model calls, training or evaluation occurred.

Two persistence responses and two research responses are preserved; the failed
fifth invocation stays charged with unknown usage. At most 59 new calls, eight
GPU-job hours total, and 20 development hours minus 579.106 seconds already spent.
Downtime is excluded. Same separate four-hour evaluation guard. No automatic retry.

After explicit exact-digest approval, register it separately and launch hidden
through `experiments/source_consistency.py run --model-id astra-supervisor-resume-v1
--output results/astra_supervisor_resume_v1`, with persistent stdout/stderr/PID.
Do not register or launch based on this preparation document alone.
