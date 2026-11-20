# Recovery prepared; not launched

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
