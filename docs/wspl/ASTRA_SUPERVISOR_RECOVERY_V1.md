# Supervisor v3 capacity-error recovery

The approved v3 run passed the actual two-call weight persistence gate and
completed two CPU inspection jobs. Its fifth invocation stopped on an archived
429/model-at-capacity error before a completed reply. No GPU time or training
was used. Preserve the interrupted invocation with unknown usage.

Recover into a new output from a hash-bound copy of the original workspace and
exact visible transcript. Do not repeat the persistence gate or completed jobs.
Require the reconstructed next prompt to equal the interrupted prompt exactly.
Reject an already executed/completed interrupted turn or unresolved active job.
The original archive stays unchanged.

Keep all five attempted calls charged: at most 59 new responses under the total
64-call allowance. Preserve the eight-GPU-hour limit and subtract all 579.106
recorded elapsed seconds from the 20-hour development allowance; exclude downtime.
Same four-hour final guard and 32 fresh paired missions (64 episodes) after a frozen
submission. No model substitution, automatic harness retry or follow-on.
The underlying CLI may internally retry transport errors before reporting them;
the harness stops on the first streamed error, as in v3.

No changed scientific task, teaching recipe, tools, prompts or deployment policy.
Existing Codex usage has no hard dollar/output-token cap or API fallback.
The added recovery source and parent bindings require a new exact digest approval
under AGENTS.md before launch. Preparation/tests make no model calls or fits.
