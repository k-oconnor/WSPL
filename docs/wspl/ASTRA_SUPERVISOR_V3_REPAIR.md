# Supervisor v3: align author permissions and require actual persistence

The v2 run completed, but its collector injected a higher-priority read-only
filesystem instruction. Astra consequently kept training in RAM and submitted
the unchanged baseline. Its final 8/8 ordinary and 0/24 adaptation scores are not
a meaningful test of the intended writable-supervisor workflow. Preserve all
ten author calls, 2,610.6 allocated GPU seconds, source, logs and failed hypotheses.

## Repair

Use a dedicated supervisor collector profile. The CLI renders danger-full-access
instead of a contradictory global read-only instruction; native command tools
remain disabled, and tool/error events still fail collection. A dedicated developer
instruction explicitly authorizes delegated writes, checkpoints and exports in
/work and declares /inputs read-only. Actual Python execution remains confined
to the unchanged offline Linux sandbox. This is not access to the host filesystem.
The original read-only collector profile remains the default for other studies.

Audit the exact rendered developer text on EVERY collection preflight. Reject a
read-only permission contract or unexpected native tools before any model call.

The first two of the 64 available author responses are a mandatory persistence
gate. Astra must author a short CPU program writing a small float32 weight NPZ
and nonce receipt to /work, then author a separate-process program that reloads
those weights, calculates a specified result, and exports a hash/result receipt.
The host independently checks array values, dtype, nonce, unchanged file hash and
computed result. CPU only, 60 seconds maximum per job. A failure stops immediately;
no research training or final evaluation follows. No automatic retry.

Scripted tests exercise the mechanics, but the REAL Astra gate is part of the
approved launch; no claim that an offline preflight substitutes for it. A passed
gate proves basic tool persistence, not model quality or eventual checkpointing.

## Continued scientific design

Fresh author session, same original/ordinary-practice/taught starting artifacts,
full development simulator and model access, eight GPU-job hours and 20 development
hours, plus four-hour final guard. Up to 62 research responses after the two gate
calls. Previous supervisor logs and code are supplied as explicitly historical
evidence, not as current instructions; old RAM-only weights do not exist to resume.

Only a submitted frozen package reaches 32 NEW paired missions (64 episodes).
Previously evaluated v2 seeds are not reused. Astra is absent during deployment;
the final evaluator is not mounted into its worker. Component attribution remains
required before a student-learning claim. No automatic follow-on or expansion.

This requires a new exact source approval under AGENTS.md. Nothing is registered,
launched or collected until approval. Existing Codex usage has no hard dollar or
output-token cap and no API fallback.
