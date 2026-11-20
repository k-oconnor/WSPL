# Frozen Astra-to-Qwen adaptation evidence

See [the audited report](../../docs/wspl/ASTRA_SUPERVISOR_RECOVERY_RESULTS.md).

This archive contains exact deployed policy/interface sources, author provenance,
all 64 final evaluation rows, and the archived development ablation summary.
The 74 MB adapter and full simulator/training/author transcripts remain in the
local frozen run; they are not included in this Git archive. Policy source alone
is not runnable without the declared model, runtime and saved adapter.

Adapter SHA-256: `d3a67a7d84c53d0041d8f6886709acac59723ca31b4c7bd784bd88962bc6c50c`.

Final evaluation: baseline 8/32, submission 32/32, 24 corrections, zero regressions.
Development ablations were authored and run by Astra; they have not been
independently rerun. No general robotics or WSPL-superiority claim is made.
