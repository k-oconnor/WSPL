# Prepared repair screen launch identity

Completion update: run and source receipt both completed at 00:18:15 PST on
November 4. Both arms succeeded 6/6; see [results](REPAIR_SCREEN_V1_RESULTS.md).
The remaining text records launch provenance, not current running status.

November 4, 2026. **Approved and launched at 00:11 PST (08:11 UTC).**

Kevin explicitly selected "Approve this manifest and launch the 12-proposal
screen" for the digest below. It was reverified and registered in both main and
the isolated checkout. Hidden launcher PID 41992, worker PID 38748. The source
gate passed and the first proposal completed evaluation. The remaining queue
continues without feedback; inspect live status before claiming completion.

[Live status](../../../Skillforge-wsp-repair-v1/results/wsp_repair_v1/status.json)
and [launch snapshot](wsp_repair_v1_launch.json). The source/auxiliary/runtime,
task/prompt and CLI identities were reverified before launch. No source edits
were made in the isolated checkout after launch.

- Protocol: [REPAIR_SCREEN_V1.md](REPAIR_SCREEN_V1.md).
- Manifest: `experiments/source_pins/wsp_repair_v1.manifest.json`.
- SHA-256: `807c27b5bc0d769969a6bae14dda809fc68b4b185cccfad704b491658e2a28fe`.
- Registered model ID: `wsp-repair-v1`.
- Isolated checkout: `C:/Users/kevin/OneDrive/Desktop/Skillforge-wsp-repair-v1`.
- Output: `results/wsp_repair_v1` in that checkout.
- Runtime: `C:/Users/kevin/skillforge-runtime/venv-wspl-match/Scripts/python.exe`.

The isolated checkout is a frozen source snapshot over Git base `29eb1cd`, with
the new implementation copied into it. Its base commit alone does not identify
the complete source. All 90 inventoried Python files match the source manifest;
14 bound auxiliary files, task identities, CLI executable and runtime dependency
hashes were verified. Approval is now registered. Main remains free for
independent development; execution must use the isolated snapshot.

All 407 primary unit checks completed: 406 passed, one native fixture skipped.
The skipped fixture is covered by 12/12 passing checks in the separate MinAtar
runtime. All 18 repair harness tests passed. Six prompt preflights and the isolated
checkout's longest-prompt preflight passed without model calls. The stress report
records 100 generated DAGs/1200 edit steps and three archived controller checks.

Twelve fresh Astra/xhigh calls, three supplied synthetic circuit repair tasks,
direct versus structured editing, two draws each. One response per call; no
feedback, training or scored episodes. Existing Codex account usage; no hard
dollar/output-token cap or API-key fallback. 20-minute invocation guard and
four-hour study guard. Expected 1–3 hours, uncertain. Final candidate/receipt per
valid proposal, with all failed and unattempted trials retained.

Launched through `experiments/source_consistency.py run --model-id wsp-repair-v1
--output results/wsp_repair_v1`, with numeric thread counts set to one. Stdout,
stderr, launch JSON, explicit approval receipt and PID are preserved beside the
output directory. Do not launch a duplicate or automatically restart a failed
attempt; preserve its invocation/usage history and inspect the failure first.
