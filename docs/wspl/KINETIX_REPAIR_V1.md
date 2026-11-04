# Fresh Astra native versus WSPL Kinetix repair

Prepared November 4, 2026. No new model requests or scored episodes have run.
Launch requires exact-source approval. This follows the completed released-repair
replay but is a new FP32 matched study, not the same numerical implementation.

## Question and design

Can the WSPL tensor editor produce useful repairs of the supplied trained car
controller, with performance comparable to native literal edits and useful
authoring-efficiency evidence? Six fresh Astra/xhigh one-shot requests: three
native, three WSPL; order native/WSPL, WSPL/native, native/WSPL. Every request
uses a fresh thread. Provider sampling seeds are uncontrolled. This is three
author draws per interface, not 3,072 independent authors.

Both receive exactly the same task packet and complete two-interface guide,
differing only in the final selected-arm line. The original rules, observation
layout, parent failure trace and all rounded weights are retained. Historical
skip-invalid parsing instructions are replaced symmetrically. Neither receives
the published repair, its explanation, local replay outcomes, other proposals,
tools, training or execution feedback. Preserve useful parent computation where
possible, but success takes priority over few changed weights.

The 171,664-parameter 679→128 ReLU→five128 tanh→16 linear network is fixed.
All fourteen tensors are accessible. Import the same rounded parent into FP32;
native edits use NumPy FP32 updates, WSPL offers matching `native_numpy_v1` or
exact-to-storage arithmetic/loops. Both deploy through the same WSL NumPy FP32
inference and FP32 Kinetix physics. Action sampling uses FP64 probabilities.
No feature adapter, circuit placement or graph augmentation. Numeric profile is
recorded: this compares interface packages, not syntax alone. Formal tanh proofs
are unsupported. The completed float64 replay is contextual evidence only.

Both reject malformed proposals in full. Zero success is assigned to invalid
proposals for the failure-adjusted arm mean, not as an observed simulator result.
A collection/runtime error stops the study without automatic retry; unattempted
proposals are not assigned zero. Every candidate is final; no best-of selection.

## Fixed evaluation and practical progression

First evaluate an untouched FP32 parent control; then 1,024 episodes per valid
proposal, using identical action seeds `10100000+i`. Identical fixed initial
level state each episode, max256 calls/frame skip2, all six categorical action
entries sampled. Level, physics and reward settings match the completed replay.
These new seeds differ from that replay. Maximum panel:7,168 episodes.
If the FP32 parent succeeds in more than10% of episodes, stop before authoring
and mark the repair setting unresolved: the original strong failure condition
was not retained. No automatic adjustment or replacement panel follows.

Primary descriptive endpoint: mean proposal success fraction, failure-adjusted,
per arm. Retain all per-proposal rates, returns, lengths, invalid/termination
counts, artifacts and paired differences. Between-author variation is distinct
from episode sampling uncertainty. Report native and WSPL usage/time, source
bytes, numeric profiles, changed parameters and construction resources.

Prospective progression rule: native mean success≥0.5; every WSPL proposal valid;
WSPL mean≥0.5 and ≥90% of native mean. Native below floor is unresolved; an
otherwise failing WSPL result is no-go for this configuration. These thresholds
are practical feasibility choices, not statistical noninferiority/superiority
or commercial acceptance. A successful fixed-scene test does not prove transfer.

## Collection and resource budget

Existing ChatGPT Codex login, explicitly `gpt-6-astra`, xhigh. No API-key fallback
or hard dollar/output-token cap. Both prompts are about580KB. Local o200k_base
estimates are approximately352k tokens each; this tokenizer is an estimate, not
an authoritative Astra count. The released request reported355,201 input tokens.
Six requests therefore imply roughly2.1million uncached input tokens plus
reasoning/output; record actual usage/caching separately. Configuration explicitly
requests872,000 context and an800,000 compaction threshold from the frozen model
catalog's supported range. Any compaction/tool/error event invalidates collection.
Provider acceptance of the larger window is not established by offline checks.

20-minute invocation guards, six-hour overall guard, two-hour per-panel worker
guard. Expected total several hours, depending on generated behavior and provider
latency. CPU evaluation; no local GPU or optimizer. The question tool remains
advertised by this CLI version; any tool-use event stops collection as before.

Official [non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode)
describes `codex exec -` for the full stdin prompt. Local CLI help confirms
`debug prompt-input` accepts only a prompt argument, so a580KB prompt cannot use
that Windows argument path. Offline checks audit context using a SHA-256 sentinel;
they do not claim to render or verify the server's receipt of the full prompt.
Collection sends the full UTF-8 bytes via stdin with a guarded writer thread and
records length/hash. A synthetic large-payload fixture verifies complete local
transport. Both arms have identical transport and limits.

## Artifacts and launch boundary

Archive prompts, replies/events, context previews and digest bindings, usage,
source text, immutable edit plans, candidate artifacts and exact deployed NPZs.
Each panel stores every episode and its first three full observation/action
trajectories. One final artifact per candidate, no redundant checkpoints.
Plans and native-to-WSPL normalization are post-hoc archives, never proposer input.
No score/checker feedback is sent to the author. Partial evaluation is labelled
incomplete. `control.json` with `{"stop":true}` requests a clean stop.

Windows host orchestrates source-gated construction/collection; WSL workers
require the same verified receipt, independently registered digest and unchanged
source/runtime/weight identities. They cannot launch a standalone unapproved
panel. Fixed artifacts and a frozen synthetic parity panel check adapter wiring;
these finite checks are not formal floating-point equivalence proofs.

Constructor `skillforge.wsp_kinetix.KinetixRepair`; host runner
`experiments/run_wsp_kinetix.py::run`; worker `experiments/kinetix_author_eval.py`.
Prior approved checkouts remain unchanged. No follow-on experiment is automatic.
