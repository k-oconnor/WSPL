# Broader-access robot authoring: completed exploratory results

Completed November24,2026. Original source pin:
`fddea45cf93addfd070f01877b1eb8f096df2bfcf18fc5597a331298f7332d71`.
Usage-limit recovery pin:
`f9beb0084f5d96aff5b67095265d5f6e98f7b3f595c67ec3102cd38dbd2e6363`.

The recovered run exited successfully, with a complete source receipt and512
saved final simulator missions. All eight frozen artifact file hashes and paired
mission identities/counts were verified. Six independent Astra author sessions
received the same original Qwen2.5-Coder-1.5B parent and acquisition data, using
native Python or Python plus WSPL. Both interfaces could inspect chosen layers
and replay candidate edits on acquisition data. No author or construction code
runs at deployment; the motor blocks were supplied.

## Final outcomes

Each tuple is ordinary / changed-from-start / mid-mission switch / return-to-nominal,
with16 missions in each condition:

- Original parent:16 /0 /0 /0; total16/64.
- Reused gradient control:16 /9 /16 /12; total53/64.
- Native0:13 /9 /8 /4; total34/64.
- Native1:16 /15 /15 /14; total60/64.
- Native2:16 /11 /14 /11; total52/64.
- WSPL0:16 /10 /9 /8; total43/64.
- WSPL1:16 /11 /13 /11; total51/64.
- WSPL2:16 /13 /14 /10; total53/64.

The predeclared combined preservation/adaptation criterion passes for2/3 native
and3/3 WSPL authors. Five authors preserve all16 ordinary successes. The best
native result exceeds the gradient control's total; the best WSPL result ties
its total. These totals do not imply identical successful missions.

This is an exploratory screen with three author draws per interface. Compared
with the earlier last-layer-diagnostic screen, the final seeds and response
allowance also changed. It does not isolate the causal effect of broader access.
The gradient control has different optimization/feedback access. No general
WSPL superiority, efficiency advantage, broad retention or hardware transfer
claim follows. Full construction-method audit remains pending.

## Usage-limit interruption and restart

Three authors completed before the usage limit; native1 had21 completed responses
and a saved new candidate. The failed22nd invocation produced no completed answer
or usage record. It consumed a response slot. Recovery preserved the three frozen
authors and restored native1's visible history and files with two responses left,
then ran the two untouched authors. There were141 completed responses overall,
plus one failed invocation of unknown backend usage. No final mission had been
opened before recovery.

Native1's60/64 adapter was constructed BEFORE the interruption. Its resumed
requests were only evaluate(submission5.json), then finish. All392 arrays and
18,464,768 coefficients in the final adapter match the pre-interruption candidate
bit-for-bit. The staged NPZ is byte-identical; the trusted reconstructed NPZ has
different archive bytes but identical tensors. The original42-entry visible
history is unchanged. The restart enabled evaluation/freezing, not new numerical
construction. What an uninterrupted session would have chosen next is unknown.

## Records and source locations

- Protocol: [ROBOT_AUTHORING_ACCESS_V1.md](ROBOT_AUTHORING_ACCESS_V1.md).
- Recovery: [ROBOT_ACCESS_RESUME_V1.md](ROBOT_ACCESS_RESUME_V1.md).
- Final counts: [robot_access_resume_v1_completion.json](robot_access_resume_v1_completion.json).
- Restart proof: [robot_access_restart_audit.json](robot_access_restart_audit.json), reproducible with `tools/audit_robot_access_restart.py` against the archived local artifacts.
- Scientific source: branches `research/robot-authoring-access-v1` and `research/robot-access-resume-v1`; predecessor branches preserve the earlier exact source identities.

Model checkpoints, runtime copies, simulator traces and full author workspaces
remain local and excluded from Git. Historical manifests retain their approved
source/resource hashes and paths; a repository checkout alone does not contain
the bound execution environment. This is archival research evidence, not a new
integrated capability. No follow-on experiment was launched.
