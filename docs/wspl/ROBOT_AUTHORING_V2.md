# Matched robot authoring: collector repair and continuation

Prepared November 24, 2026; not approved, registered or launched. This document
changes only transport and reuse of the completed prefix of ROBOT_AUTHORING_V1.md.
The task, model, interfaces, author order, development/final missions, deployment
checks and exploratory criteria remain the same. A new exact digest is required.

## Failure and repair

The approved v1 run stopped after718.0317004seconds at the first native author
invocation. Codex emitted an error stating that Code Mode was unavailable because
the Code Mode host was disabled. The collector had disabled that required host
while suppressing tools. Offline prompt rendering accepted the configuration and
did not test runtime host initialization. This was a collector configuration
failure, not a failed learned adapter or scientific comparison.

The run completed16/16 supplied-teacher acquisition missions (255decisions), all16
parent development missions, and384 gradient-control updates/3072targets. The
final control agrees bit-for-bit with its rolling checkpoint. Source/resource
identities and198 saved artifact files were verified read-only. One author CLI
invocation was attempted; no author message or completed turn was recorded. The
answer file is empty. Backend/token usage for that failed invocation is unknown.
No final mission was opened. The original source, logs and output are preserved.

The replacement enables `features.code_mode_host=true`, retains all native-tool
restrictions, and continues to reject tool/error/unknown events. Runtime errors
now retain their actual message. Both offline prompts still expose zero tools.
One generic live healthcheck returned `WSPL_TRANSPORT_OK`, with zero tool calls,
3132 input tokens,9 output tokens and0 reasoning-output tokens. That engineering
check supplied no research data and did not request an adapter or task prediction.
Its usage is recorded separately, not hidden among scientific author draws.

The installed Code Mode host already existed; no upgrade, model substitution,
API-key fallback or user-global configuration change was needed. The full local
CLI binary/resource directory is now bound into the candidate manifest, including
`codex-code-mode-host.exe`.

## Reuse and accounting

The new runner verifies every saved-prefix file against the recovery inventory.
It copies byte-identical common inputs to the new output and reuses the original
parent development records and frozen gradient-control NPZ. It does not recollect
missions, recompute diagnostics, rerun baseline development, or retrain the control.
The rolling optimizer checkpoint remains in the original archive. Final evaluation
still uses that exact frozen control alongside the original parent and six authors.

Reused gradient NPZ SHA256:
`8193f2a9f29636e62525fa49f21131eaa17071fdd3d5c5c7d18c1682e6c20ba5`.
Reused gradient adapter identity:
`b8b49c387ed14e60808cdda0760ca5cc69bda9f27a275a04d75a67c2e288e451`.

All six author sessions start fresh with the same16-response/four-evaluation
allowances: at most96 new author CLI invocations, additional to the single failed
v1 invocation. This preserves matched author budgets; the aborted invocation and
unknown usage remain disclosed. No old private reasoning or failed turn resumes.

The measured693.1130339seconds of completed v1 development execution count toward
the original8hour execution budget. The new development wall allowance is85681
seconds (24hours minus the rounded-up719seconds from v1); repair downtime is
excluded. Final guard6hours,20minutes/author response,300seconds/Python call,
180seconds/reconstruction, and all memory/process limits remain unchanged.
Existing Codex usage; no hard dollar/output-token cap. No automatic retry or
follow-on follows this continuation. Maximum final episodes remain512.

## Verification and launch boundary

55 focused checks pass, including host-enabled configuration, preserved runtime
error messages and a fake continuation that proves no acquisition, diagnostic
forward or gradient-fit function executes. It also proves final evaluation waits
until all six new author outcomes freeze. A separate read-only fixture verifies
and copies the actual saved prefix without constructing a model or scoring tasks.
The previously completed full-size numerical and AppContainer fixtures remain
applicable: those implementations and runtime files are unchanged.

The continuation requires user approval of its new source digest and separate
registration. The prior approval does not authorize this new source identity.
No scientific continuation has launched during repair.
