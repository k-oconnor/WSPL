# Kinetix replay: reference acquisition and remaining work

November4,2026. Preparation only: zero simulator episodes, optimizer updates or
model requests. Following the completed reworked Breakout feasibility screen,
Kevin requested moving to Kinetix.

Downloaded14 released files (1,904,907bytes) under the ignored local directory
`data/csail_kinetix_reference_v1`. Upstream commit is
`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`; every byte count, Git blob SHA-1 and
local SHA-256 was verified. [Provenance](../../provenance/csail_kinetix_reference_v1.json).
The duplicate message.txt was omitted because upstream identifies it with the
same blob as prompt.txt. No downloaded code has been executed.

The primary [upstream experiment](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp4_fix_car/README.md)
uses Kinetix level `l/h2_car_ramp`, a171,664-parameter actor with a679-input
ReLU embedding, five128-unit tanh layers and16action logits. It applies the
published edit to the rounded actor shown in the prompt, not the full-precision
trained NPZ. The released report records0/256successes for the parent and
1024/1024for the repair; these are upstream results, not local reproductions.

Required before a replay manifest:

1. Resolve and pin the official simulator, level assets, observation encoding,
   action distribution, versions and runtime. The released repository's generic
   harness directory does not supply the Kinetix evaluator or simulator itself.
2. Parse the exact shown rounded tensors from the577,774-byte prompt. Verify
   shapes and the171,664actor-element count; reconstruct the published edit and
   compare to the released edited_actor.npz. Keep the full-precision PPO/value
   checkpoint distinct. Do not supply the published repair to future writers.
3. Check scalar/network inference and simulator wiring using bounded unit fixtures.
   Freeze paired replay seeds and report differences from upstream RNG procedure
   if the original evaluator cannot be recovered.
4. Prepare the parent/repair replay runner, source manifest and evaluation budget,
   then request the exact source approval required by AGENTS.md.

Fresh-authoring preparation comes after replay: original tensor aliases such as
actor0.W require an explicit importer/name mapping for WSPL identifiers. Preserve
the original nonlinearities; formal tanh proofs remain unsupported. The current
collector's offline preflight passes prompt text as a Windows command argument;
the full577KBprompt exceeds that path's capacity. Implement a verified file/stdin
preflight before any authoring call, without truncating the supplied weights.
The actual collection path already sends prompt text through stdin.

No Kinetix environment has been installed or validated as part of this acquisition.
No replay or fresh repair is yet launch-ready, registered or approved.
