# Native repair of a partially learned movement adapter

Three arms start from the exact completed movement adapter: unchanged;1,024
additional AdamW updates; one Astra/xhigh native Python repair session, at most
six responses. The copy-trained base stays frozen. All arms retain the same
37,472-parameter layer27 ReLU architecture. This is native weight repair, not a
WSPL-interface comparison or capability construction from scratch.

Common acquisition data:64 direction/color-balanced original movement TRAINING
cases and32 copy-training cases. Both repair arms may use these labels; Astra
also receives native teacher-forced activations, digit logits and digit-head rows,
plus unrestricted generation feedback. No gradients or full base weights supplied.
NumPy numerical fitting is allowed. No compute/query matching claim: report
gradient training budget, Astra calls/tokens, construction time and feedback use.

Gradient arm restores the parent adapter and AdamW history, then trains1,024
updates at fixed.001,zero decay,clip1,FP32. Each batch samples three of64movement
and one of32copy cases with replacement,seed431. Same answer-only cross entropy.
Its result is frozen before authoring and never shown to Astra. Astra starts from
the original parent, receives at most six responses including inspections/FINAL,
and retains the last accepted state. Four editable tensors; each norm bounded by
10*max(1,parent norm). No gradient control norm constraint; disclose this asymmetry.
These guards prevent extreme proposals, not equivalent optimization budgets.

Freeze both repair artifacts before final predictions on128 previously unscored
L inputs,64 new larger-shape inputs (three-cell bars and2x2 squares), and64 new
random copy grids. Every final input is disjoint from all earlier movement-study
inputs and copy-foundation training/evaluation data. Directions/colors balanced
within both movement panels. Original simple-shape domain was exhausted, so there
is no fresh simple-shape panel. L orientations were absent from gradient training
and acquisition, but previous aggregate L results informed this study's motivation.
These are new examples within a known synthetic generator, not an ARC benchmark.

Primary metric is exact16-digit unrestricted generation. No palette filtering,
trimming, per-case editing or retries. Three arms x256 examples =768 final grids.
Report each panel, paired corrections/regressions and all failures. Predeclared
descriptive repair signal: Astra gains at least16/128 on freshL and8/64 on larger
versus unchanged, with at most3 copy regressions and at least61/64 copy successes.
Compare with gradient continuation separately; meeting this threshold does not
by itself imply superiority to training. One seed/session, no significance claim.

Archive all author prompts/replies/programs, diagnostics, accepted/rejected edits,
compact final tensors and every evaluation output. Gradient rolling checkpoint
every30minutes plus final/clean-stop; no full-model checkpoint duplication.
20-minute invocation/four-hour overall guard, no automatic retries or follow-on.
Expected30–90minutes. Existing Codex account usage, no hard dollar/output-token
cap or API-key fallback. GPU local training/inference, CPU sandbox construction.
No test predictions or author calls during preparation.
