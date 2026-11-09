# One-question native attention diagnostic

Prepared November 5, 2026; requires exact source approval. No WSPL arm, gradients
or training. One authoring trajectory with two stages, Astra/xhigh via existing
Codex account: six responses per stage, at most12 calls. Twenty-minute call guard,
four-hour overall guard, no automatic retry. Expected30-90minutes, uncertain.
No enforceable dollar/output-token cap; no API-key fallback.

## Question and evidence boundary

ARC-Challenge validation Mercury_SC_408700: hot tea warms a spoon; mechanism
conduction. The archived base predicts evaporation. This question was manually
selected after inspecting failures involving heat/freezing; it is not a randomly
sampled or held-out test. Its four answer texts each tokenize to one token.
Author sees question/gold and repeated feedback on this question only.

We distinguish forced message delivery from contextual retrieval. Stage1 constructs
one static value and a forced mixing strength in one of18 available layer/KV groups.
Stage2 freezes that artifact and authors a key for ordinary softmax competition.
Both use native NumPy construction and bounded read-only internal diagnostics.
See ATTENTION_PROBE_INTERFACE_V1.md for exact semantics. Restriction to last six
layers and one KV group is a scoped test, not an exhaustive test of Astra.

This is KV-like runtime attention augmentation: no persistent weight modification,
cache API mutation, module execution, Q/K projection editing or general skill
installation. Keys are fixed post-RoPE vectors. No added token shifts native
positions. Only the final question-prefix position is affected, identically on
all examples; no answer text, correct label or question matcher enters inference.
The three query heads sharing a KV group receive the same value/key.

## Evaluation

Model/scoring inherited from ARC baseline: frozen FP32 SmolLM2-135M-Instruct,
eager attention, TF32off, zero-shot question/answer prefix, next-token likelihood
over the four supplied single-token choices. Reproduce archived failure within
2e-5 logprob before authoring; disabled intervention must match exactly. Abort
before authoring if either check fails. Weights hashed before/after.

Freeze both stage-final artifacts before opening evaluation panel to the runner.
No best-of-six selection; all accepted steps archived, final accepted retained.
Evaluate original +12 manually authored questions: four conduction paraphrases,
six controls (two each convection/radiation/evaporation), two unrelated questions
with the same answer words. All panel questions have the same four choices;
also evaluate a reversed-choice copy of every row (including target), with labels
remapped. The score is independent answer likelihood, so choice-order invariance
is an implementation check, not semantic evidence. Panel created before calls,
never exposed to author; designer-visible and not a standardized benchmark.

Variants: untouched, forced stage final, retrieval stage final, retrieval with
zeroed key. Report per-question logprobs/margins, actual attention mass, correct
counts by category, newly corrected/broken examples vs base, and choice-order
parity. Zero-key retains V and ordinary softmax normalization; it is NOT a no-op.
Disabled injection is the no-op control. No fixed universal success-rate claim.

Interpretation: original flipped = controllability; transfer to paraphrases with
different-answer control preservation = preliminary integration evidence. Higher
retrieval mass alone is not useful computation. Even perfect results cannot
distinguish installed science reasoning from a context-sensitive answer rule.
This is an exploratory one-case, one-author trajectory with disclosed answer,
not a general capability result or interface comparison.

## Integrity and artifacts

Native code executes in existing bubblewrap NumPy sandbox; bounded bank only,
no model/data/network/GPU. Native stdout deliberately enabled for inspecting
provided tensors. Host validates all shapes/finite values/read-only arrays and
frozen stage2 value/group. Programs cannot run at inference. No training loops
allowed; archived program compliance must be audited before claiming no fitting.
Every prompt/reply/usage/construction/program/bank and feedback retained, plus
both final banks, panel predictions, hashes and exact source/runtime identities.
CPU construction, local CUDA inference only. No independent follow-on run.
