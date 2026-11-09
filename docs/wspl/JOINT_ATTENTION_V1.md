# Joint WSP + ASP narrow feasibility pilot

One Astra/xhigh native-Python session, six responses maximum. No gradient training.
Start from original successful layer26/group0 conduction key/value, unchanged
SmolLM2-135M-Instruct weights. Author may edit its two64-number vectors and one
downstream576x576 attention output matrix (layer27), bounded to5% Frobenius change.
The rest of the model remains frozen. This is native weight-space programming,
not a WSPL-syntax comparison or LoRA experiment.

Acquisition: eight designer questions (four mechanisms, word/situation formats),
the known ARC spoon target, and twelve previously correct unrelated ARC validation
cases. Reserved: sixteen designer questions from two new settings and twelve other
known-correct ARC validation cases. Retention IDs are selected deterministically
from saved baseline records, excluding the broad heat lexical slice. Both sets
are disjoint by ID/full question-and-choice record. Situation-question stems repeat
across settings while their answer descriptions differ. This is correlated designer data plus reused ARC
validation, not independent natural science questions or a sealed ARC benchmark.

The author receives baseline and initial-ASP acquisition scores, original weights,
acquisition-only prefix activation diagnostics before/after initial ASP, and tied
output embeddings for acquisition answer tokens. Each valid program is evaluated
on acquisition examples. Last accepted artifact is frozen. Invalid programs and
inspections consume response slots. No score-based best-state selection by harness.

Only after freeze, open reserved examples. Evaluate unchanged, initial ASP,
final combined, final WSP-only and final ASP-only. Component removal is a frozen
ablation, not a retrained matched control. Report each group, all corrected/broken
IDs, raw/normalized scores, attention masses and proposal usage. A useful signal
is a reserved conduction correction relative to unchanged with retained negative
and unrelated answers. A stronger joint-specific signal is a corrected example
that neither final component alone solves. Report all outcomes without selecting
a winning subset; no automatic scaling or follow-on experiment.

Full-answer teacher forcing injects only at the last question-prefix position.
Causal masks prevent candidate answer tokens from affecting routing. W27 changes
apply at all positions. Record base identities and restore exactly after every
candidate evaluation. Frozen keys/values/weights remain native FP32; no distillation
or optimization. Program restrictions on iterative fitting are instruction-level,
with archived code for audit; the sandbox enforces resource/access boundaries.

Archive prompts/replies, diagnostics, each accepted bank and feedback, frozen final
bank and all reserved scores. Banks are small compared with whole-model checkpoints;
do not save full model copies. CPU sandbox construction; FP32 GPU evaluation.
Existing Codex usage, no enforceable dollar/output-token cap or API-key fallback.
Twenty-minute call guard, four-hour study guard, no automatic retry. Estimated
30–90 minutes. Exact source approval required before calls or scientific scoring.
