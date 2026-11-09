# Frozen attention transfer to ARC validation

Prepared November 5, 2026. User requested progressing toward ARC correction.
This is a diagnostic of existing artifacts, with no author calls or training.

## Question and comparisons

Does the fixed conduction intervention improve real ARC answers, including
sentence-valued choices, without damaging unrelated answers? Compare six fixed
conditions: untouched model; original forced value at alpha .75; original broad
key; previous 13-example ridge key; fixed-grid 1,024-example key; Astra-selected
1,024-example key. The two latter artifacts are the full-data endpoints, not
post-hoc selections of the highest-scoring learning-curve subsets.

Use the original 299 ARC-Challenge and 570 ARC-Easy validation rows. Every row is
retained, for 5,214 row-condition evaluations. These validation sets have already
been used in prior adaptation studies; this is NOT a fresh held-out confirmation.
Do not open or score ARC test files. No example-based candidate selection or tuning.

Report full-set normalized and raw answer likelihood accuracy, corrected/broken
IDs, and a predeclared broad lexical heat slice matching conduction, convection,
radiation, evaporation, heat transfer, or thermal energy in question/choices.
This lexical slice is not a semantic applicability label. Preserve all row scores
and per-choice attention masses, including failures and unrelated questions.
The original spoon question is a known target replay, not transfer evidence.

## Scoring and intervention

Keep the existing question/answer format, separate tokenization and token-mean
likelihood primary score. Score every full candidate continuation under teacher
forcing, with no cache. Insert the saved intervention only at the final question
prefix position, layer 26 / KV group 0. Future answer tokens are causally masked
from the routing query. Later answer positions receive only ordinary downstream
effects; do not reinject the slot at each answer token. Record raw likelihood too.
The single-token case agrees with the former prefix-only scorer within numerical
tolerance. Disabled scoring agrees with the ARC baseline within tolerance.

This tests the existing intervention's reach, not a newly acquired science skill.
A positive result on word-valued answers alone is weaker evidence than correction
of sentence-valued options. Inspect both with archived outputs after completion.
No promise of a gain is made; a failure distinguishes delivery limits from simple
answer-word bias and informs the next authoring experiment.

## Runtime and artifacts

Frozen 134.5M SmolLM2-135M-Instruct, FP32 eager GPU inference; no gradients, model
calls, downloads or new checkpoint copies. Six artifacts bound before scoring;
base hash and resource hashes checked before/after. Archive all records, summaries,
source receipt, artifact identities and status. Two-hour guard; expected 10–30
minutes depending on answer lengths. No automatic retry or follow-on study.

Synthetic checks cover explicit-position locality/causality, answer-independent
routing, disabled/single-token scoring parity, restoration, all six runner arms,
artifact freeze before evaluation and correction/retention accounting. Exact
source-manifest approval remains required before scoring ARC.
