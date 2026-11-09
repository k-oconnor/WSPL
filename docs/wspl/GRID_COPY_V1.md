# Copy-trained foundation for a domain adapter

The frozen SmolLM grid gate copied only 48/64 palette-constrained grids. Prepare
a new foundation by full-model copy-only fine-tuning, then freeze it before any
movement adapter work. This stage is not an Astra authoring or movement result.

Start from the original FP32 SmolLM2-135M-Instruct, 134,515,008 parameters. Train
all parameters with tied embeddings retained. AdamW LR 0.00002, betas .9/.999,
epsilon 1e-8, zero decay; 32-update linear warmup, then constant rate. Clip norm1.
512 updates, batch4, seed211: 2,048 distinct examples and 32,768 answer targets.
Loss is ordinary full-vocabulary cross entropy on all16 answer digits only;
prompt/demo/padding positions contribute no loss. No movement labels or prompts.

Deterministic pool:4,096 random 4x4 grids, half dense uniform colors0..3, half
sparse with2..6 nonzero cells. All560 original movement-domain inputs are excluded.
Shuffle once and use the first2,048 without replacement. Same eight copy-only
demonstrations and original prompt codec; no leading-space format change.
Prepare256 disjoint fresh random copy holdouts with the same half/half mixture.
Retain the original64 copy cases as an explicitly previously observed diagnostic.
No development selection, early stopping on scores, or best-checkpoint search.

After the fixed budget, freeze the final model before evaluating both panels,
each with palette-only and unrestricted16-token decoding (640 generations).
Both gates must pass: at least244/256 fresh random copies and61/64 historical
copies, palette-only exact-grid accuracy. Report unrestricted performance separately.
This does not establish general-language retention; full fine-tuning may change
other abilities. The resulting foundation is a new experimental base and must
be shared by future unchanged and movement-adapter conditions.

No movement evaluations, movement adapter training, Astra calls or downloads.
No automatic follow-on. GPU FP32 training/inference; two-hour wall guard.
Expected approximately10–30minutes, hardware dependent. One rolling last.pt
every30minutes and final/clean-stop, with weights/optimizer/RNG/source identity;
no separate best or duplicate final checkpoint. A stop during training preserves
the current state but does not claim the gate was evaluated or passed. Resource
hashes verify before and after. Any continuation needs a separately approved run.
