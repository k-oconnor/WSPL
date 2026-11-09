# ARC baseline and retention reference

Preparation for the user-requested native/WSPL/WSPL-LoRA adaptation comparison.
This first run establishes the untouched model's behavior before spending any
authoring calls. It does not automatically launch the editing screen.

Model: HuggingFaceTB/SmolLM2-135M-Instruct, pinned revision
12fd25f77366fa6b3b4b768ec3050bf629380bac. Local safetensors/tokenizer hashes,
dedicated runtime and every inventoried source file are bound in the manifest.
No remote code or network model loading during the run. FP32 CUDA inference,
eager attention, TF32 disabled, eval mode, no gradients, frozen tied embeddings.

Dataset: allenai/ai2_arc revision 210d026faf9955653af8916fad021475a3f00453.
Only validation files provisioned: ARC-Challenge 299 and ARC-Easy 570 examples.
No hidden test scores or test examples enter this baseline. ARC-Easy supplies a
related-task retention reference; it does not cover generic language retention.

Scoring: zero-shot plain prefix `Question: {question}\nAnswer:` followed by a
leading space plus each complete answer choice. Prefix and continuation are
tokenized separately without automatic special tokens. Sum conditional log
probabilities over answer tokens only; primary score divides by answer token
count, raw sum is a secondary result. Choose first maximum on exact ties.
No chat template, generation, few-shot examples or answer-label bias correction.
This fixed custom scoring protocol is not claimed to reproduce model-card scores.
Maximum 1024 tokens including answer; refuse overlength rather than truncate.
Evaluate all options for a question together with masked right padding.

Archive per-choice summed and normalized log probabilities, token counts, gold
and predicted indices, IDs, correctness and per-example chance rate. Report
aggregate primary/raw accuracy and mean chance on each validation subset.
Do not cherry-pick prompts or normalization based on observed results.

Descriptive authoring-suitability rule, fixed before local scores: Challenge
primary accuracy at least three percentage points above its mean uniform-choice
chance rate and at most 75%. A miss prompts discussion, not an automatic change
of benchmark, model or hidden-test access. No inferential significance claim.

Before/after model tensor hashes must agree. Report peak allocated GPU memory
and elapsed time, with progress per example. One source-gated constructor.
No training, LLM authoring calls, paid API use or new checkpoints. Two-hour wall
guard; clean stop produces incomplete output. Expected minutes on the 3090,
subject to preflight/runtime timings. Exact pin approval required before scoring.

## Prepared adaptation bridge (engineering checks only)

Proposed equal direct-edit scope: o_proj weights in layers26–29, four576x576
matrices =1,327,104 editable coefficients. LoRA scope: rank8 per same matrix,
scale1,36,864 factor coefficients; base copies read-only. Both factors initially
zero because programs author them directly (not a gradient initialization).
Enforce finite FP32 values, exact shapes, unchanged read-only bases, and a common
10% per-matrix Frobenius effective-update budget. This bound is a pilot design
constraint, not a semantic preservation guarantee.

The bridge uses the existing WSPL opaque tensor-bank representation plus an
explicit adapter execution contract. It does not claim a new grammar feature,
whole-transformer formal verification, or that latent factors are interpretable.
Zero-adapter equality, factor-to-effective-weight arithmetic and WSPL round-trip
are engineering tests; no author edits have yet been applied to the real model.

Source pages:
- https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- https://huggingface.co/datasets/allenai/ai2_arc
