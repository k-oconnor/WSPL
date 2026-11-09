# SmolLM adaptation: three-interface benchmark design

Status: design under preparation, not a launch protocol or approved source pin.
User selected SmolLM from Hugging Face and requested a benchmark on which it is
weak, comparing native edits, WSPL edits, and WSPL edits to LoRA adapters.
This supersedes the earlier default proposal of synthetic structured extraction.

## Model and benchmark recommendation

Candidate: HuggingFaceTB/SmolLM2-135M-Instruct, revision
12fd25f77366fa6b3b4b768ec3050bf629380bac. Provisioning is separate from
scientific evaluation. Verify all downloaded files before use.

Recommend ARC-Challenge as the first target, subject to a pinned baseline
feasibility measurement. The publisher reports 37.3% ARC *average* and 1.4%
five-shot GSM8K for this instruction model. These are published results under
their evaluation setup, not local measurements or an ARC-Challenge-only score.
GSM8K risks a floor effect at 135M. IFEval is another candidate for an explicitly
instruction-following skill, but requires its full instruction-specific verifier.

Source: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
Retrieved November 4, 2026. Do not assume local prompt/scoring reproduces that card.

## Required arms

| Arm | Authoring interface | Editable object |
| --- | --- | --- |
| Untouched control | None | Frozen original model |
| Native | Python/NumPy | Declared base-weight tensors |
| WSPL | WSPL fixed-shape edits | Exactly the same base-weight tensors |
| WSPL-LoRA | WSPL factor edits | Low-rank updates attached to the same target matrices |

Use the same frozen base, tokenizer, task examples, visible diagnostics,
evaluation feedback, author model and call budget. Independent author draws
are the replicate; benchmark examples are not author replicates. Begin with
three draws per editing arm if the baseline is suitable. No hidden-test feedback.

Native versus WSPL isolates the authoring interface only if access and execution
capabilities are matched. WSPL versus WSPL-LoRA also changes parameterization
and edit capacity. Report that as an adaptation-method comparison, not pure
syntax superiority. Native LoRA would be a later fourth authoring arm to separate
interface and low-rank effects. Do not substitute gradient training for the
requested native-edit control. A gradient adapter is an optional learnability
reference, not currently required by the user-selected three-arm design.

## Protocol decisions to finalize before pinning

- Explicit target matrices/layers and rank; do not silently allow broader access
  in one direct-edit arm. Preserve embeddings and tied-weight relationships.
- Model information supplied to authors: task packet, readable parent tensors,
  any activation summaries, and diagnostic examples must be symmetric. Large
  weight access needs an actual bounded interface, not an unexplained prompt dump.
- Initial zero effective LoRA update, with declared factor initialization and
  fixed scale; no hidden gradient training to create the authoring starting point.
- Fixed answer scoring and prompts, including length normalization and handling
  of multi-token choices. Report chance rate for variable-size answer sets.
- Use official training data for author-visible examples, validation for bounded
  development feedback, and untouched test for final evaluation after all
  candidates freeze. Public-benchmark pretraining exposure remains unknown.
- Fixed evaluation/authoring budgets and an explicit baseline suitability rule
  before observing scores. Do not choose a weak task by inspecting hidden tests.
- Retention on a separate task and generic language loss, with fixed samples;
  no assumption that an unmodified base guarantees retention with an adapter.
- Record edits, exact weights/factors, effective deltas, failures, usage,
  program lengths, construction costs, inference costs and retention changes.

No model benchmark, authoring call or training is authorized by this document.
Prepare the concrete runner and tests, then request the source approval required
by AGENTS.md. The initial baseline must also use an approved scientific pin.
