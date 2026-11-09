# SmolLM ARC adaptation task

Improve the supplied frozen SmolLM2-135M-Instruct model on ARC-Challenge
multiple-choice science questions. You author neural-weight changes, not answers
to test questions. No training, fitting, optimizer updates or programmatic search.
Do not attempt to access external files, network, hidden data or tools. Return
only the selected response format. Your program is executed separately.

The unchanged model has thirty Llama-style decoder blocks, hidden width576,
intermediate width1536, nine attention heads, three KV heads, SiLU MLPs and tied
input/output embeddings. Editable targets are only the four576x576 attention
output projections in layers26,27,28,29, using out_in layout: y = W x.
All other model tensors, tokenizer, prompts and scoring remain unchanged.
Do not presume coordinates correspond to particular semantic concepts.

The packet contains32 training examples, untouched-model validation accuracy,
and detailed predictions for a fixed first16 validation examples. Every accepted
edit receives aggregate accuracy on all299 Challenge validation examples and
the same16 detailed examples. No held-out test or retention outcomes are shown
during authoring. No gradients or activation probes are supplied.

The evaluator uses `Question: {question}\nAnswer:` and scores each complete
answer text preceded by a space. It does not use a chat template or display
option labels. Prefix and answer are separately tokenized without special tokens.
Primary prediction maximizes mean answer-token log likelihood; summed likelihood
accuracy is also reported. No answer generation or truncation. Improve the
declared primary while avoiding destructive changes to general capabilities.

## Equal tensor access

All arms have immutable original matrices P26,P27,P28,P29, with shape[576,576].
Direct native/WSPL arms also have W26,W27,W28,W29, initialized from these originals.
The LoRA arm instead has A26..A29 with shape[8,576] and B26..B29 with shape[576,8].
Both factors start zero. Effective deployed weights are P + B @ A with scale1.
No base matrices can change in the LoRA arm. P tensors must remain bit-identical
in every arm. Do not change any shape or introduce new tensors.

Each effective update must satisfy ||W-P||_F <= 0.1 ||P||_F per matrix, evaluated
in float64 from the effective FP32 weights. LoRA composition is performed in
FP32; it is merged into the model for scoring. This is a bound, not a promise
of behavioral preservation. NaNs, infinities, illegal scope/shape or over-budget
edits are rejected, retaining the previous accepted state.

You have at most FOUR responses. A response can be an inspection, one program,
or exactly `FINAL` to keep the latest accepted state. Inspections and rejected
programs consume a response. Programs edit the latest accepted state. The final
state is the latest accepted state, NOT a validation-best checkpoint. There is
no automatic rollback when a valid edit lowers the validation score.

All calls receive the full visible transcript; hidden reasoning is not resumed.

## Inspection response, identical for all arms

Return INSPECT on the first line followed by a JSON list, for example:

    INSPECT
    [{"tensor":"P26","row_start":0,"row_count":2,"col_start":0,"col_count":4}]

At most16 rectangular requests and512 returned coefficients per response. Indices
must be nonnegative integers and within the tensor shape. Initial tensor means,
standard deviations, extrema and nonzero counts are supplied in the packet.
Only this interface returns tensor values; native printed stdout is archived
but suppressed from feedback. Construction errors are bounded diagnostics.

## Native arm

Return one fenced python code block. Available: np (NumPy), NAMES, SHAPES,
load(name) returning a copy, save(name,array) replacing one tensor, saved()
returning nonzero counts. No Torch, model evaluator, training data, host files,
network or GPU. You may use NumPy calculations to construct an edit; no fitting
or programmatic search. Unchanged tensors persist automatically. P arrays are
readable but must not change. Returned tensors must be finite FP32 with exact
shapes. The harness converts save inputs to FP32.

## WSPL and LoRA arms

Return one fenced wspl code block using this structure:

    wspl 0.2;
    target base=artifact("parent",sha256="{{PARENT_SHA256}}");
    edit base as candidate mode=fixed numeric=exact_to_storage {
      // tensor assignments here
    }

The placeholder is replaced by the exact current tensor-bank identity. Supported
forms include `const R=8;`, integer `for i in range(8) { ... }`, tensor indexing,
basic slices, assignments and +=/-=, rational scalar arithmetic and `parent.W26`
references (or parent.A26 in the LoRA arm). Parent references read the immutable
start of this transaction. P tensors always identify the original base model.
Helpers include copy, transpose, reshape, zeros, full, eye, f32bits and tensor.
No imports, arbitrary Python, function definitions, gradients or policy execution.
The available target is an opaque tensor bank; do not use graph growth or
circuit placement here. Exact construction arithmetic is rounded to stored FP32.

Limits:3M tensor elements,100M charged interpreter operations,100k statements,
100k loop iterations;100KB response,800KB serialized prompt. Python and WSPL
have different runtimes and resource accounting, which will be reported.
Construction wall limits are180seconds externally (native internal150seconds),
and native output files are capped at32MiB. No tool calls in your response.
