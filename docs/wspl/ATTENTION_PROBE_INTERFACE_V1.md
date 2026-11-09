# Native attention authoring interface

You are investigating a frozen SmolLM2-135M-Instruct model's internal interface.
The supplied ARC question and correct answer are visible supervision. Correcting
this one answer is controllability, not evidence of acquiring a general skill.
Final evaluation includes undisclosed paraphrases, other heat-transfer mechanisms,
and unrelated questions. Do not simply promote the correct word everywhere.

Return one fenced Python program (NumPy available as `np`) or exactly FINAL.
The program runs on CPU with no model, gradients, data files, network or GPU.
`load(name)` returns a copy; `save(name, array)` saves FP32 of the same shape.
`NAMES` and `SHAPES` list the bank. `print` is allowed (12,000-character feedback).
Inspection consumes a response. Six responses per stage, no automatic retries.
Programs construct static tensors; they do not execute during model inference.
No optimization/training loops or numerical gradient estimation are permitted;
ordinary linear algebra including least squares/SVD is allowed. Code is archived
for compliance audit; the sandbox cannot semantically enforce this distinction.

The base has 30 layers, width576, 9 query heads, 3 KV groups, head dimension64.
You may intervene in layers24..29 (bank index0..5). GQA query heads0..2 share
KV group0, heads3..5 group1, heads6..8 group2. Weight matrices use out_in layout.
All evaluation supplies only `Question: ...\nAnswer:`; choices are single-token
continuations scored from that prefix. No candidate answer enters the forward
pass. Intervention applies only to the final prefix position, identically for
every question. No question-ID/text matcher, gold label or answer-position trigger
is present in inference. No base weights change. This is runtime attention
augmentation, not a persistent weight edit or a merged LoRA.

Editable tensors:
- `V`: [6,3,64] virtual values.
- `K`: [6,3,64] virtual keys in post-RoPE coordinates, fixed virtual position0.
- `alpha`: [6,3], entries in[0,1], at most ONE nonzero layer/group.

Forced stage: K must remain zero. Each selected query head's native head output
is replaced by `(1-alpha)*native_output + alpha*V` at the final prefix position.
Choose a layer/group, value and alpha. Other heads and positions are unchanged.
The final accepted state, not the best score, is retained.

Retrieval stage: V and alpha are frozen from forced stage. Only K may change.
The nonzero alpha identifies the enabled group; its numerical magnitude is NOT
used for retrieval. For each of that group's three query heads, append one virtual
slot with logit `q dot K / 8` to the original masked native attention logits.
Its softmax mass m gives `(1-m)*native_output + m*V`. Native positions/RoPE are
unchanged. If forced alpha is allzero, retrieval stays disabled.
This implementation uses an exactly equivalent mixture formula (subject to FP32
rounding), without modifying the persistent KV-cache storage API.

For each slot, ||K|| and ||V|| must each be <=10 times the corresponding native
mean norm supplied as K_scale/V_scale. These are engineering magnitude bounds,
not scientific fidelity guarantees. Unused slots remain counted and archived.

Read-only tensors captured from the ORIGINAL target prefix:
- Q [6,9,64]: final-position queries, post-RoPE.
- native_K / native_V [6,3,prefix_length,64]: keys/values for every prefix position.
- attention [6,9,prefix_length]: original final-position attention probabilities.
- O [6,576,576]: attention output projection weights.
- answer_embeddings [4,576]: tied output embedding rows in target choice order.
- final_norm [576]: final RMSNorm weights.
- K_scale / V_scale [6,3]: mean native vector norm over prefix positions.
No read-only tensor may change. Diagnostics are not refreshed after intervention.

After each accepted program you receive target choice log probabilities, gold
margin, correctness and actual virtual attention mass. Failed proposals retain
the previous state. All proposals, including inspections/no-ops, consume a slot.
The next stage starts a fresh visible transcript and receives the saved bank.
