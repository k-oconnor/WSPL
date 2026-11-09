# Native WSP + ASP authoring interface

Construct a static combined weight/attention intervention for a narrow heat-transfer
family in SmolLM2-135M-Instruct. The packet provides acquisition questions, gold
answers, unchanged scores and initial attention-only scores. Improve conduction
answers while preserving other mechanisms and known-correct unrelated answers.
Reserved questions include new situations, full-sentence choices and retention
cases. They are never exposed during authoring. This is an engineering pilot,
not a claim of general ARC performance or a formally verified semantic skill.

Six responses total, including inspections, rejected proposals and FINAL. Return
one fenced Python program or exactly FINAL. NumPy is `np`; `load(name)` returns
a copy; `save(name, array)` writes FP32 with the same shape. `NAMES` and `SHAPES`
list tensors. Print diagnostic values (12,000 character feedback limit). Programs
run in an isolated CPU sandbox, without the model, filesystem data, network or GPU.
Programs construct tensors only; no code runs during inference. No gradients,
iterative optimizer/training loops or numerical gradient estimation. Ordinary
analytic linear algebra, SVD, least squares and outer products are permitted.
This restriction is auditable from code, not semantically enforced by the sandbox.
Last accepted state is frozen, not the best-feedback checkpoint. You can restore
the original weight using original_W27. Preserve every other read-only tensor.

Editable:
- K and V [6,3,64]: ONLY entry [2,0,:], layer26/KVgroup0. Three query heads share
  this virtual slot. Each vector norm <=10*K_scale[2,0] or 10*V_scale[2,0].
- W27 [576,576]: attention output projection at layer27, out_in orientation.
  Frobenius norm of (W27-original_W27) <=5% of original_W27 norm. This weight
  applies at ALL positions, unlike the attention intervention.

Alpha and all other tensors are read-only. Alpha .75 marks the one enabled slot;
its magnitude is NOT used in retrieval. At the final question-prefix query only,
virtual mass m=sigmoid(q dot K/8 - logsumexp(native masked attention logits)).
Replace native head output with (1-m)*native+m*V. Native positions are unchanged.
K uses post-RoPE coordinates; no persistent-cache API mutation. Later answer
positions receive downstream computation, not repeated slot injection.

Inference scores every answer continuation using mean token log likelihood under
`Question: ...\nAnswer:` with separately tokenized leading-space answer text.
Candidate continuations are teacher-forced under causal attention; future answer
tokens cannot influence the question-boundary routing query. Gold labels, IDs and
question text matchers are absent from the intervention. Raw likelihood is saved.

Read-only diagnostics (fixed at original weights, not refreshed):
- original_W27 [576,576], final_norm [576].
- base_layer26_out / asp_layer26_out [N,576]: residual at question boundary.
- base_o27_input / asp_o27_input [N,576]: input to editable projection.
- base_layer27_out / asp_layer27_out [N,576]: downstream layer residual.
- answer_token_ids [M], answer_embeddings [M,576]: sorted unique acquisition
  answer-token IDs and tied output embedding vectors. IDs are exact FP32 integers.
- K_scale / V_scale: original native norm bounds.
N rows follow acquisition packet order. `base` means no ASP; `asp` means original
successful ASP. The reference W27 includes no prior weight edit. These diagnostics
show propagation at the prefix; they are not gradients or a complete causal model.

Each valid proposal gets full acquisition answer scores, corrections, regressions,
group counts and actual attention masses. Rejected proposals retain prior state.
No reserved scores are supplied. No additional tools or external files permitted.
