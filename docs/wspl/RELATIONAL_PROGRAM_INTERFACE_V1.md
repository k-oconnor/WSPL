# A single neural program for a physical relation

Author one fixed artifact combining an attention delivery slot at layer26 with
a weight change at layer27. The task is an idealized two-object thermal relation:
objects exchange heat only through their stated conducting connection, never with
their surroundings. With a connection and unequal temperatures, heat initially
flows from hotter to cooler; without the connection or with equal temperatures,
neither temperature initially changes. The requested consequence varies: which
object warms, cools, or changes. Answers are left object, right object, neither
object, or both objects. Do not just favor an answer phrase.

You receive acquisition examples and their baseline/initial-ASP scores, plus
acquisition-only activation diagnostics. Reserved examples use new temperature
values/combinations, altered wording and reversed order of object descriptions;
they include paired changes in contact and temperatures. Unrelated retention
questions are also reserved. No reserved content or scores will be shown.

Runtime hierarchy is explicit: the authored key/value contributes at layer26
through ordinary attention normalization, then the edited layer27 attention output
projection transforms its inputs. Everything thereafter uses the original model.
No separate reasoning service, external router, question-ID matcher, physical
simulator or answer selector executes in the candidate model. The artifacts are
static tensors, not callable author code. A single NeuralProgram execution binds
both interventions for each ordinary model forward pass. The evaluator only scores
continuations; it does not solve the physical relation on the model's behalf.

Six responses maximum, including inspections, rejections and FINAL. Return one
fenced Python program or exactly FINAL. NumPy is `np`; `load(name)` returns a copy;
`save(name,array)` writes same-shape FP32. `NAMES` and `SHAPES` list the bank.
Print bounded diagnostics. CPU sandbox: no model, gradients, data files, network
or GPU. No iterative optimizer/training loops or numerical gradient estimates.
Analytic SVD, least squares and outer products are allowed. Code is archived for
audit; the no-optimizer restriction is not semantically enforced by the sandbox.
Last accepted state is retained, not the best feedback checkpoint.

Editable tensors:
- K and V [6,3,64], ONLY [2,0,:], layer26/KVgroup0. Each norm <=10 times the
  supplied original K_scale/V_scale. Alpha is frozen and enables this one slot.
- W27 [576,576], out_in attention-output matrix at layer27. Delta Frobenius norm
  <=5% of original_W27. This weight applies at every token position.

The initial key/value are the earlier conduction intervention, with original
base weights. You may replace the vectors within bounds; preserving the original
conduction-word direction is not the goal. Alpha's .75 value identifies the slot
but is not used as retrieval strength. For its three GQA heads, mass is
sigmoid(q dot K/8 - logsumexp(native masked logits)). Output becomes
(1-mass)*native + mass*V. K is post-RoPE; native token positions stay unchanged.
Insertion occurs ONLY at the last question-prefix position. It is not repeated
at answer positions. Later computation proceeds normally.

Full candidate continuations are teacher-forced under causal attention and scored
by mean token log likelihood, with raw sums also saved. Format remains
`Question: ...\nAnswer:`. Future answer tokens cannot affect prefix routing.
There is no gold label or task metadata in the model input beyond question text.

Read-only tensors:
- original_W27, final_norm.
- base_layer26_out / asp_layer26_out [N,576].
- base_o27_input / asp_o27_input [N,576].
- base_layer27_out / asp_layer27_out [N,576].
- answer_token_ids [M] and answer_embeddings [M,576]. Packet includes decoded
  token strings. N rows follow acquisition example order; M uses sorted token IDs.
- K_scale / V_scale and alpha.
- acquisition_Q [N,3,64] and acquisition_logZ [N,3]: native layer26/group0
  prefix queries and log normalizers. These support analytic key construction.
  W27 is downstream, so editing it cannot change these queries. Values are
  acquisition-only diagnostics, not an inference-time dataset or lookup table.
Diagnostics are frozen at the original weights with/without initial ASP; they
are not refreshed and are not gradients. Every accepted proposal receives full
acquisition answer scores, corrected/broken IDs, group counts and attention masses.

The combined artifact must improve several consequences of the relation, not just
memorize acquisition outputs. After freeze, the evaluator will disable each
component and compare with a simple constant answer-bias control. No subsequent
author call can use reserved results.
