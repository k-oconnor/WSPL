# Repair one reusable movement adapter

You repair a learned native-state ReLU adapter in frozen SmolLM2-135M, previously
fine-tuned only for copying. Learn a domain-level transformation, not a separate
answer program for each example. Only four tensors are deployed at inference.
No runtime grid solver, lookup table, task router or Python execution is deployed.

The supplied domain rule: a4x4 grid, background0, one connected monochrome object
(1 or2) and marker3. Marker lies outside the object's bounding box, aligned with
its row or column span. Move the object exactly one cell toward the marker,
preserving its shape/color and marker, without collisions or leaving the grid.
Copy tasks instead preserve every cell. The model receives an explicit task
instruction and eight demonstrations. Output is16 row-major color digits.
Reserved cases are unavailable; aim for reusable geometry across shapes/positions.

The adapter acts at layer27 output on EVERY position:
`h_new = h + write.weight @ ReLU(read.weight @ h + read.bias) + write.bias`.
There are30 layers (0-indexed), hidden width576, bottleneck32. Layers28 and29,
final RMS normalization and vocabulary head remain downstream and frozen.
Editable FP32 tensors: read.weight[32,576], read.bias[32], write.weight[576,32],
write.bias[576]. You receive trained values, not zero initialization.
Each tensor norm must remain at most10*max(1,original tensor norm). No architecture
or base edits; all other arrays must remain byte-identical within each proposal.

Read-only acquisition diagnostics, refreshed after each accepted proposal:
- H[N,16,576]: native layer27 output BEFORE the adapter, at positions predicting
  each gold answer digit, using teacher-forced gold prefixes. It is not an
  autoregressive trajectory and its first row predicts digit0 from the prompt.
- digit_logits[N,16,4]: full model logits for digits0,1,2,3 with current adapter.
- gold[N,16]: target color indices, stored as FP32.
- digit_head[4,576]: rows of the frozen vocabulary head. These are not an exact
  local mapping from H to logits because two transformer layers and normalization
  follow the adapter. Do not assume a linear identity between them.
Acquisition examples and unrestricted autoregressive outputs are supplied in the
packet/feedback. No model gradients, Jacobians, test examples, or full base weights.
You may use NumPy linear algebra or numerical fitting on these acquisition arrays;
this is supervised diagnostic-assisted repair, not a zero-shot/gradient-free claim.

Return exactly one fenced python program or FINAL. Up to six responses, including
inspection-only programs and FINAL; last accepted tensors are retained, not best.
The Python environment provides np, NAMES, SHAPES, load(name), save(name,array),
saved(). load returns a copy. Print compact inspection diagnostics as needed.
The worker automatically saves all arrays after the program; absent edits preserve
their values. No filesystem/network/model access, only NumPy and provided arrays.
8GiB memory,120CPU seconds/150wall seconds, bounded output. Every accepted edit
receives the same96 acquisition examples' unrestricted generation scores and
updated diagnostic arrays. Construction failures count against six responses.
Do not use tools or ask questions. No scores or examples from final evaluation.
