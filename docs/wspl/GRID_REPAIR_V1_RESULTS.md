# Adapter repair: small improvements, transfer criterion not met

Run and source receipt completed in1,573.969seconds (26.23minutes). Six Astra
responses: five accepted construction/inspection programs, then FINAL. Only
two programs actually changed weights; three retained the previous tensors.

| Fresh evaluation panel | Unchanged adapter | Additional AdamW | Astra repair |
|---|---:|---:|---:|
| L shapes | 3/128 | 5/128 | 4/128 |
| Larger shapes | 3/64 | 4/64 | 4/64 |
| Copy retention | 63/64 | 64/64 | 64/64 |

All figures are exact unrestricted16-token generations. Astra's movement total
is8/192 versus6/192 unchanged and9/192 continued training. Astra corrects three
movement cases and breaks one previously correct case; it repairs one copying
case without breaking another. The predeclared repair signal fails: gains are
one per movement panel, below16 fresh-L and8 larger-shape thresholds. One session
does not support a reliable advantage or general domain-capability claim.

On acquisition examples, Astra improves movement28/64 to34/64 and preserves
copying32/32. It uses NumPy supervised surrogate fitting over supplied native
activations, logits and digit-head weights. This is not gradient-free analytic
weight invention: programs implement numerical optimization of a surrogate,
although they never backpropagate through or access the full language model.
Changes occur in turns1 and3 (zero-indexed). Turn1 changes17,856 read weights;
turn3 changes576 read weights,one read bias,576 write weights and576 write biases.
Programs in other turns inspect or reject their own proposed changes.

Author usage:660,937 input tokens, zero cached;24,852 output tokens including
16,037 reasoning tokens. Five program sources total28,372bytes. The gradient
control used1,024 additional full-model-backpropagation adapter updates; budgets
are not matched. Full prompts, replies, programs, diagnostics and outputs remain
in the isolated run archive.

Read-only audit verified archived source, all768 final rows against reserved
fixtures and exact scores, paired corrections/regressions, frozen artifact hashes,
final author weights matching the last accepted candidate, and usage/edit counts.
No new predictions or fitting during audit. Companion summary:
grid_repair_v1_results_summary.json.

Interpretation: Astra can make modest useful edits here, including complete
copy retention on this panel, but neither repair approach recovers dependable
movement transfer. The result does not identify whether representation, adapter
placement/capacity, optimization budget or training distribution is responsible.
Do not scale authoring or claim an adapter repair breakthrough from this pilot.
No follow-on run has launched.
