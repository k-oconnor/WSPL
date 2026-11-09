# Native key-only contextual retrieval

Program a context-selective key for a FROZEN value intervention. Return one fenced
Python/NumPy program or exactly FINAL. Six responses total, including inspections
and FINAL. `load(name)` returns a copy; `save(name,array)` saves same-shaped FP32.
NAMES/SHAPES list tensors. print is allowed,12000characters returned. CPU sandbox,
no model execution, GPU, external files, network or gradients. The host evaluates
each accepted candidate on the supplied acquisition examples, not the final panel.
Last accepted state is frozen; no automatic best-checkpoint selection or retry.

You may change ONLY K[2,0,:],64 numbers. All other tensors, including other keys,
V, alpha and diagnostics are read-only. Begin with the previously authored key.
Its norm must stay <=10*K_scale[2,0]. Setting it to zero is not disabling injection.
There is no extra bias, nonlinear router, query projection change or text matcher.

Frozen SmolLM2-135M-Instruct:30layers,width576,9queryheads,3KVgroups,head_dim64.
The slot is at layer26,KVgroup0 shared by queryheads0..2. Its value is fixed from
the earlier successful intervention. alpha[2,0]=.75 identifies the slot but is
NOT a retrieval multiplier. Earlier layers and all base weights are unchanged.
Only the final position of `Question: ...\nAnswer:` receives the injected slot.
All four choices are single-token continuations; no answer enters the forward
pass. The same static key/value applies to every question without gold labels.

For head h, native logits are q_h @ native_keys.T /8. The virtual logit is q_h@K/8.
Let logZ_h=logsumexp(native logits). Virtual mass is sigmoid(q_h@K/8-logZ_h).
The head output becomes (1-mass)*native_output+mass*V. Keys/queries are post-RoPE;
the virtual key is fixed in that coordinate system; native positions never shift.

Additional read-only diagnostics, aligned with the supplied example order:
- train_Q [13,3,64]: original final-position query vectors at layer26,heads0..2.
- train_logZ [13,3]: native attention log partition, no virtual slot.
- train_applicable [13]:1for conduction questions,0for other mechanisms/lexical controls.
These were measured without intervention. Since all changes occur after these
queries are computed, they exactly locate this key's attention calculation, up to
rounding. They permit offline mass predictions; runtime feedback is authoritative.

The remaining bank is the original target's saved diagnostics:
Q[6,9,64],native_K/native_V[6,3,35,64],attention[6,9,35],O[6,576,576],
answer_embeddings[4,576],final_norm[576],K_scale/V_scale[6,3]. Layer indices0..5
map to24..29. Those single-question diagnostics are historical, not per-example.

The previous key increased conduction probability broadly. Aim for substantial
mass on conduction examples and low mass on all others, while preserving correct
negative answers. Feedback includes individual masses and choice log probabilities,
positive/negative mean mass, routing AUROC, correctness changes and retention.
Descriptive targets: positive mean mass>=.5,negative mean<=.1,AUROC>=.9; positive
answer improvement and no losses on initially correct negatives. These thresholds
are engineering targets, not significance tests. The final panel has new physical
situations with overlapping vocabulary and is never shown during authoring.

Linear algebra (least squares/SVD, including supervised calibration from the
provided labels) is allowed. No iterative optimizer, gradient training, numerical
gradient estimation or model search loops. This is informed key construction;
do not claim that using labeled examples makes it learning-free or zero-shot.
Code compliance is audited after collection. Author code never executes at inference.
