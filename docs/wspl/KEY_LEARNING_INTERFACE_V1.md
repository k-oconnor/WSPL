# Choose regularization for a fixed attention-key learner

Return exactly one JSON object with these fields (a fenced json block is also
accepted). You select a bounded recipe; you do not supply code or new objectives.
Four proposals total. Each valid proposal fits the SAME64-number key from the
same original initialization on256fixed acquisition examples for1000Adam steps,
then receives measured development results on128different examples. Invalid JSON
or out-of-range fields consumes a proposal; no automatic retry. No tools available.

Defaults and allowed ranges:
positive_target:0.75 [0.5,0.95]
negative_target:0.02 [0.001,0.1]
negative_weight:3 [1,100]
retention_weight:3 [0,100]
ridge:0.001 [0,0.1]
worst_negative:1 [0,100]
head_variance:0 [0,10]
anchor:"initial" (or "zero")

Frozen SmolLM2-135M-Instruct, layer26/KVgroup0, shared by three query heads.
A fixed value promotes conduction. Only a64-dimensional key is learned. Cached
original queries Q and native logpartitions Z define m=sigmoid(Q@key/8-Z).
The real LM receives (1-m)*native_head_output + m*fixed_value at its final question
prefix position. No answer text, labels or keyword matcher enter inference.

Inner objective, on labeled training examples only:
- Binary cross entropy with logits, averaged over three heads, with target
  positive_target for conduction and negative_target otherwise.
- Example weights:1for positives;negative_weight for negatives;add retention_weight
  for negative examples the base originally answers correctly. Normalize weighted
  BCE by total example weight.
- Add ridge*mean((key-anchor)^2).
- Add worst_negative*mean(max_over_heads(m_negative)^2).
- Add head_variance*mean(variance_over_heads(m)).
The last two penalties address high individual-head leakage and unequal delivery.
Retention weighting is a PROXY for answer preservation, not the actual LM loss.

Optimizer fixed:CPU FP64 full-batch Adam,LR0.03,1000steps,gradient clip1,project
to original allowed norm after every update,FP32final key. Base weights/value
frozen. No fitting through the LM; gradients run through cached attention only.

Development reward is evaluator-owned and cannot be changed:
positive_accuracy_gain -2*damage_rate_on_initially_correct_negatives
 -0.5*negative_mean_attention_mass.
No eligible correct negatives means damage term0and denominator explicitly0.
You receive reward components and accuracy/routing/retention metrics, not final
questions or predictions. Highest development reward selects the recipe; earliest
tie wins. No valid recipe falls back to the published default.

The chosen recipe is then frozen and the learner is restarted at training sizes
16,64,256,1024. Three nested, balanced subset seeds at smaller sizes; full1024fit
once. An unseen final panel evaluates all keys only after ALL keys are frozen.
You must not adjust a recipe after final evaluation. Your task is regularization
selection within this family, not unrestricted optimizer/objective invention.
