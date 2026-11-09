# What the joint pilot actually programmed

Read-only investigation of archived programs, tensors, activation diagnostics and
predictions. No new model forwards, benchmark evaluations, author calls or fitting.
Executable audit: `tools/audit_joint_attention_v1.py`; numerical evidence:
`joint_attention_v1_investigation.json`. Original source identity reverified.

## A small structured edit, not arbitrary matrix changes

Astra's final construction is the sum of two outer products:

`delta_W = scale * (repair_direction @ repair_gate.T + support_direction @ support_gate.T)`

The code normalizes the two terms and the sum, stores FP32, then makes a final
norm adjustment. An independent reconstruction of those equations reproduces the
saved W27 bit-for-bit. Top singular values of the stored delta are 5.66713 and
1.41467; the third is 6.89e-7. The first two account for more than 99.9999999999%
of squared Frobenius norm. The remaining tail has norm 4.20e-6, consistent with
the recorded storage rounding. Its generating factors are rank two; the stored
matrix need not be exactly rank two.

This is structurally compatible with a rank-two additive adapter. Such factors
would contain 2,304 coefficients versus 331,776 matrix entries. No adapter runtime
or factorized deployment has been tested; this run used the dense stored edit.
It was constructed through supervised analytic linear algebra and proposal
feedback, not gradient descent or a zero-shot discovery.

## The two intended computations

The repair direction came from the difference between the original and ASP-affected
acquisition residuals, projected into the supplied answer-embedding subspace.
Its anchors were two questions ASP damaged: water movement through cell membranes
(osmosis) and the cycle affected by fossil-fuel combustion (carbon cycle).
Least squares constructed a readout that responds to those acquisition inputs
while suppressing its response on the known spoon target's diagnostic inputs.

The support direction contrasted conduction acquisition residuals with other
mechanisms. Another least-squares readout was intended to support conduction
while avoiding the two fragile retention cases. These are authoring intentions,
not proof that the factors implement those semantic functions elsewhere.

The stored attention key never changed. The value shrank to approximately 38%
of its original magnitude. Thus this session repaired/attenuated a broadly routed
signal; it did not learn a more selective retrieval key.

## The gains look like broad answer calibration

Across all eight reserved word-answer questions, the weight edit changes relative
log probabilities similarly regardless of the correct mechanism. Taking convection
as the reference removes the common softmax normalizer:

| Answer | Mean WSP-only change relative to convection | Across-question SD |
|---|---:|---:|
| conduction | -1.138 nats | 0.114 |
| radiation | +0.044 nats | 0.100 |
| evaporation | -0.417 nats | 0.084 |

Conduction is suppressed relative to convection on every one of the eight cases,
including the actual conduction questions. That does not rule out contextual
computation, but supports a broad calibration/repair interpretation more strongly
than installation of a new heat-transfer reasoning procedure.

The three WSP-only corrections were already close to the decision boundary:

| Correct answer / description | Original gold margin | WSP only | Combined |
|---|---:|---:|---:|
| convection / circulating warm air | -0.283 | +0.101 | -0.467 |
| convection / circulating hot oil | -0.325 | +0.115 | -0.020 |
| radiation / spacecraft receiving waves | -0.326 | +0.399 | -0.796 |

Positive margin means the gold option outranks all alternatives. The convection
gains largely come from suppressing competing answers, not boosting convection's
own log probability. This is still a real decision change, but the mechanism is
important. Conversely, a correctly answered conduction case's margin shrinks from
2.719 to 1.524 under WSP alone; unchanged accuracy hides that tradeoff.

Final ASP alone raises conduction relative to convection by a mean 1.619 nats
(SD 0.154) across the same eight cases. Combined, the mean increase is 0.483 nats.
The two interventions oppose each other at the answer readout. This is an empirical
pattern, not an exact additive identity for the nonlinear network.

## Some repair transfers, but the stronger capability test is unresolved

On the reserved question about water vapor's state, the base chooses “an invisible
gas” with margin 0.120. Final ASP alone changes it to “a visible gas” (gold margin
-0.019). The combination restores the correct answer (margin 0.032). The weight
edit therefore mitigates a real ASP regression outside its two acquisition anchors.
The margin remains small; this single case is not a general preservation guarantee.

All eight situation-answer predictions stay unchanged across conditions. There
are score changes: one convection situation approaches a margin of -0.003 under
WSP alone, but never flips. These answers involve complete continuations, whereas
ASP is injected only at the prefix boundary. The present evidence does not isolate
the roles of language-model answer priors, length normalization, intervention
placement and semantic limitations. No causal claim about which one prevents
transfer is justified from saved scores alone.

The 4.99% matrix norm bound also does not mean a 4.99% activation perturbation.
On cached original acquisition inputs, the edit's output norm is about 15–47%
of the original projection's output norm. This is local projection arithmetic,
not the change in the full residual stream or in reserved-example representations.

## Most informative next diagnostic

Keep the final artifact frozen. First compare its predictions on a new, larger
balanced word/situation panel with an explicit answer-bias calibration control.
If it beats that control, test the repair and support factors separately to identify
which contribution transfers. Such new model evaluations need a separate approved
source pin. Neither a new evaluation nor a factorized deployment was launched here.

The current evidence supports an LLM-authored, low-rank neural repair with modest
transfer. It does not yet establish reliable concept acquisition, an advantage
from combining WSP and ASP, or general ARC improvement.
