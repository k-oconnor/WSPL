# Joint WSP + ASP pilot: completed

Run and source receipt complete in 1,583.781 seconds (26.4 minutes). Six accepted
responses: two inspection/no-op programs and four changed proposals. No gradient
training. Astra used analytic SVD/least-squares constructions and feedback across
proposals; this is supervised authoring, not zero-shot invention.

| Frozen condition | Total /28 | Retention /12 | Word answers /8 | Situation answers /8 |
|---|---:|---:|---:|---:|
| Unchanged | 17 | 12 | 3 | 2 |
| Initial ASP | 15 | 9 | 4 | 2 |
| Final combined | 18 | 12 | 4 | 2 |
| Final WSP only | 20 | 12 | 6 | 2 |
| Final ASP only | 17 | 11 | 4 | 2 |

The combined artifact corrects one reserved conduction word-answer case and breaks
none. The same correction occurs with ASP alone. No reserved correction requires
both components to be present. The final weight edit mitigates the remaining
ASP retention regression, but ASP suppresses three improvements visible when the
weight edit runs alone: two convection word answers and one radiation word answer.
No condition improves situation-answer correctness. The positive combined result
therefore does not establish a joint advantage or general conceptual capability.

Final K is byte-identical to the original key. V is scaled to approximately 38%
of its original magnitude. W27 changes by 4.99% Frobenius norm, within the5% cap.
The author constructed a sum of repair/support outer products using acquisition
activation diagnostics and analytic linear algebra, then rescaled it. FP32 storage
rounding means this is not a claim of exactly rank-two stored delta.

Acquisition final:16/21 versus unchanged15/21; the original spoon target is fixed,
all12 acquisition retention answers preserved. Earlier initial ASP scored14/21
and lost two retention answers. Reserved gains are a tiny correlated designer
panel with reused ARC validation retention examples. The WSP-only ablation was
not independently optimized and is not a matched WSP-only experiment. Existing
final artifacts should be frozen before any broader confirmation.

Read-only audit verifies source identity, final artifact hash, tensor constraints,
unchanged key, value/weight magnitudes and recomputed saved summary counts. All
programs and feedback remain archived. Source shows analytic construction, not
iterative gradient optimization. Usage across six completed calls:276,713 input,
zero cached input,22,662 output including17,850 reasoning tokens. No dollar cost
inferred. Full visible transcripts were resent each response.

Artifacts: Skillforge-joint-attention-v1/results/joint_attention_v1.
Machine-readable summary: joint_attention_v1_results_summary.json.
No follow-on run launched.
