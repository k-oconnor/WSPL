# Frozen ARC attention transfer: completed

Run and source receipt complete; 563.25 seconds (9.39 minutes). All 5,214 row-condition evaluations saved. Source identity and saved row counts/accuracy totals checked read-only. No author calls or training.

| Condition | Challenge /299 | Easy /570 | Combined /869 | Corrected | Broken |
|---|---:|---:|---:|---:|---:|
| untouched | 97 | 285 | 382 | 0 | 0 |
| forced | 91 | 268 | 359 | 32 | 55 |
| original_key | 93 | 276 | 369 | 29 | 42 |
| ridge_key | 93 | 282 | 375 | 27 | 34 |
| astra_key | 95 | 286 | 381 | 20 | 21 |
| grid_key | 93 | 288 | 381 | 7 | 8 |

The original disclosed spoon question (Mercury_SC_408700) changes from evaporation to conduction with forced/original/ridge/Astra keys, but not the grid key. This is a known-target replay, not a new correction-generalization result.

The Challenge lexical heat slice has ten questions. Its sole correction is the original spoon target. The Astra key changes no answers in the thirty-question Easy heat slice. The lexical slice includes unrelated questions through distractor words (e.g. radiation), so it is not a semantic applicability label.

Astra fixes 20 and breaks 21 answers overall; grid fixes 7 and breaks 8. No condition beats the untouched model on combined normalized accuracy. Raw likelihood results are also saved: baseline398/869, Astra383/869, grid397/869. Small Easy-only gains do not establish a transferable capability.

The frozen intervention can change full-answer decisions, but this test does not demonstrate reliable conduction concept transfer or a general ARC gain. These validation sets have previously been used; no sealed-test claim. No follow-on run launched.

Artifacts: Skillforge-arc-attention-transfer-v2/results/arc_attention_transfer_v1.
