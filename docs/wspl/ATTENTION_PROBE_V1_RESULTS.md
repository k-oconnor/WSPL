# Native attention pilot: completed

Original run stopped on a status-file sharing error; recovery completed in
485.703 seconds with source receipt complete. Two previous calls retained;
nine completed Astra responses total, plus one interrupted call with unknown
usage. Seven accepted programs (including inspections) and two FINAL responses.
No new inference or author calls in this read-only completion audit.

| Original questions only | Untouched | Forced value | Authored key | Zero key |
|---|---:|---:|---:|---:|
| Disclosed target | 0/1 | 1/1 | 1/1 | 0/1 |
| Conduction paraphrases | 1/4 | 4/4 | 4/4 | 1/4 |
| Other heat mechanisms | 2/6 | 2/6 | 2/6 | 2/6 |
| Lexical controls | 1/2 | 1/2 | 1/2 | 1/2 |

All reversed-choice checks reproduce scores exactly; these are implementation
checks, not additional independent examples. Four original wrong answers became
correct, no previously correct answers became wrong. However, the controls were
already weak: the base failed every radiation/convection case and one lexical
case. Preservation of their correctness counts is therefore limited evidence.

Astra chose layer26/KV group0 (query heads0..2), replaying four times an existing
value vector from target-prefix position10, with forced mixing0.75. It then
constructed a 64-dimensional key by SVD/least-squares calibration of the target's
three query logits to attain the same0.75 mass. A heuristic projection attempted
to suppress shared framing directions. These linear-algebra operations were
explicitly allowed; this is target-calibrated construction, not zero-shot or
independent evidence of semantic discovery. Base weights stayed frozen.

The key reproduces target mass0.7500001 and correct-answer margin+1.21081,
versus base-2.48350. Zeroing the key removes the correction. Nevertheless,
retrieval is broad: it attracts substantial mass on negative controls too.
Conduction log probability increases on ALL13 original questions, by2.52 to5.02
nats. Two already-wrong controls switch their wrong answer to conduction.
Forced and key-based interventions yield identical answer choices on this panel.

Conclusion: successful native attention delivery and a functioning authored key;
no demonstrated semantic selectivity or newly acquired scientific reasoning.
The result is consistent with a broadly applied answer-direction bias that helps
this selected question family. A follow-on study should use negative controls
that the base actually answers correctly and measure conditional routing directly.
No follow-on run is approved or launched.

Audit checked pinned source identity, final artifact hashes, intervention bounds,
frozen value/alpha across stages, and all stored argmax/correctness labels.
Artifacts remain in Skillforge-attention-probe-resume-v1/results/attention_probe_resume_v1;
original failure archive preserved. See attention_probe_v1_results_summary.json
for usage, norms, attention masses, and counts. No general ARC/WSPL advantage.
