# Key-only selectivity: completed

Source receipt and run complete,228.75active seconds. Three Astra responses:
inspection, one accepted changed key, FINAL. Value,layer26/KVgroup0,base weights
and all non-key tensors frozen. Closed-form supervised calibration was explicitly
allowed; this is not zero-shot key authorship or evidence of learning without fitting.

Astra wrote ridge regression around the original key (regularization0.03), using
39query/head constraints from13disclosed questions. Desired virtual masses were
0.85 on conduction examples and0.02 elsewhere. It also computed analytic
leave-one-question-out diagnostics. No iterative optimizer or gradient updates.
An independent read-only reconstruction reproduces the final key bit-for-bit.

| New32-question panel | Base | Forced | Old key | New key | Zero key |
|---|---:|---:|---:|---:|---:|
| Conduction | 2/8 | 8/8 | 8/8 | 8/8 | 2/8 |
| Convection | 2/8 | 0/8 | 0/8 | 0/8 | 2/8 |
| Radiation | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| Evaporation | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| Total | 12/32 | 16/32 | 16/32 | 16/32 | 12/32 |

Routing changed substantially despite identical answer choices for forced/old/new.
Old key: conduction mean mass0.4876,negative0.6611,AUROC0.0625.
New key: conduction0.7694,negative0.2678,AUROC0.97917 (188/192 positive-negative
pairs correctly ordered). Negative means: convection0.3252,radiation0.2319,
evaporation0.2462. Only4/24negative cases fall at or below0.1mass;7/8positive
cases reach0.5. Training means were0.7257positive/0.03569negative,AUROC1.0.
There is therefore a material transfer gap in negative suppression.

Predeclared criteria: useful answer improvement YES; selectivity NO (negative
mean exceeds0.1); retention NO (8/10initially correct negatives retained).
The two lost answers are ironing and oven convection; they also fail with forced
and original-key interventions. Radiation has zero initially correct examples,
so no radiation retention claim is supported. Counts cover a small correlated,
designer-authored panel; no uncertainty/general superiority claim.

Interpretation: the supplied query space supports strong held-out ranking of
conduction applicability with one64-dimensional key, and Astra successfully wrote
an allowed supervised calibration that finds it. This improves on broad activation,
but is insufficiently selective in magnitude to preserve all correct answers.
No new scientific knowledge, general reasoning skill, WSPL advantage or complete
causal model is demonstrated. No follow-on run is approved or queued.

All programs, diagnostics, bank snapshots and evaluation records retained in
Skillforge-selective-key-v1/results/selective_key_v1. Summary JSON includes token
usage, per-setting and per-mechanism metrics. Audit verified pinned source, frozen
hashes/contract, independent ridge replay and all160stored argmax/correctness
labels plus independent routing AUROC. No new forwards or author calls in audit.
