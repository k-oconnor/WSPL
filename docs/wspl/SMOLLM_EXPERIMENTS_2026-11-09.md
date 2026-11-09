# Completed SmolLM capability and retention experiments

November 9, 2026. Archived research, not an adopted WSPL integration mechanism.
All four runs below have complete source receipts and results. Aggregate evidence
and original result-file/source hashes are in
[the machine-readable summary](smollm_experiments_2026-11-09.json).

## 6x6 spatial learning

Frozen copy-trained SmolLM2-135M; five fresh attention-adapter arms trained for
8,192 updates each. Authors choose bounded architectures and training recipes.
The spatial skill is learned through gradient training, not authored directly
as a complete weight-space program. This synthetic task is not ARC-AGI.

| Arm | Familiar /512 | Novel /512 | Larger /512 | Copy4 /512 | Copy6 /512 | Composition /256 |
|---|---:|---:|---:|---:|---:|---:|
| Foundation | 0 | 0 | 0 | 246 | 389 | 0 |
| Gradient | 490 | 418 | 353 | 512 | 495 | 199 |
| Native 0 | 511 | 498 | 486 | 512 | 509 | 233 |
| Native 1 | 512 | 498 | 484 | 512 | 512 | 243 |
| Native + WSPL 0 | 512 | 502 | 491 | 511 | 512 | 236 |
| Native + WSPL 1 | 511 | 506 | 504 | 512 | 512 | 245 |

All four author arms passed the descriptive grid criteria. Two authors per
interface, different final architectures/recipes and parameter counts, and host
training do not establish a general WSPL advantage. Runtime15.35hours, including
about2.07hours of training; generation-based evaluation dominated the remainder.
Full [protocol](CAPABILITY6_V1.md).

## ARC science retention

Unchanged answer-likelihood scoring on previously used ARC test sets. All24,836
row-condition scores and paired corrections/regressions were checked against the
saved summaries. This is descriptive retention, not fresh benchmark confirmation.

| Condition | Challenge /1172 | Easy /2376 |
|---|---:|---:|
| Original HF model | 336 (28.67%) | 1207 (50.80%) |
| Copy foundation | 343 (29.27%) | 1221 (51.39%) |
| Gradient adapter | 258 (22.01%) | 687 (28.91%) |
| Native 0 | 276 (23.55%) | 970 (40.82%) |
| Native 1 | 273 (23.29%) | 849 (35.73%) |
| WSPL 0 | 281 (23.98%) | 985 (41.46%) |
| WSPL 1 | 274 (23.38%) | 798 (33.59%) |

Every active movement adapter substantially degraded ARC performance. The raw
answer-likelihood metric also declined. Runtime17.80minutes.
[Protocol](CAPABILITY_ARC_V1.md).

## Simple prose degradation

64 FineWeb-Edu validation documents,32,768 prediction targets per condition.
Copy foundation NLL2.89384/PPL18.0625; WSPL0 NLL4.72341/PPL112.5517.
Loss worsened on all64documents; perplexity increased6.23x. All three fixed greedy
adapter completions became repetitive. Runtime39.359seconds.
[Protocol](PROSE_DEGRADATION_V1.md).

## Astra integration repair

Four responses: three accepted edits, then FINAL. No host training. Native Python
appended one scalar output multiplier per branch. All2,490,840trained parameters
remained bit-identical. Final gains across all30layers:Q=K=1,V=.5,O=.625.
No WSPL syntax was used. The sequence tested V/O=.5/.5,then.8/.5,then.5/.625;
Q/K stayed1. The same adapter remained active on grids and prose.

| Condition | Reserved prose NLL | Novel /64 | Larger /64 | Copy /32 |
|---|---:|---:|---:|---:|
| Foundation | 2.85156 | 0 | 0 | 24 |
| Original adapter | 4.83048 | 62 | 64 | 32 |
| Selected uniform strength .75 | 3.61888 | 60 | 60 | 32 |
| Astra repair | 3.52224 | 59 | 59 | 32 |

The repair recovered66.1%of the prose NLL gap and passed the predeclared integration
tolerance. Its slight prose advantage over uniform attenuation costs one extra
failure in each movement panel. This is not Pareto dominance or decisive evidence
of intelligent circuit repair; ordinary attenuation explains most recovery.
Prose final documents were separate; grid panels had been evaluated previously.
Runtime32.02minutes. [Protocol](INTEGRATION_REPAIR_V1.md).

## Decision

Retain the code, programs, protocols and findings as research evidence. Do not
promote these adapters, attention interventions or scaling repairs into the
supported WSPL design. Evidence for constructing/repairing simpler neural
controllers remains separate from the substantially harder problem of adding
reusable capability inside a pretrained language model.

Raw datasets, large weights, local runtime bindings and full execution archives
remain local. Repository source is current development code, not a promise of
byte-identical historical reruns. Historical manifest digests/source inventories
are recorded in [the source index](../../experiments/archive/research_source_identities_2026-11-09.json);
new execution requires resource preparation and a newly approved source identity.
