# Read-only investigation of the strongest Go policy

This inspection reads saved programs and tensors only. It runs no new model
calls, game episodes, fitting, policy forward probes or ablation evaluations.

## Actual edits

| Accepted program | Coefficients changed from previous state | Final nonzero count | Intended change from source inspection |
| --- | ---: | ---: | --- |
| 0 | 14,551 (from zero) | 14,551 | Spatial and tactical policy skeleton |
| 1 | 8,608 | 17,925 | Extra liberty propagation, capture/rescue saturation, connection handling |
| 2 | 7,554 | 22,999 | Complementary liberty estimates; move positional computation |
| 3 | 8,793 | 23,980 | Free channels for friendly three-liberty features; capture exception |
| 4 | 5,671 | 20,512 | Replace empty-diagonal correction with contact-redundancy correction |
| 5 | 981 | 20,674 | Combine liberty evidence and rescale downstream thresholds |
| 6 | 256 | 20,674 | Strengthen diagonal opponent contributions to false-eye handling |

Counts compare stored FP32 values, not source statements or charged editor
operations. Table labels describe authored intent; they do not establish
semantic correctness or a performance effect. Tensor counts are saved in
`go_winner_edit_inventory.json`.

The last edit changes only W5: 196 weights from -0.5 to -1 and 60 from -0.5
to -2. Every changed edge connects an opponent-occupancy channel to a diagonally
adjacent point's eye feature. No other coefficient changes. The source intends
different treatment of interior versus edge/corner eyes. This is a particularly
clean before/after contrast: no added parameters or unrelated channel changes.

The preceding edit changes exactly 162 W3 weights from 0 to 0.25, 162 b3 values
from -1 to -1.25, 288 W5 weights from -8 to -16 and 369 from -2 to -4. It combines
two sources of liberty evidence and adjusts downstream sensitivity. These are
coupled changes and should first be evaluated together, rather than assuming
the new 0.25 connections alone explain any gain.

## Comparison with other candidates

| Policy | Nonzero coefficients | Mean win rate |
| --- | ---: | ---: |
| Python 0 | 38,235 | 42.52% |
| Python 1 | 36,459 | 32.67% |
| Python 2 | 44,619 | 21.32% |
| WSPL 0 | 28,171 | 38.28% |
| WSPL 1 | 28,147 | 30.92% |
| WSPL 2 | 20,674 | 52.15% |

The strongest policy is also the sparsest of these six: 1.397% of 1,480,274
coefficient slots are nonzero. The evaluator still executes a dense network;
this does not establish an inference-speed advantage or that sparsity causes
better play. The best Python policy also implements explicit liberty reasoning,
so the difference is not presence versus absence of an algorithmic controller.

## Next test, not launched

Evaluate all seven stored WSPL-2 checkpoints with identical paired environment
seeds and opponents. Fix the episode budget and contrasts before launch, report
every checkpoint, and do not retune the author programs from the outcomes.
First compare checkpoints 5 versus 6 (eye patch), 4 versus 5 (liberty patch),
and the complete sequence descriptively. This localizes improvement or regression
without assuming the seven edits improved monotonically.

Then select an explicitly labeled exploratory mechanism test: board fixtures
with independently counted group liberties, plus targeted removal of the
identified feature. A score change measures the whole edit's effect; matching
its claimed semantics requires these separate checks. More independent author
draws would still be needed to test an interface-level advantage.

These new benchmark evaluations require their own source pin and approval.
