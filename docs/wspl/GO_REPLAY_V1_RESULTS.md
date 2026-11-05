# Go released-policy replay: completed

All 12,000 games completed in 125.53 active seconds (2.09 minutes), with one
constructor call and a complete source receipt. Finished November 4, 2026 at
22:05:18 UTC. No fresh model calls, training or GPU use.

| Authored policy | PPO opponent | Local win rate | Released win rate | Difference |
|---|---|---:|---:|---:|
| Zero feedback | seed 0 | 40.85% | 42.55% | -1.70 pp |
| Zero feedback | seed 1 | 27.90% | 26.30% | +1.60 pp |
| Zero feedback | seed 2 | 36.05% | 36.25% | -0.20 pp |
| 50 feedback passes | seed 0 | 75.00% | 74.15% | +0.85 pp |
| 50 feedback passes | seed 1 | 69.25% | 69.95% | -0.70 pp |
| 50 feedback passes | seed 2 | 82.15% | 78.55% | +3.60 pp |

Each row comprises 2,000 games, evenly split between Black and White. Exact
released figures come from the retained `h2h_lr_scan.json`; its README rounded
some of them. Mean win rate across the three opponents is 34.93% for budget0
and 75.47% for budget50. These are two released policies, not independent
replicated training/authoring experiments or a causal estimate of feedback value.

The broad ranking reproduces: budget0 loses to each same-size PPO opponent;
budget50 beats each. Five absolute discrepancies are <=1.7 percentage points.
The budget50/seed2 gap is 3.6 points, with local 95% Wilson interval 80.41?83.77%,
which excludes the released point estimate 78.55%. Do not describe every rate as
statistically matching or attribute that discrepancy solely to chance. The
original evaluator/version/RNG are unavailable; ours uses an explicitly pinned
PGX implementation and JAX sampling procedure. No tuning or retries against the
published results were performed.

## Audit

Verified 121 source files and 3,469 bound resources, all five archived actors
bit-for-bit against released tensors, all 378 NPZ batches and 12,000 JSONL rows,
unique episode identities, seeds, balanced colors, action counts, terminal flags,
win counts, confidence intervals and termination reasons. Independently computed
area scores from final boards using a separate flood-fill implementation; checked
repetition-loss outcomes against mover color. All outcomes agree with the ledger.
Checked occupied-point exclusion for 77,747 recorded moves in the selected
trajectories. This is an archive/final-score audit, not a full re-execution of
all trajectory transitions, legality checks or policy logits. No new games run.

Evidence: [audit](go_replay_v1_audit.json); auditor:
`tools/audit_go_replay_v1.py`. Full local archive remains in sibling
`Skillforge-go-replay-v1/results/go_replay_v1`.

## Implication

We now have a tested, fast local Go evaluator and can prepare a fresh Python
versus WSPL authoring comparison. This replay is not a WSPL result. Preserve
competent executable Python construction and equal feedback budgets in that
comparison; original zero-feedback still allowed construction-code execution
and continuation. Fresh authoring requires a separate protocol and approval.

Approved manifest: `a0044b77433bd6262a77e93454be06d0721e32c6eecaa136307471d1d44d600c`.
