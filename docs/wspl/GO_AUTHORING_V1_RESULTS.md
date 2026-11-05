# Go authoring: completed exploratory comparison

Completed November 4, 2026 at 21:39:55 PST (November 5, 05:39:55 UTC).
Original source pin: dcf87c7a310ea245af2411152065b5c6e9593d1cbc576862562c522cdfe27f1d.
Recovery pin: cf96be2cc6651cd382e27f77f7cfdc347b98a5a124e9e924c587e569fe3eb108.

Both recovery result and source receipt report complete. The original run remains
incomplete with `call_wall_guard`, preserving its failed request. Four candidates
were imported byte-for-byte; the recovery instantiated the two remaining author
models and the zero control. No gameplay feedback or gradient training occurred.

## Scores

Each policy played 2,000 games against each of three PPO opponents, balanced
by color. All 42,000 games are retained, including the control.

| Policy | PPO seed 0 | PPO seed 1 | PPO seed 2 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Python 0 | 47.40% | 34.45% | 45.70% | 42.52% |
| WSPL 0 | 41.50% | 31.95% | 41.40% | 38.28% |
| Python 1 | 37.95% | 25.15% | 34.90% | 32.67% |
| WSPL 1 | 35.75% | 23.50% | 33.50% | 30.92% |
| Python 2 | 25.60% | 15.90% | 22.45% | 21.32% |
| WSPL 2 | 57.75% | 41.70% | 57.00% | 52.15% |
| Zero | 2.05% | 1.40% | 1.40% | 1.62% |

Python mean 32.1667%; WSPL mean 40.45%. Paired differences are -4.2333,
-1.75, +30.8333 percentage points. The best WSPL policy exceeds the best Python
policy by 9.6333 points. The third WSPL policy leads all Python policies against
each of the three opponents. These policies did not play each other.

All six exceed the zero control, meeting the prospective descriptive feasibility
conditions. The mean advantage depends on one authoring pair; the authoring
sample size is three per interface, not 18,000 independent author replicates.
Do not infer a general interface advantage or assign the gain to specific edits.

## Construction and usage

| Arm | Completed responses | Accepted programs | Reported input tokens | Reported output tokens | Of which reasoning |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python | 22 | 19 | 232,218 | 261,509 | 234,482 |
| WSPL | 23 | 20 | 307,130 | 288,427 | 251,047 |

Each session ended with FINAL; no returned construction was rejected. One
additional Python call timed out after 20 minutes with no answer and unknown
usage. Totals above exclude that unreported usage. Input totals include reported
cached tokens (29,824 Python; 2,176 WSPL). No dollar or FLOP accounting exists.
There is no observed WSPL token-efficiency advantage in the completed calls.

Original active wall time 16,330.953 seconds; recovery 6,714.532 seconds, about
6.40 hours combined, excluding the intervening pause. The recovery had a fresh
four-hour guard. Actual completed responses: 45, plus one interrupted attempt.

## Source inspection, not causal attribution

The strongest WSPL session builds spatial channels, propagates liberty evidence,
adds complementary overlap corrections, relocates positional features to free
neuron channels, and adjusts self-atari/capture/eye handling. Its seven accepted
programs use fixed-shape tensor edits and parent references; no graph growth or
high-level circuit placement. "Certificate" is the author's description of
intended feature semantics, not an independently verified theorem.

The strongest Python session also builds explicit two-/three-liberty features
and tactical penalties. Whether any particular refactoring caused the WSPL
gain requires intermediate-checkpoint evaluation and targeted ablations under
a separately approved scientific pin. No such evaluations are included here.

## Verification and archive

Read-only checks reproduced all final weight hashes, complete per-opponent
episode-ID coverage, color assignment, valid/terminal flags and win totals from
stored batch files. This is accounting validation, not independent game replay.

Original archive: `Skillforge-go-authoring-v1/results/go_authoring_v1`.
Recovery archive: `Skillforge-go-authoring-resume-v1/results/go_authoring_resume_v1`.
Both are local siblings of this repository. Full weights and game records remain
excluded from Git. Protocols, source pins, implementation and this report are
tracked. Original evaluator differences and smaller construction allowance
preclude a strict replication claim against published CSAIL scores.
