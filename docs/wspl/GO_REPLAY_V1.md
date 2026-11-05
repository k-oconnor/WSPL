# Go released-policy replay v1

Prepared, not approved or launched. No released-policy games or authoring calls
have been run. Unit tests use rule fixtures and synthetic network weights only.

## Question

Reconstruct the released Experiment 6 same-size-PPO comparison before fresh WSPL
authoring. Two authored policies (`budget_0`, `budget_50`) versus three released
PPO seed checkpoints (`ppo_same_size_lr3e-5_s0/s1/s2`), 2,000 games per matchup,
12,000 total. The authored policy is Black for even episode IDs, White for odd
IDs: exactly 1,000 of each per matchup. Same 1,480,274-parameter fixed architecture,
243 inputs, six 512-wide ReLU layers and 82 logits. All five actors use released
FP32 tensors with no modification, fitting, search or authoring.

Published same-size opponent win rates are 42.6/26.3/36.3% for budget0 and
74.2/70.0/78.5% for budget50. Those are comparison anchors, not success gates.
Do not compare with the hidden-small-PPO headline numbers: those actors are absent.
Original `ppo_ref.h2h`/`play.py` wrappers and exact runtime are unavailable.
This is a statistical replay, not bitwise reproduction. Large disagreement prompts
an audit of the evaluator and actors, not tuning against the published scores.

## Rules, inference and randomization

Pinned PGX commit `87278d2d6e677fd87248c457207b59cfa42e578d`, package 2.6.0.
Use its Go Game implementation with size9, komi7.5, history_length1 and 162-move
limit. Preserve PGX rules unchanged: captures, suicide/simple-ko masking,
repeated-position loss for the mover, two-pass and move-cap termination,
area scoring. PGX uses a fixed two-word Zobrist board hash for repetition;
this replay inherits that implementation, not a collision-free board-history proof.
Passes are exempt from repetition loss. Finished states freeze for later ticks.

Input (r*9+c)*3+[0,1,2] is own stone/opponent stone/White-to-move. The first
two planes derive from the current board and match PGX's latest observation;
there are no additional history planes. No random player-order wrapper: actor
color is explicit and terminal rewards are indexed by that color.

JAX0.4.38/jaxlib0.4.38, NumPy1.26.4, CPU FP32, highest matmul precision,
x64 disabled. Both actors evaluated on every active batch position; choose the
current actor's logits, mask illegal actions to negative infinity, then JAX
categorical sampling at temperature1. Nonfinite logits or illegal selected
actions invalidate the batch and stop the study. No reruns or substitutions.

Episode key = JAX PRNGKey(10300000 + opponent_seed*100000 + episode_id).
Move key = fold_in(episode_key, tick), tick0..161. Same episode keys/colors for
both authored policies against each opponent; trajectories can diverge.
Batch32, final batch16. RNG procedure differs from the missing original helper.

## Outputs and guards

Save exact five actors, all 12,000 episode outcomes, colors, seeds, move counts,
termination reasons, complete action sequences and final boards. Save before-move
boards for the first two games of each batch (one game of each color), permitting
trajectory inspection without new game execution. NPZ batches and JSONL ledger,
per-match overall/color win rates and descriptive 95% Wilson intervals. No
aggregate superiority claim or selection of a best opponent/seed.

CPU only, no GPU/API/model calls. Separate `/home/kevin/venvs/skillforge-go` runtime;
completed scientific runtimes are unchanged. Pin installed dependencies, interpreter,
core shared libraries, source and input files. Expected tens of minutes, estimate
not a measured scored-game benchmark. Two-hour wall guard checked between batches;
initial compilation/batches can delay the check. Clean stop via control.json
`{"stop":true}` preserves completed batches and marks the run paused. No resume
or automatic next experiment. Status updated after every batch.

Constructor `skillforge.go_replay.GoReplayStudy`; runner
`experiments/run_go_replay.py::run`. Planned output `results/go_replay_v1` in
isolated `Skillforge-go-replay-v1`. Launch only through approved source gate.

## Provenance

Actors and released evaluation scripts: RikiyaT/lm_understand-logs commit
`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`; acquisition receipt
`provenance/csail_relu_reference_v1.json`. PGX source:
https://github.com/sotetsuk/pgx/tree/87278d2d6e677fd87248c457207b59cfa42e578d
(Apache 2.0). Rules and encoding tested before launch. Learned actors are only
imported/validated during preparation; full games require exact digest approval.
