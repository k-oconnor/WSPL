# Seven-checkpoint Go edit evaluation

Post-hoc analysis of the strongest of six previously scored author sessions,
WSPL draw 2. Evaluate all seven accepted stored checkpoints without modification,
2,000 games against each of three original PPO opponents, 1,000 games per color:
42,000 new CPU games. Same fixed FP32 architecture, PGX runtime, legal-action
mask, sampling, seed base 11300000, per-opponent offset 100000, 162-step cap,
and batch32/final16 as the completed authoring comparison. No author calls,
training, GPU use, policy tuning or automatic follow-on run.

Report every checkpoint, all six adjacent differences, per-opponent rates and
paired win/loss switches. Primary descriptive contrasts: edit6 minus edit5 (256
eye-handling weights), and edit5 minus edit4 (981 liberty-related coefficients).
Intervals are normal paired-difference intervals stratified over the three fixed
opponents, not inference over author replicates. No multiplicity-adjusted or
confirmatory claim. Winner and seed set were selected/seen previously; this is
mechanism localization on the original evaluation set, not held-out validation.

Archive every episode's actions, outcome, final board and selected trajectories.
Require edit6 replay arrays to match all 189 historical final-policy batches;
any mismatch fails the run as incomplete. Bind all seven candidate files,
historical batches, opponent weights, runtime and protocol hashes. Preserve all
prior archives. One supplied constructor loads all frozen policies. Live status
updates per batch; a two-hour guard and stop control interrupt without automatic
resume. Expected roughly 5–15 minutes; compilation/runtime may vary.

The sequence can identify a helpful/harmful whole edit. It does not independently
prove the claimed liberty semantics or isolate interacting subcomponents within
an edit. No expectation of monotonic improvement is assumed.

Scientific launch requires exact source-pin approval under AGENTS.md.
