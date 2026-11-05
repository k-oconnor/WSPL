# Fresh zero-game-feedback Go authoring screen

Preparation only. No model calls, authored-policy games or scientific launch.
Three Python sessions and three WSPL sessions, interleaved by draw with order
Python/WSPL, WSPL/Python, Python/WSPL. Each starts from the same all-zero
1,480,274-parameter architecture. No published controller code, final-answer
descriptions, replay outcomes or opponent tensors enter proposer context.

## Matched opportunity and scope

Eight response steps per session, maximum48 fresh Astra/xhigh calls. Each step
can submit one construction program or finish. Accepted weights persist; failure
keeps last accepted state and consumes a step. Both get construction diagnostics
and tensor statistics, never games/logits/activations. Python printed output is
also returned; WSPL includes plan resource/placement counts. No gameplay feedback,
training, fitting or search. Freeze latest state at explicit FINAL or step cap.
Zero/unchanged sessions remain in all denominators and are evaluated.

The unchanged one-shot collector receives a serialized full visible transcript
for each response. Each call has a fresh thread; hidden reasoning state is not
continued. Thus these are matched visible-transcript sessions, not a replication
of the original resumed Codex sessions. Original budget0 allowed up to120 Python
runs, more software and arbitrary persistent scratch; this screen allows8 steps,
NumPy-only Python and weight-only persistence. Do not claim a strict comparison
with the original authors' scores or resource efficiency.

Same CPU evaluator and opponents from the completed statistical replay. Freeze
all six candidates before **any** evaluation, then evaluate each and a uniform
zero-weight control against all three PPO seeds. Each matchup2,000games with
1,000 per color:42,000 total. Key = PRNGKey(11300000+100000*opponent+episode),
move key fold_in(...,tick), shared across candidates. Batch32/final16; FP32
JAX0.4.38 on CPU, pinned PGX2.6.0, same game rules and architecture.

Archive every response, prompt, usage record, accepted/rejected transaction,
construction diagnostics and source, accepted candidate files, final weights,
frozen identity table, every game's actions/reward/steps/final board and selected
trajectories. Per-opponent win rate and descriptive Wilson intervals; aggregate
mean over three opponents per author session. Session is the author replicate;
6,000 games from one policy are not6,000 independent authoring replicates.
No best-of selection, gameplay-based rewriting, or automatic follow-on run.

Primary comparison: six session scores, three paired WSPL-minus-Python mean-rate
differences (pairing by draw, not provider RNG). Also report finalization mode,
accepted/rejected step counts, source size, all reported token categories,
construction/authoring/evaluation time, nonzero coefficients and actual WSPL
features used. Training updates zero; hosted dollars, FLOPs and human time
unmeasured. Runtimes and resource accounting differ; no compute-matching claim.

Prospective descriptive triage: incomplete collection/evaluation is unresolved.
If fewer than two Python sessions exceed the zero-control mean rate, baseline
feasibility is unresolved. Otherwise at least two WSPL sessions exceeding that
control supports another screen. This is an exploratory threshold, not a
statistical advantage claim. A large interface gap requires inspection of
construction failures and actual strategy differences before scaling.

## Tool engineering and limits

WSPL basic tensor-slice reads/writes now avoid boxing/indexing the entire tensor;
dense graph metadata is cached during construction. Exact stored arithmetic,
overlap snapshots, negative/empty slices, signed-zero copies and immutable parent
reads remain tested. Charged work reflects selected size/rank; old published
operation counts remain associated with their original source pins. Historical
isolated archives are unchanged. New pin covers these implementation changes.

Python runs under bubblewrap with no network/GPU/host home and8GiB address space,
120CPU/150wall seconds. WSPL is a separate bounded interpreter subprocess with
160wall seconds. These differ; expose costs rather than claiming equal compute.
100KB reply limit;800KB visible prompt limit. A context or collection failure
stops the screen without automatic retry. Any evaluation infrastructure failure
or nonfinite policy execution stops the screen as incomplete, retaining artifacts.

Existing ChatGPT Codex usage, explicit gpt-6-astra/xhigh, no API-key fallback,
no enforceable dollar or output-token cap. At most48 calls, usually fewer if
authors finish early;20minutes per call,6hours overall. Expected1–3hours, mostly
authoring. No calls before exact source approval. Evaluation and construction
are CPU-only. Common task/interface packet: GO_AUTHORING_INTERFACE_V1.md.
