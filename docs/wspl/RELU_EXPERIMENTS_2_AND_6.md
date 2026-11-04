# Next ReLU studies: released Experiments 2 and 6

Kevin selected both on November 4, 2026, after the completed Kinetix repair
comparison. This authorizes preparation; no new scientific run or model budget
has an approved source pin. Keep the completed Kinetix archives unchanged.

## Reference acquisition

Pinned upstream commit: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f` in
`RikiyaT/lm_understand-logs`. Acquired 115 files / 31,144,905 bytes, including
three diffusion networks and descriptions, Go budget-0/budget-50 actors, three
same-size PPO opponents, protocols and evaluation records. All sizes, Git blob
SHA-1 and SHA-256 identities independently verified. Provenance:
`provenance/csail_relu_reference_v1.json`; ignored local bundle:
`data/csail_relu_reference_v1`. Upstream LICENSE retained. Downloaded author code
and evaluator answer keys are evidence only, not proposer input or executable
preparation scripts.

The acquisition finished all downloads and wrote its receipt before a final
PowerShell summary-print error. The original preparation script is preserved in
the bundle with its recorded hash; independent verification confirmed every file.
The future-facing downloader fixes only that print operation. No download was
silently retried or overwritten.

## Experiment 2: diffusion interpretation, then a distinct editing extension

The [original protocol](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp2_diffusion/README.md)
asks Astra to identify what a model generates from rounded weights alone, without
tools or forward passes. It does not ask for a repair. Three 23,824-parameter
160→64→64→144 ReLU denoisers generate filled circles, squares or triangles.
The network predicts clean images; a DDPM sampler wraps it. Step conditioning
uses eight sine and eight cosine channels.

1. Import all three exact FP32 artifacts and separately preserve the rounded
   text presented to the original author. Verify layer orientation and activation
   semantics before introducing any WSPL transformation.
2. Reconstruct and pin the sampler from the supplied schedule/description. Its
   implementation, RNG and original shape-classifier code are not supplied in this
   selected release. Do not call locally chosen sampling/classification procedures
   exact replicas. Unit-check the time embedding, clipping, schedule and final-step
   noise convention before a scored replay.
3. A faithful interpretation replication must retain the no-forward-pass rule;
   generated samples, semantic class labels and evaluator keys stay out of proposer
   packets. The original reports correct shape classes but incorrect filled/outline
   descriptions, so class accuracy alone would miss an important error.
4. A WSPL editing test is a separate extension, with matched raw graph access,
   clear output-change objectives and preservation checks. A known input/output
   coordinate transformation can validate circuit placement and composition as an
   engineering fixture; it would not demonstrate learned-weight understanding.
   Freeze a nontrivial editing objective before spending authoring calls.

The ReLU denoiser is eligible for supported real-graph checks. Its stochastic
sampling loop and distributional image quality are separate empirical claims;
ReLU activations alone do not certify the whole generator.

## Experiment 6: Go network authoring with a competent Python comparator

The [original protocol](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp6_go_from_scratch/README.md)
uses a fixed 243→six512-ReLU-layers→82 network, 1,480,274 parameters, initialized
to zero. Python construction is allowed; gradient/data fitting and search are not.
The released budgets are 0, 10, 25 and 50 evaluation-feedback passes. Zero passes
does not mean one response: its construction harness still allows code execution
and continuation. Do not replace that comparator with literal-only assignments.

The prompt identifies public [PGX](https://github.com/sotetsuk/pgx) `go_9x9`:
162-move cap, two-pass termination, area scoring, komi 7.5, suicide/simple-ko legal
masking and loss on repeated whole-board position. Encode three interleaved planes:
current-player stones, opposing stones, White-to-move. Sample legal logits at
temperature one; no search. These details require fixtures, not assumptions based
on generic Go rules.

The release's evaluation scripts import missing `scripts/gohand/play.py` helpers.
The exact original PGX version and two small hidden PPO opponents are not present
in the acquired tree. Three same-size PPO opponents are released. Therefore:

1. Resolve and pin a PGX implementation and reproduce the prompt's rules, input
   encoding, color balance and action sampling. Declare any differences and test
   captures, suicide, ko, repeated-position loss, passing, horizon and scoring.
2. Prepare a released-artifact replay using budget-0/budget-50 policies against
   the available same-size PPO opponents. Compare with the corresponding published
   head-to-head results, not the unavailable small-opponent headline scores.
3. Prepare fresh zero-feedback Python versus WSPL authoring, allowing the richer
   WSPL circuit/placement interface within the same fixed architecture. Give both
   arms equal construction execution/feedback budgets and a real isolated sandbox;
   ordinary Python subprocess execution is not a sandbox. No hidden opponent or
   published construction code enters authoring context.
4. Introduce matched evaluation-feedback budgets only after the zero-feedback
   harness is validated. Preserve fresh author replicates, token accounting and
   final evaluation separation. General topology-changing surgery is an explicitly
   separate extension with equal raw-graph access and capacity in the comparator.

WSPL defaults to one million artifact elements; Go requires explicit opt-in to
1,480,274. Do not silently raise global defaults. Check actual full-size import,
plan/apply, immutable preservation and bounded construction memory before launch.
ReLU compatibility does not imply that full-network SMT equivalence is tractable.

## Completed input compatibility checks

`tools/audit_csail_relu_inputs.py` independently rechecked every downloaded file,
then imported all three FP32 diffusion networks into WSPL. It imported the actual
released budget-0 Go actor with an explicit 1,480,274-element limit and replayed
a no-op exact-to-storage plan/apply transaction, preserving the full artifact
identity bit-for-bit. This took about 9.84 seconds for Go import and transaction
on the preparation host; it is not a comparative benchmark. Report:
`docs/wspl/csail_relu_input_audit.json`. No global library limits were changed.

## Launch boundary

Acquisition and input compatibility checks are complete, with zero model calls, diffusion
samples, Go games or training. Source approval comes after each concrete runner,
protocol, runtime, data isolation and bounded test suite are ready. No source pin
or registry entry exists for either new study. Authoring and editing extensions
will be reported separately from released-artifact replay and interpretation.
