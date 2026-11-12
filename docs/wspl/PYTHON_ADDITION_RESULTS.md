# Example-driven Python versus neural predictors

Completed November 12, 2026. All six ordinary Python predictors were valid.
Python had lower factual prediction RMSE than both native WSP and WSPL on five
of six tasks, and lower RMSE than both saved AdamW baselines on all six.
ERCOT favored the neural predictors. This is evidence against assuming that
neural deployment is the best default for standalone prediction; it does not
establish that neural programs lack integration or verification benefits.

## Design

Six tasks were selected alphabetically within regression, forecasting and causal
categories (two each). Each Python session received the same 128 acquisition
rows, preprocessing and aggregate validation feedback as the earlier panel,
with two responses and the last valid program retained. No target rule was
supplied. Executable fitting, optimization and external tools were prohibited;
formulas, interpolation and example-based prediction were allowed.

The six Python artifacts froze before test scoring. Native WSP, WSPL, fixed AdamW
and Astra-guided AdamW results are reused from the completed panel; none were
rerun. Both neural authoring interfaces used the same fixed small ReLU networks.
Python had greater algorithmic expressivity and could embed acquisition examples.
This compares deployment representations, not matched capacity or compute.

## Prediction RMSE

Lower is better; scales differ between tasks. Do not average these raw errors.

| Task | Python | Native WSP | WSPL | Fixed AdamW | Guided AdamW |
| --- | ---: | ---: | ---: | ---: | ---: |
| Abalone | 2.232717 | 2.288313 | 2.291920 | 3.051302 | 2.315694 |
| Diamonds | 0.252243 | 0.256530 | 0.271490 | 0.465238 | 0.278226 |
| Dominick | 33.727241 | 34.803870 | 34.117950 | 63.308213 | 45.573525 |
| Ercot | 35.872566 | 32.317073 | 32.613291 | 104.567337 | 86.312137 |
| ACIC 2018 | 0.394266 | 0.400813 | 0.404188 | 0.474010 | 0.486097 |
| IHDP 948 | 1.071599 | 1.986541 | 2.751281 | 4.091069 | 3.962843 |

Dominick's training-mean RMSE was 32.663734, better than every authored or
AdamW predictor in this comparison, including Python (33.727241).

## Causal effects are a separate outcome

| Task | Python effect RMSE | Native WSP | WSPL |
| --- | ---: | ---: | ---: |
| IHDP 948 | 0.374502 | 1.654040 | 2.944635 |
| ACIC 2018 | 0.082599 | 0.013677 | 0.122187 |

IHDP uses the supplied conditional mean contrast mu1-mu0. ACIC uses released
potential outcomes y1-y0, which are not assumed to be noise-free CATE.
Python's ACIC effect error was slightly worse than the zero-effect reference
(0.080261), despite having the best factual prediction RMSE.
Python/native/WSPL ATE absolute errors were 0.309791/0.247631/0.115502 on IHDP,
and 0.011352/0.002980/0.121455 on ACIC. Better individual effect RMSE therefore
does not imply better ATE estimation either.

## What the programs do

- Abalone: nonlinear physical formulas plus neighbor-based residual correction.
- Diamonds: logarithmic size/price formulas plus kernel-weighted residuals.
- Dominick: availability/level heuristics and matching observed history suffixes.
- ERCOT: local trend extrapolation plus nearest-neighbor forecast corrections.
- ACIC: explicit feature-conditioned baseline and heterogeneous treatment effect.
- IHDP: an exponential control-outcome formula and linear treated-outcome formula.

These mechanisms illustrate a representational advantage of ordinary code;
for example, a small ReLU network must approximate the exponential used by IHDP.
The available evidence does not isolate this as the cause of each score difference.
Final source inspection found formulas and example-based prediction rather than
an executable optimizer. This is not a formal semantic compliance certificate.
Familiar public datasets and recognizable semi-synthetic structure make prior
knowledge a plausible contributor, especially for IHDP; no pretraining exclusion
or novel causal discovery is claimed.

## Execution and audit

Twelve completed responses across six sessions, including one preserved Abalone
response from the deliberately stopped larger run. Only eleven new calls were
needed; no neural reruns occurred. Total output usage was 205,849 tokens including
reasoning across all twelve completed responses. The resumed run reported
6,732.890 active seconds (about 112 minutes), excluding the earlier execution of
the preserved first response. These totals do not establish a token-efficiency
advantage over neural authoring.

Read-only audit verified the approved source manifest, completed source receipt,
six final program hashes, twelve distinct completed threads, and exact metric
recomputation from saved predictions. All recorded reversed-batch output
differences were zero. No new model calls, fitting or inference were performed
for this report. Historical neural inference timings are not a matched timing
comparison with this addition.

## Limits and research implication

One author draw per task, six selected tasks, previously evaluated partitions,
small acquisition samples, limited gradient baselines and unequal expressivity
make this exploratory. The result supports ordinary-code prediction as a required
baseline. Claims for WSPL should separately demonstrate benefits from neural
integration, composition, verification or deployment constraints, rather than
infer them from beating a small gradient-trained network.

The earlier supplied-rule comparison and this example-driven addition address
different settings; their evidence should not be pooled as repeated trials.
Prepared packets were preserved and verified; 48 unavailable upstream provenance
paths from the panel folder move were not freshly revalidated.

## Artifacts

- [Protocol](PYTHON_ADDITION_V1.md)
- [Machine-readable results and audit](PYTHON_ADDITION_RESULTS_AUDIT.json)
- [Six final Python programs](../../artifacts/python_addition_v1/)
- [Historical panel results](PANEL_STAGE1_RESULTS.md)

Full prompts, replies, collection receipts and per-row predictions remain in the
local immutable run `Skillforge-python-addition-v1/results/python_addition_v1`.
The public archive contains compact programs and metrics, not the full runtime
or dataset bundle. Approved manifest SHA-256:
`f39e7ef741785f2691e7d1414156c980d7156b281109f68645a5de85aad22f9c`.
