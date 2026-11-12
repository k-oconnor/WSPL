# Stage-one authoring panel: completed results

The recovery completed on November 11, 2026 at 16:16 PST. All 96 final candidates are valid across 24 problems: 14 forecast series, eight regressions and two causal instances. There were 144 completed Astra calls (19 preserved and 125 recovery calls). The interrupted invocation remains archived with unknown usage. Recovery active time was 26,399 seconds.

## Factual prediction

Counts below compare test RMSE separately within each problem; raw errors are not pooled across datasets.

| Direct author | Lower RMSE than fixed AdamW | Lower RMSE than Astra-guided AdamW | Lowest RMSE among four methods |
|---|---:|---:|---:|
| Native WSP | 22/24 | 20/24 | 14/24 |
| WSPL | 24/24 | 23/24 | 9/24 |

Astra-guided AdamW won the remaining problem. Native beat WSPL on 14 problems; WSPL beat native on ten. Forecast winners were WSPL seven, native six and guided one; regression winners were native six and WSPL two. Native won factual prediction on both causal instances.

Against forecast persistence, WSPL improved on 12/14 series and essentially tied one (exchange rate); native improved on 11/14. Both direct methods beat the training-mean predictor on 21/24 problems. All four neural methods lost to the training mean on Dominick, SGEMM and yprop. The neural-method win counts therefore do not establish universally useful predictions.

## Causal estimates

| Instance / individual-effect RMSE | Fixed AdamW | Guided AdamW | Native WSP | WSPL |
|---|---:|---:|---:|---:|
| IHDP replication 948 | 5.71082 | 5.30951 | 1.65404 | 2.94464 |
| ACIC 2018 selected instance | 0.107512 | 0.094308 | 0.013677 | 0.122187 |

The IHDP reference is mu1-minus-mu0; ACIC uses the released y1-minus-y0 contrast, not an assumed noise-free CATE. Fixed AdamW had the best IHDP average-treatment-effect error (0.02977), despite its worse individual-effect error. For ACIC, native estimated ATE 0.07616 against truth 0.07914; WSPL estimated -0.04231. ACIC zero-effect RMSE was 0.08026, better than fixed, guided and WSPL. Factual gains cannot be substituted for treatment-effect evidence. Authors received factual outcomes and feedback only.

## Cost and construction

| Author interface | Completed calls | Input tokens | Output tokens, including reasoning | Author time |
|---|---:|---:|---:|---:|
| Native | 48 | 1,396,928 | 622,124 | 18,056 s |
| WSPL | 48 | 1,438,774 | 342,036 | 10,277 s |
| Guided AdamW | 48 | 1,369,794 | 27,695 | 1,287 s |

WSPL used approximately 45% fewer output tokens and 43% less author time than native in this panel. These are descriptive totals, including reasoning, with different cache use; they are not controlled compiler-speed measurements. Fixed and guided fitting each totaled approximately 85 CPU seconds, excluding authoring and other overhead. Four native response attempts failed the literal-dtype check; every final candidate nevertheless remained valid under the last-accepted-artifact rule.

## Interpretation and audit

This supports direct weight construction on these small-data problems. It does not establish WSPL accuracy superiority: native won more head-to-head problems. There is one draw per method/problem, at most 128 shared acquisition rows, fixed small ReLU architectures and no compute matching. Gradient controls are bounded AdamW MLP fits, not tuned ridge, boosted-tree, forecasting or causal-estimation competitors. Public datasets may have appeared in author pretraining. No automatic stage-two expansion is authorized.

Read-only audit verified all 299 pinned source files, hashes of all 96 frozen final artifacts, 144 distinct completed threads without tool calls, and all 13 preserved parent candidates. See `PANEL_STAGE1_RESULTS_AUDIT.json`; full evidence is in `../../results/panel_stage1_resume_v1`. This audit did not rerun model predictions. Previously disclosed missing upstream provenance files remain unavailable; prepared packets were preserved and verified during recovery.
