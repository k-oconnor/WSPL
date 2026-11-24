# Matched robot authoring: interpretation of the completed screen

November 24, 2026. Source pin `6640c6968490e742abbc619f3619207d9047d1171cd0f3328207e93fa68354bb`.
The continuation completed in 9,839.529 seconds with 93 author responses and 512 final missions. The original failed attempt and its unknown author usage remain separate. No follow-on was launched.

## Finding

Astra produced persistent, useful but incomplete adapter repairs in this engineered robot domain. One Python-plus-WSPL author preserved all 16 ordinary missions while adding 11 successes on 48 adaptation missions. This meets the predeclared exploratory criterion. None of the three native authors met that combined criterion.

The result does not establish that WSPL improved construction. All three final WSPL bodies assign entire matrices fitted in Python. Their deployed tensors exactly match the supplied bindings. The successful numerical repair was therefore constructed outside WSPL; this run demonstrates WSPL reconstruction/deployment of those matrices, not WSPL-driven synthesis.

All arms already use the same supplied, analytically compiled WSPL motor blocks. “Native” denotes the authoring interface, not a robot system without WSPL. These comparisons cannot measure the value of the shared motors.

## Outcomes and the preservation tradeoff

Each tuple below is ordinary / changed-from-start / switch-after-first-waypoint / return-to-nominal-after-first-waypoint, with 16 missions per condition:

- Parent: 16 / 0 / 0 / 0.
- Gradient control: 16 / 9 / 16 / 8; combined criterion passes.
- Native 0: 8 / 7 / 6 / 3; fails preservation.
- Native 1: 11 / 7 / 1 / 1; fails preservation.
- Native 2: 9 / 10 / 10 / 8; fails preservation.
- WSPL 0: 12 / 6 / 0 / 0; fails preservation and adaptation threshold.
- WSPL 1: 16 / 2 / 8 / 1; combined criterion passes.
- WSPL 2: 9 / 8 / 12 / 5; fails preservation.

Native authors average 17.67/48 adaptation successes and 9.33/16 ordinary successes. WSPL authors average 14/48 and 12.33/16. Mean total success is 27/64 native versus 26.33/64 WSPL. The single passing WSPL session reflects a useful preservation/adaptation balance; it is not an overall accuracy advantage. Three sessions per interface are the author-level sample size. Repeated missions and shared panels do not create additional independent author draws.

Because the parent gets all ordinary missions right, the predeclared at-most-one-regression requirement effectively requires at least 15/16 ordinary successes, despite the separate >=14/16 clause. With zero baseline adaptation successes, the no-condition-net-loss requirement places no further restriction on these particular adaptation results. Passing requires eight adaptation gains in aggregate; it does not require strong performance in each adaptation condition. WSPL 1 still fails 37/48 adaptation missions and is especially weak at starting under changed calibration and returning to nominal.

## What the authors built

Exact frozen-array comparisons show all six authors changed only `model.layers.27.mlp.down_proj` A/B: 167,936 coefficients. The other 195 LoRA pairs remain bit-identical. The gradient control changed all 196 pairs, totaling 18,464,768 coefficients. All full-vocabulary final runs produced zero invalid tokens, so these outcome differences are not formatting failures.

The supplied diagnostics expose final-block inputs, residuals, action logits, action-head rows and normalization scales. These make a final-block supervised correction especially accessible; the common choice of location is not independent evidence that authors discovered the best site. Reviewed construction code fits teacher-derived corrections with NumPy ridge regression, maps them through the frozen output-head/normalization rows, and uses matrix factorizations to fit the rank constraint. Solving a supervised regression without gradient descent is still learning from labels.

The passing WSPL 1 construction retains seven singular components of the original pair and adds a nine-component correction. After development feedback it scales the general correction by 0.8 and the motor contrast by 0.65. Its final WSPL program is `A[:,:]=RepairA; B[:,:]=RepairB;`. The binding file contains the complete replacement arrays. This points to conservative fitting and preservation as plausible reasons for its result, but there is no ablation establishing which construction choice caused the gain.

The final native/WSPL numerical deployments use the same model runtime. The WSPL text is not an additional solver running during robot inference. Exact binding equality supplies no mechanism for different inference behavior from loading the same arrays natively.

## Behavioral evidence and limits

The parent never probes on the final panel. WSPL 1 does probe, and saved successful switch trajectories show a motor change followed by completion. For example, switch mission 32 has action sequence `PA0001PB111112222S`: probe, select nominal motor A, reach the first waypoint, encounter the changed calibration while trying the second, probe/select B, finish remaining waypoints and stop. Another successful trajectory oscillates between motors before finishing. These are useful closed-loop behaviors, not uniformly reliable execution.

The motor controllers themselves were supplied. The added behavior concerns their selection and use from feedback. New mission seeds demonstrate within-domain transfer beyond acquisition and development; they do not establish a general calibration algorithm, new motor synthesis, real-world transfer, or language retention. A feedback ablation would be required to isolate causal dependence on the relevant feedback.

The gradient control preserves all ordinary successes and adds 33/48 adaptation successes. Every WSPL 1 success is also a gradient success. However, other authors collectively solve seven adaptation missions the gradient control misses. This suggests different failure patterns; no ensemble, selector or oracle combination was deployed or evaluated. The control has different optimization access and edits far more pairs, so it establishes attainability under this interface, not a matched efficiency comparison or an inherent limit on authoring.

## Implications for the release and next question

Supported wording: “Astra constructed persistent LoRA repairs for a Qwen robot executive using supervised numerical methods. One WSPL-packaged repair preserved ordinary behavior while improving adaptation on fresh simulator missions. WSPL reconstructed the authored matrices exactly.”

Unsupported from this screen: WSPL outperforms native authoring, generates these repairs more efficiently, installs a general adaptation capability, or guarantees behavioral preservation. The present evidence can fill the experiment section as a bounded mixed result, not support the proposed efficiency headline.

Before another scientific run, the most informative design decision is which WSPL contribution to test: synthesis under shared numerical primitives, verification/reconstruction reliability, or author productivity. A comparison where both interfaces perform all fitting in Python and WSPL only copies arrays is weakly targeted at synthesis. A common final-pair gradient/ridge baseline would separately help distinguish author strategy from the current all-pair control's broader optimization access. These are proposed questions only; no follow-on is prepared or authorized here.

## Audit scope

`tools/analyze_robot_authoring_v2.py` rechecks all 377 inventoried source files; all eight frozen file hashes and adapter identities; 512 paired episode records, per-mission records, decision counts and invalid-token counts; final comparisons; exact changed coefficients; author usage; and final WSPL binding equality. Output: `robot_authoring_v2_analysis.json`. No model forwards, fitting, author requests or simulator executions occur in this analysis.

Targeted inspection reviewed the final submission programs and the passing author's construction path. An AST call screen is included as navigation evidence, not a proof of semantic compliance. It records a saved malformed native request rather than silently excluding it. Full semantic review of every executed program, including dynamically executed source, remains outstanding. The original study's `pending_source_audit` classification has not been overwritten.
