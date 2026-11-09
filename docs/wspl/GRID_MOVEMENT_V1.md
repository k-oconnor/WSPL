# Movement adapter reachability

Freeze the completed copy-trained SmolLM2-135M foundation. Train only one
37,472-parameter ReLU residual adapter at layer27 output, at every token:
`h' = h + B ReLU(Ah+a)+b`. Native hidden states are the only inputs; no external
object detector, movement solver, task router or per-example deployed edit.
The supplied movement instruction and eight demonstrations remain in the prompt.

Training pool:168 single-cell/bar inputs remaining after excluding all original
demonstrations, copy cases and movement evaluation cases. No L shapes in training
or demonstrations. Each direction/color combination has21 cases. Train1,024
updates, batch4: three movement cases and one random copy case sampled with
replacement. Copy rehearsal uses the previously prepared4,096-grid copy training
pool, not copy holdouts. Total65,536 answer targets:49,152 movement,16,384 copying.
Repeated training cases are intentional; this is a small feasibility domain.

FP32, AdamW LR.001, betas .9/.999, epsilon1e-8, zero decay,32-update linear warmup,
constant rate afterward, clip norm1. Adapter seed173, sampling seed347. Zero write
matrix/bias recovers the foundation exactly; read matrix starts random. Base
parameters remain frozen/eval, gradients and tensor hashes checked. Loss is
full-vocabulary answer-only cross entropy over16 digits, excluding prompt/padding.

After the fixed budget, freeze the final adapter. Compare unchanged foundation
and final adapter on32 training diagnostic cases,64 familiar movement cases,
64 novel-L movement cases,64 historical domain copy cases and64 random copy
holdouts. Both palette-only and strict unrestricted16-token decoding:1,152 grids.
The movement panels were evaluated previously on the original base, so these
are training-disjoint feasibility panels, not untouched confirmatory tests.
The random copy panel was used in the prior copy gate. No checkpoint/threshold
selection on evaluation results; no intermediate evaluation feedback to training.

Predeclared descriptive reachability gate: at least48/64 exact palette movement
grids on EACH familiar/novel panel, at least16 additional successes over unchanged
foundation on EACH, and copying at least61/64 on BOTH copy panels with at most
three regressions in aggregate count per panel relative to foundation. Report
all raw scores and unrestricted generation regardless. Training performance
distinguishes failure to fit from failure to transfer. This is one seed, not a
claim about general ARC-AGI or the best achievable adapter architecture.

One compact rolling adapter/optimizer/RNG checkpoint every30minutes plus final
or clean-stop; reference the parent checkpoint without copying it. Source pin
stored in checkpoint; parent identity bound in manifest. No automatic resume.
Two-hour wall guard; expected10–30minutes. GPU fitting/inference, no Astra calls
or downloads. No automatic authoring, tuning or follow-on run. A passing result
supports preparing a matched Astra authoring test, not claiming Astra can do it.
