# Released diffusion artifact replay v1

Preparation only. No sampled images, author calls, training or GPU work has run.
The exact candidate source digest requires user approval before launch.

## Question and scope

Can the three released 23,824-parameter ReLU denoisers produce inspectable images
under a fully specified reconstruction of the published DDPM description?
This is infrastructure for Experiment 2, not a WSPL editing comparison or a
fresh replication of Astra's weight-interpretation task.

Reference: RikiyaT/lm_understand-logs commit
`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`, `exp2_diffusion`.
NPZ models `246a5b58`, `a6a9dfb7`, `d88aba69`, exact released FP32 weights.
Network 160→64 ReLU→64 ReLU→144 linear, input-row/output-column matrices.
No weights are modified. The sampler predicts x0, not epsilon.

## Fixed reconstruction

1,024 images per model, 16 batches of 64. Paired PCG64 seeds 9,200,000 through
9,200,015 across models. Each batch starts with standard Gaussian noise; draws
are NumPy float64 normals converted to FP32. Every image is retained.

1,000 reverse steps, t=999 down to 0. Beta is linearly spaced from .0001 to .02.
Let alpha=1-beta, A=cumprod(alpha), and A_previous=1 at t=0. Clip the model's
clean prediction to [-1,1]. Transition mean coefficients are
`beta*sqrt(A_previous)/(1-A)` on clean prediction and
`sqrt(alpha)*(1-A_previous)/(1-A)` on noisy pixels. Posterior variance is
`beta*(1-A_previous)/(1-A)`. At t=0 return clipped clean prediction exactly,
without drawing noise. Coefficients/time embedding are computed in float64,
cast to FP32; model inference and transitions are NumPy FP32. Eight sine then
eight cosine channels use frequencies `10000**(-k/8)` and integer t in radians.
CPU only, one BLAS thread, pinned NumPy 1.26.4 and runtime file hashes.

The release does not supply the original sampler implementation, RNG or shape
classifier. This is a declared reconstruction, not bitwise reproduction. Report
sample count, mean pixels, range and unfiltered images; do not substitute visual
inspection for their classifier accuracy or publish a replicated accuracy number.

## Artifacts and stopping

Archive exact weights, all 3,072 generated images in FP32 NPZ batches, the first
image's intermediate states at t=999/749/499/249/0 and final output for every
batch, three lossless PGM contact sheets (first 64 samples), per-batch seeds and
hashes, summary, status and source receipt. Previews are only a display transform.
No checkpoint copies or gradient state. Source identity in the result/receipt.

Status updates every 30 seconds. Wall guard 30 minutes, checked each reverse
step; `control.json` with `{"stop":true}` requests a clean stop. A partial batch
is discarded and explicitly excluded from completed counts; completed batches
remain archived. No auto-resume/retry. Expected several minutes, estimate not a
measured released-model sampling benchmark.

Constructor: `skillforge.diffusion_replay.DiffusionReplayStudy`.
Runner: `experiments/run_diffusion_replay.py::run`.
Candidate model ID: `diffusion-replay-v1`; proposed output
`results/diffusion_replay_v1` in isolated `Skillforge-diffusion-replay-v1`.
No proposer context contains these outputs or evaluator keys in a subsequent
no-forward-pass interpretation experiment. Editing objectives require a separate
protocol and source approval.
