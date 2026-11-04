# Diffusion replay v1: completed

The approved CPU replay completed all 3,072 images (1,024 per model) in 10.344
active seconds. The source receipt completed with one constructor call and no
stderr errors. No author calls, training or GPU use.

The first 64 unfiltered samples from each network visibly show filled triangles,
squares and circles, respectively, with variation in size/position and occasional
noisy or malformed samples. This is visual inspection, not classifier accuracy.

| Artifact | Released class | Mean pixel |
|---|---|---:|
| 246a5b58 | triangle | -0.662724 |
| a6a9dfb7 | square | -0.507447 |
| d88aba69 | circle | -0.513785 |

Read-only audit checked all 48 batch hashes, 3,072 FP32 finite images in [-1,1],
exact archived weights, seeds, paired initial noise, final trace equality and
independently recomputed means. Evidence: `diffusion_replay_v1_audit.json`.
Full local archive: sibling `Skillforge-diffusion-replay-v1/results/diffusion_replay_v1`.
All generated images and selected intermediate states remain in NPZ batches;
`contact_sheets.png` combines the three unfiltered first-64 previews.

This establishes a usable reconstructed sampling/evaluation pipeline. The original
sampler/RNG/classifier was not released, so it does not replicate their exact
shape accuracy, demonstrate new LLM authorship, or establish WSPL editing value.
Future no-forward-pass interpretation packets must exclude these samples and keys.

Approved manifest: `6c2420f54ce0d8bdff4640eb50271b8c3b7f646c23bdbc1fdb57ef9fd972b401`.
