# Astra-supervised Qwen adaptation: completed pilot

The independently run, post-submission evaluation scored the frozen submission
32/32 missions, versus 8/32 for the saved ordinary-practice executive. Each arm
used identical initial states. Astra was absent during deployment.

| Condition | Baseline | Submission |
|---|---:|---:|
| Ordinary | 8/8 | 8/8 |
| Changed calibration | 0/8 | 8/8 |
| Mid-mission switch | 0/8 | 8/8 |
| Return to original calibration | 0/8 | 8/8 |

There were 24 paired corrections, zero regressions, and no ordinary-condition
probes. The predeclared exploratory system-performance gate passed.

## What Astra built

Astra retained the common motor library, redesigned the public observation/action
interface, generated training examples, trained Qwen's LoRA and selected a saved
checkpoint using development evidence. The deployed Qwen emits one unconstrained
full-vocabulary argmax token per decision. A/B select a documented motor, digits
select waypoints, P invokes a fixed bounded probe, and S stops. The codec aliases
IDs and formats public measurements; its inspected code does not diagnose the
regime, choose a controller or override model actions. Probe parameters are
engineered, not learned. No attention/KV intervention is deployed.

Selected weights descend from 600 v1 updates (batch four, LR 1e-4), then the
300-update checkpoint of a 600-update raw-observation continuation (AdamW,
cosine LR 4e-5 to 5e-6). A matched arithmetic-feature continuation was also tested.
Training uses full-vocabulary next-action-token cross entropy. This is
Astra-directed gradient adaptation with interface engineering, not direct
hand-authored weight-space programming or evidence of WSPL superiority.

Archived author-run development ablations show 384/384 diagnostic decisions for
the selected adapter versus 128/384 for parent weights with the same codec.
Removing feedback or documentation removes corresponding paired sensitivity.
These are development diagnostics, not independent final-set ablations. They
support a learned contribution; the primary gain bundles training, data and
representation changes. No independent inference rerun was performed in this audit.

## Scope and evidence

This supports persistent adaptation of a small Qwen executive within two known
motor regimes and a bounded reaching task. It does not establish arbitrary robot
adaptation, physical deployment or general language capability retention.
The author also records failures for unsupported faults and very low gains.

Read-only audit verified all 326 pinned sources, the frozen bundle and adapter
hash, 64 result rows, all 32 paired start hashes, and one observed Qwen forward
per submitted-policy decision. Forward counts alone are not causal attribution.
Source receipt and run status are complete. See
`supervisor_recovery_result_audit.json` for audit details.

Recovery wall time was 4,884.8 seconds (~81.4 minutes), including verification
and final evaluation. Recorded development GPU-job time was 2,341.9 seconds
(~39.0 minutes), not total GPU inference time. Across original v3 plus recovery:
28 completed Astra calls, one capacity-interrupted call with unknown usage;
known input tokens 1,112,864, output 40,118 including reasoning 15,907, zero cache.
These counts exclude the earlier invalid v2 study.

Frozen artifact:
`results/astra_supervisor_resume_v1/supervisor/submissions/a5b1d71c2321cc10cc97819c8a522027ac9b04bace62ee49aa48579bcb240b83`.
Adapter SHA-256: `d3a67a7d84c53d0041d8f6886709acac59723ca31b4c7bd784bd88962bc6c50c`.
No further experiment launched.
