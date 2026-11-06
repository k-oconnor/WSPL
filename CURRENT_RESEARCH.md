# Current research handoff — November 6, 2026

Snapshot: **05:55 PST / 13:55 UTC**. This is a dated handoff, not live status.
**All five training/authoring arms are complete and frozen. Final held-out
evaluation is running. No final capability or WSPL advantage claim yet.**

## What we are testing

The research question has moved beyond whether an LLM can edit neural weights:

**What must an LLM engineer to add a reusable capability to another model, and
does WSPL improve that process?**

Small changes to a few correct predictions are not evidence of general capability.
The current assay asks for reliable one-cell movement of connected objects in6x6
grids, transfer to unseen shape families/larger objects, retained copying, and
two applications of the same learned transformation. It is a generated spatial
domain, **not ARC-AGI-1** and not a demonstration of a causal world model.

## Active experiment: capability6-v1

- Foundation: frozen copy-finetuned SmolLM2-135M-Instruct.
- Default adaptation:120zero-write rank16 branches on q/k/v/o attention
  projections across all30layers,1,891,200 trainable parameters including biases.
- Conditions: unchanged foundation; ordinary-gradient control; two native Astra
  authors; two native-plus-WSPL Astra authors.
- Both author interfaces share the same module expressivity and host training
  service. WSPL adds checked circuit/tensor editing and replay; it does not get
  extra model capacity. Author envelope:2.5M parameters and bounded static DAGs.
- Train16,384examples,development1,024. Each trained arm has at most8,192updates
  or one synchronized training hour. First512updates are copy-only. Four chunks
  or author responses, up to16Astra/xhigh calls overall. Numerical fitting and
  gradient training are allowed; this is not a gradient-free claim.
- Fixed instruction-only prompts, no worked demonstrations or CoT. Direct greedy
  unrestricted digit generation. No runtime object detector, movement solver,
  answer repair, candidate search or per-example weight editing.
- Final familiar/novel/larger/copy4/copy6 panels each512cases; composition256cases.
  Every trained candidate froze before any final prediction. Canonical shape
  families and complete inputs are separated. Composition feeds generated cells
  into the same model again without calculating a movement in the bridge.

See the [full protocol](docs/wspl/CAPABILITY6_V1.md),
[author interface](docs/wspl/CAPABILITY6_INTERFACE_V1.md), and
[Astra design review](docs/wspl/CAPABILITY_ENGINEERING_DESIGN_REVIEW_V1.md).
Current WSPL represents branch circuits, not the full transformer/attention runtime.

## Completed development scores — not final results

| Arm | Movement /512 | Copy4 /256 | Copy6 /256 |
|---|---:|---:|---:|
| Gradient control | 481 | 256 | 252 |
| Native Astra0 | 511 | 256 | 256 |
| Native+WSPL Astra0 | 510 | 256 | 256 |
| Native+WSPL Astra1 | 511 | 256 | 255 |
| Native Astra1 | 512 | 256 | 256 |

All reached8,192updates. The control's earlier development peak was499/512, but
the final checkpoint is retained as specified. Author feedback used development
scores, so these numbers cannot establish held-out transfer or author superiority.
No clear WSPL advantage appears in this development table. Audit actual WSPL use
before attributing any outcome to the language.

[Machine-readable dated snapshot](docs/wspl/capability6_handoff_snapshot.json).

## Independent capability criteria

Require familiar487/512,unseen-family461/512,larger-object461/512,copying507/512
on both sizes with at mostfive paired regressions each, and composition231/256.
Also require at least103additional successes over the same-prompt unchanged
foundation in EACH movement panel. These are descriptive engineering thresholds,
not power-derived statistical tests. A pass shows a bounded neural skill; two
author pairs cannot establish general WSPL superiority.

## Why this setup

The earlier late-layer37k-parameter ReLU adapter learned only limited movement;
Astra repair added two net successes across192fresh movement cases. We explicitly
rejected calling that general capability. This led to an Astra design review and
the broader attention adaptation tested here.

- [Copy foundation](docs/wspl/GRID_COPY_V1_RESULTS.md): perfect held-out copying.
- [Late adapter](docs/wspl/GRID_MOVEMENT_V1_RESULTS.md): failed transfer criterion.
- [Astra repair](docs/wspl/GRID_REPAIR_V1_RESULTS.md): small gains, failed criterion.

## Local operational handoff

Approved source SHA-256:
`39074af2e7bf08aab2f7691654c9e9b3771c09b53bb152e84aae9337cdfa6a02`.
Registry ID`capability6-v1`; manifest`capability6_v2.manifest.json`.
User explicitly approved this identity and launch. Started November5 at19:34PST.

Isolated checkout:
`C:\Users\kevin\OneDrive\Desktop\Skillforge-capability6-v2`

Output relative to that checkout:`results/capability6_v1/`.
Read`status.json`,then`source_receipt.json`and`results.json`when present.
Per-arm`session.json`records usage/budgets;`final.json`contains the frozen deployed
modules. Each turn/chunk has`development.json`and`feedback.json`. Author turns
retain prompts,replies,programs and construction records. Final files are
`<arm>_<split>.json`;`frozen.json`binds final artifact hashes.

Windows launcher PID2816; actual Python worker observed as54772. Verify process
identity before acting; do not assume PIDs remain valid. Launch logs are adjacent
to the output directory as`capability6_v1.launch.*`.

**Logging limitation:** status is not refreshed per grid during development or
final evaluation. A panel can appear static for many minutes. Check process CPU
advancement,errors and newly completed panel files before declaring a stall.
Training steps reset per arm, so repeated step counts can also mislead.

Do not edit the isolated source or bound resources while active. No automatic
follow-on. The18hour guard is an incomplete-run guard, not evidence of completion.
Main checkout development is separate from the active pinned checkout.

## Next audit when complete

1. Check BOTH run results and source receipt; independently recompute exact scores
   and paired retention from saved outputs. Distinguish complete from guard stop.
2. Verify frozen hashes,source identity,base immutability evidence and split rules.
3. Report all six conditions on all final panels and fixed criteria, including
   failures. Do not select the best development checkpoint post hoc.
4. Inspect author programs/transactions: placements,ranks,training recipes,actual
   WSPL operations,native fallback,invalid proposals,tokens and resource budgets.
5. Separate installed capability,authoring benefit over ordinary training,and
   WSPL benefit over native tooling. Save results before proposing more runs.

This push preserves research context and supporting documents. It is not a
portable runnable release of the active local experiment: model/data/runtime
resources and the full pending implementation are not uploaded by this handoff.
The isolated run remains the authoritative execution record.
