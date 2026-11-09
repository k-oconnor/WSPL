# Current research handoff - November 9, 2026

All SmolLM capability, ARC retention, prose degradation and integration-repair
runs are complete. No follow-on experiment is running or queued by this work.

The user decision is to preserve these as useful experiments, with nothing from
this branch of work promoted into the supported WSPL design yet. Simpler-network
controller construction and repair remain a distinct, more promising evidence
base. Transformer capability integration remains an open research problem.

## What the completed experiments establish

- Gradient-trained attention adapters learned the bounded 6x6 movement task and
  transferred to unseen shape families. This is not direct authoring of the skill
  and is not ARC-AGI.
- Every active movement adapter reduced ARC science accuracy. WSPL0 also raised
  prose perplexity from18.06 to112.55; all64sample documents worsened.
- Astra repaired part of the interference by scaling existing adapter outputs:
  Q/K1,V.5,O.625. Every trained parameter remained unchanged. No WSPL syntax was
  used, and uniform attenuation explains most of the recovery.
- Passing the repair tolerance is not a decisive advantage over simple tuning,
  evidence of algorithm reconstruction, or evidence of a WSPL benefit.

See [completed results](docs/wspl/SMOLLM_EXPERIMENTS_2026-11-09.md) and its linked
protocols and machine-readable summaries. The previous November6 handoff and
its development-only snapshot remain in Git history.

## Repository and execution status

The repository includes the experimental modules, runners, preparation tools,
unit checks and documentation. These are archived research code, not a portable
one-command release or newly adopted product features. They require local model,
data and runtime resources; preparation tools document the original environment.

Historical manifest digests and source inventories are in
[the source index](experiments/archive/research_source_identities_2026-11-09.json).
Machine-specific full manifests/approval registries, model checkpoints, datasets
and raw execution directories remain local. Historical approvals must not be
transferred to the current source. Any future scientific run needs a new verified
source pin and explicit approval under AGENTS.md.

GitHub's already-merged PRs11/12/13 bring artifact storage/caching, vectorized
writes and optional plan_and_apply into the development checkout. These compiler
performance changes are separate from the SmolLM experiments. Existing study
callers still use their original plan/apply path; isolated past runs are untouched.

## Local evidence locations

Under the sibling checkout directories:

- Skillforge-capability6-v2/results/capability6_v1
- Skillforge-capability-arc-v1/results/capability_arc_v1
- Skillforge-prose-degradation-v1/results/prose_degradation_v1
- Skillforge-integration-repair-v1/results/integration_repair_v1

Each retains source_receipt.json, approved_manifest.json and results.json.
The copy-trained full foundation remains in
Skillforge-grid-copy-v1/results/grid_copy_v1/last.pt; subsequent deployments use
compact adapters rather than duplicated whole-model checkpoints.
