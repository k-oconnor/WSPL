# Pseudo-causal logging for opaque neural networks

Implemented November 3, 2026 as **empirical computational intervention evidence**
in `skillforge.interventions`. This complements formal checks for networks that
are outside the current verifier's operator/size limits. It is not a fallback
proof system and does not establish world causation or feature meanings.

A small [known-answer log](../../examples/wspl/intervention_fixture.jsonl) and
[audit summary](intervention_fixture_audit.json) are tracked for inspection. The
hand-constructed model is identity Linear -> Tanh -> readout [1,0], input [1,2].
Zeroing feature 0 changes its scalar output by 0.761594; zeroing feature 1 has no
measured effect. Controls, chain verification and an exact replay pass. This is
12 engineering fixture forwards, no learning, dataset sweep or causal science claim.

## What a record establishes

For one pinned runtime/model/input context, replace a selected activation while
holding registered model state, input and controlled RNG streams fixed. Compare
the output against matched baseline/no-op/repeat runs. An observed difference is
evidence of that intervention's effect in that computation on that input.

The claim is not that the neuron means a concept, that the intervention is
on-distribution, that the effect generalizes, or that changing it causes a robot
outcome. Those require separate grounding, distribution and physical experiments.
An effect under zero ablation may be an off-manifold response. No measured effect
does not prove irrelevance: redundancy, other inputs, interactions and measurement
tolerances remain possible. A declared hypothesis is logged as unverified text.

## Current adapter scope

The first adapter accepts a PyTorch nn.Module, one dense tensor input, a finite
floating tensor output, and tensor-valued named module outputs as intervention
sites. Arbitrary internal operators are allowed: the regression fixture uses
Tanh, which is outside the current ReLU verifier. Tensor input can hold token IDs
for a caller-supplied logits wrapper, but full generation/decoding is not logged.

Every audit declares a stateless eval-mode runtime. Stateful/recurrent/KV-cache
models need an explicit reset adapter, which is not implemented. Tuple/dict-valued
sites and non-PyTorch frameworks need adapters. This is not support for every
neural framework or arbitrary hidden state. Caller-owned exclusive inference is
required; concurrent use of the same model/hooks is not supported.

The logger temporarily sets eval mode, then restores individual module modes,
temporary hooks and Python/NumPy/Torch RNG states. It clones the input for every
forward and uses inference mode. It does not train or patch model parameters.
Registered parameter/buffer hashes, including nonpersistent buffers, are checked
before and after. A mutation invalidates evidence but is **not rolled back**.
Use a suitable owned/resettable model; do not treat this as a transaction restoring
unknown custom state. Unregistered Python state and external code cannot be fully
fingerprinted, and their coverage limitation is recorded.

## Four matched runs

1. Baseline forward.
2. No-op hook through the same clone/replace instrumentation path.
3. Selected intervention.
4. Repeated baseline.

All runs use fresh copies of the same input and replayed Python, NumPy and Torch
seeds, including already-used model/input CUDA devices where applicable. The
logger does not silently enable deterministic algorithms or claim all kernels
are reproducible. Baseline repeat and no-op must agree under the predeclared
tolerances (default exact); otherwise status is INVALID_CONTROL. A hook that
changes aliasing behavior can invalidate its own comparison, and has a regression.

The target is module path plus invocation index, allowing shared/recurrent-depth
blocks to be distinguished within one stateless forward. Select the full tensor
or explicit last-axis feature indices. Supported edits: zero, scalar scale,
scalar replacement or exact-shape tensor replacement. Tensor replacement can be
supplied from an external donor capture, but donor acquisition/provenance and
distribution validity are not automatically established by this API.

## Logged evidence and replay

Each record includes:

- Runtime/adapter ID, registered-state digest, module classes/configuration and
  available class-source hashes, with explicit incomplete-identity coverage.
- Input hash/schema, operator/site/invocation/features, scalar value or tensor
  payload hash, RNG settings, framework/Python versions and instrumentation hash.
- Per-run output hashes/summaries and selected original/installed activation
  hashes/summaries; four forward calls and total audit wall time.
- No-op/repeat controls, state integrity and max/mean absolute output difference.
- Optional logit argmax-change fraction and baseline top-two margin, only when the
  caller declares logit output semantics. These are descriptive, not guarantees.
- Evidence tier and explicit not-established flags for universality, semantic
  grounding and physical causation; no PROVED status is accepted in these logs.

Default logs contain hashes and compact numeric summaries, not complete weights,
inputs or activations. Replay is not self-contained: the caller supplies the
matching model and input, and any tensor replacement payload. `replay_intervention`
checks model/input/operator/runtime/instrumentation identity and exact output
digests, returning individual mismatch checks. It makes four additional forwards.
A seed alone is not a reproducibility certificate.

`append_evidence` writes a single-writer JSONL hash chain; `verify_log` checks
indices, predecessor/content hashes and evidence tier. A separately retained
expected head detects rewriting/truncation relative to that anchor. Without a
trusted anchor, someone can rewrite and rehash the entire chain. The log is not
signed, authenticated, a locked immutable filesystem or safe for concurrent
writers. No hidden tensor payloads or evaluator truth are reconstructed from a hash.

## Example

Requires optional PyTorch and NumPy; tested in the existing CPU research
environment. The following uses a caller-provided stateless model and input:

```python
from skillforge.interventions import (
    Intervention, audit_intervention, append_evidence, replay_intervention,
)

record = audit_intervention(
    model, input_tensor,
    Intervention(module="encoder.block", call_index=0, kind="zero", features=(3,)),
    runtime_id="my-pinned-model-and-adapter-v1",
    declared_stateless=True, seed=7,
    hypothesis="Feature 3 contributes to this output on this input",
)
head = append_evidence("work/interventions.jsonl", record)
replayed = replay_intervention(model, input_tensor, record)
```

Create the parent directory first. Status is OBSERVED_EFFECT, NO_OBSERVED_EFFECT,
INVALID_CONTROL or INVALID. It is never a universal or real-world causal proof.
The logger operates on supplied models; it has no model loader, API client,
scientific runner or automatic sweep over a held-out dataset.

## Research integration

Keep three evidence classes distinct: observational summaries, controlled
computational interventions and scoped formal contract results. Activation
correlation alone cannot be promoted to intervention evidence. Successful
intervention controls cannot be promoted to a universal contract.

For opaque models, the logs can support hypotheses and repair diagnosis. A
scientific population-level claim needs a preregistered input distribution,
prespecified sites/comparisons, matched sham controls, effect estimates and
uncertainty, multiplicity/selection accounting, and a separate held-out panel.
Never cherry-pick an impressive ablation and call it a discovered semantic rule.
Weight-edit lineage, stateful adapters, donor provenance, multi-site interaction
studies and model-independent runtime adapters are further work.

The cross-domain plan records this empirical extension separately; its outcomes
do not enter universal proof-coverage counts. New scientific audits/benchmarks,
large-model probes, robot evaluations and API use still need their concrete
configuration and source approval. Tests here use known-answer synthetic models,
including no-effect, changed input, stochastic RNG, invalid shared state, alias
sensitivity, reused calls, Tensor patch replay and corrupted/false-proof logs.

The implementation uses PyTorch's
[module forward-hook interface](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/module.py)
and records limitations consistent with its
[reproducibility guidance](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/randomness.rst).
