# Frozen-tensor import and relational contracts (G3a)

Added November 3, 2026 (issue #6). This is the first slice of G3 and of the S2
prerequisites in FORMAL_EXPERIMENTS_V1.md: **import** of frozen MLP tensors and
**relational queries** between a parent and an edited network. Output-patch
composition, pre/postprocessing composition and transaction manifests/rollback are
G3b. No network was trained; parent training requires its own approved source identity.

## Current engineering results

| Fixture | Result |
| --- | --- |
| BrakePatch compiled weights re-expressed as a float32 import bundle | Stored layers identical to the committed artifact; relational preservation PROVED; the three existing BrakePatch contracts PROVED on the imported artifact |
| ApproachP, same | Same, with the two ApproachP contracts |
| Deterministic pseudo-random float32 3-6-6-2 MLP (synthetic, untrained) | A hidden-bias edit is REFUTED globally and PROVED preserved on the region where that unit is inactive; a 2^-10 output-bias edit is REFUTED at tolerance 0 and 2^-11 and PROVED at 2^-10 |

See the [fixture results](import_fixture_results.json). The two
`*.compiled_as_import.json` bundles are **import-fidelity fixtures**: WSPL-compiled
weights, not learned networks, and never S2 baselines. Their `provenance.note` says so.

## Import bundle (`wspl.mlp_import.v0`)

| Field | Meaning |
| --- | --- |
| `dtype` | `float32` or `float64`; float32 values must be exactly representable in float32 |
| `tensor_layout` | `out_by_in` (PyTorch `nn.Linear.weight`) or `in_by_out` (transposed). Required; orientation is never guessed |
| `inputs`, `outputs` | Named ports with kind, unit and exact rational domain bounds (strings) |
| `layers` | `{activation, weight, bias}`; hidden `relu` and a final `linear` only |
| `preprocessing`, `postprocessing` | Required; only `[]` (identity) is accepted in G3a |
| `provenance` | Nonempty free-form record: training seed, data hash, notes |

Every tensor value is a `0x`-prefixed hex float string as produced by `float(x).hex()`
(NumPy float32 values need upcasting with `float(x)` first). Hex floats are exact, so
import is lossless and each weight is an exact dyadic rational. Bare decimals such as
`"1.5"` are rejected because `float.fromhex` would silently read them as hexadecimal;
NaN, infinity and values that overflow the declared dtype are rejected.

`skillforge.wspl_import.import_bundle(bundle)` returns an ordinary `ReLUArtifact`:

- `compiler: wspl.mlp_import.v0` and `allocation.mode: imported`.
- **`source_sha256` is the canonical bundle hash, not a scalar-IR hash.** `wspl_equiv`
  therefore returns INVALID and `ParsedModule.contract_suite` raises `ValueError` on
  imported artifacts, which is correct: they have no WSPL source.
- The stored layers are exactly the imported tensors (transposed for `in_by_out`).
  The source map records dtype, layout, per-tensor SHA-256 of the original values and
  the provenance record.

Because the result is a `ReLUArtifact`, `wspl_verify` contracts run on imported
networks unchanged. The one change to `wspl_artifact.py` is accepting allocation mode
`imported` alongside `fresh_mlp`; other modes are still rejected. That change alters
the file's hash, which earlier records store as `evaluator_source_sha256`
(contract_fixture_results.json), `artifact_evaluator_sha256`
(equivalence_fixture_results.json) and the `src/skillforge/wspl_artifact.py` entry of
validation.json's `source_sha256` map. Those records describe the previous version of
the file; import_fixture_results.json records the new one.

`bundle_from_artifact(artifact, provenance=...)` re-expresses an existing artifact's
stored weights as a bundle; it is how the fidelity fixtures were made.

## Relational contracts (`wspl.relational_contract.v0`)

`skillforge.wspl_relational.check_relation(parent, edited, contract)` encodes both
stored graphs over one shared input point and checks a Boolean combination of affine
comparisons over `input.*`, `parent.*` and `edited.*` ports. The schema binds both
artifact hashes. Parent and edited must have identical input ports (names, order,
kinds, units and bounds); their outputs are namespaced separately.

- Domain SAT plus violation UNSAT: PROVED under the recorded trust model.
- Violation SAT whose witness, replayed through both artifacts' rational evaluators,
  satisfies the assumptions and violates the conclusion: REFUTED, reporting the
  inputs and both output records.
- Budget, capacity or backend limits, or a failed replay: UNKNOWN.
- Hash or input-port mismatch, unknown ports, unit errors or a vacuous region: INVALID.

`preservation_contract(parent, edited, region=..., outputs=..., tolerance=...)` builds
the common case: on an input region (default everywhere), each named output of the
edited network equals the parent's, or stays within plus or minus a tolerance. Repair
targets such as "edited.y0 <= parent.y0 - 1/1024 on the target region" are written
directly as contracts.

## Use

```sh
mkdir -p work
python -m skillforge.wspl_import examples/wspl/brake_patch.compiled_as_import.json --output work/brake_imported.json
python -m skillforge.wspl_relational PARENT.json EDITED.json SUITE.json --output work/relational.json
```

A suite is `{"format": "wspl.relational_suite.v0", "parent_sha256", "edited_sha256",
"contracts": [{"name", "assumptions", "ensures"}]}`. Outputs are created exclusively.
Relational exit codes: 0 PROVED, 2 REFUTED, 3 UNKNOWN, 1 INVALID/error.

## Scope and limits

These are rational real-arithmetic statements about stored coefficients under a
trusted solver, encoders and replay evaluators, with no independently checked proof
trace. They are not IEEE/FP32 execution guarantees: the float runtime's rounding and
evaluation order are not modelled. A wrong `tensor_layout` is detected only when
layer shapes are not square. Only hidden-ReLU/final-linear MLPs are supported; the
relational checker caps the combined graphs at 20,000 parameters by default. No
patch composition, preprocessing composition, transaction rollback or learned
baseline exists yet.
