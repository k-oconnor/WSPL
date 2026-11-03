# Output-patch composition and transactions (G3b)

Added November 3, 2026 (issue #8). G3b completes the engineering side of the S2
prerequisites with IMPORT.md: **output-patch composition**, **transactions with
rollback** and **import pre/postprocessing**. All of it is augmentation, labelled
as such. No network was trained; the fixtures are compiled stand-ins or untrained
synthetic networks, never S2 baselines.

## Current engineering results

| Fixture | Result |
| --- | --- |
| Compiled BrakePatch composed onto the compiled stand-in [Cruise](../../examples/wspl/cruise_baseline.wspl) baseline through `a_base <- parent.a` | Whole composed artifact solver-EQUIVALENT to one WSPL module written as baseline-then-patch; every parent and patch tensor verified as an exact slice |
| Same, as a transaction with the BrakePatch actuator/emergency contracts and clear-path preservation | COMMITTED: the `a_base` interface obligation, both I/O contracts and the relational preservation contract PROVED |
| Untrained synthetic parent whose output can leave [-1, 1] | REJECTED: interface obligation REFUTED with a replayed witness; parent returned unchanged |
| A false "unchanged everywhere" claim | REJECTED with a counterexample |
| Zero budget | UNRESOLVED (all UNKNOWN), not REJECTED |

See the [fixture results](patch_fixture_results.json).

## `compose_output`

`skillforge.wspl_compose.compose_output(parent, patch, bindings=..., replace=...)`.

- **Bindings** map every patch input to `input.<parent input>` or
  `parent.<parent output>`, each source at most once. Units must match. A parent
  input's domain must lie inside the patch input's domain (checked statically).
- **Replace** maps every patch output to a distinct parent output with the same kind
  and unit. Unreplaced parent outputs pass through. New outputs are not supported.
- **Edited interface:** the edited inputs are exactly the parent inputs, and the edited
  output types are the parent's declared types. The patch's own output types are
  recorded in the source map.

The composed network is one hidden-ReLU/final-linear artifact (`allocation.mode:
augmented`, compiler `wspl.compose_output.v0`, `source_sha256` = hash of the
composition record naming both artifact hashes, bindings and replacements):

1. Parent hidden layers, copied verbatim, alongside signed relays `relu(x)`, `relu(-x)`
   for parent inputs the patch reads.
2. A boundary layer `relu(Wh+b)`, `relu(-(Wh+b))` replacing the parent's final linear
   layer, for parent outputs the patch reads or that pass through.
3. The patch's layers, reading each value as `relu(+) - relu(-)`, with passthrough
   outputs carried on signed relays.
4. A final linear layer: replaced outputs from the patch, the rest from passthrough.

No two tensors are multiplied together, so every parent and patch weight appears
exactly (or negated, on the `relu(-)` channel) in the composed weights.
`verify_placement(edited, parent, patch)` checks every composed row against its
source tensor using the row tags in the source map. The source map also reports
added depth and added parameters.

`interface_obligations(parent, patch, bindings)` produces, for each parent output that
feeds a patch input, an I/O contract on the parent: the output stays inside that
patch input's declared bounds. Declared output types are not guarantees; the proposal
requires the composed graph to "establish the bound on a_base".

## Transactions

`skillforge.wspl_transaction.patch_transaction(parent, patch, bindings=, replace=,
io_contracts=, relational_contracts=)` composes against the pinned parent, verifies
placement and then checks:

- the generated interface obligations, on the parent;
- user I/O contracts (`{name, assumptions, ensures}` over `input.*`/`output.*`), on the
  edited network;
- user relational contracts (over `input.*`, `parent.*`, `edited.*`), via
  `wspl_relational`.

| Decision | Rule | Returned artifact |
| --- | --- | --- |
| COMMITTED | every result PROVED | edited |
| REJECTED | any result REFUTED or INVALID, or the composition is invalid | parent, unchanged |
| UNRESOLVED | nothing refuted or invalid, some result UNKNOWN | parent, unchanged |

REJECTED and UNRESOLVED are kept apart so that counterexamples and budget or backend
limits can be counted separately. The parent's hash is re-checked before returning.

The `wspl.patch_transaction.v0` manifest records the parent, patch and candidate
hashes, bindings, replacements, the contract templates of each kind and their counts,
every result, the decision rule, added depth and parameters, and the number of
placement rows checked. A manifest-body hash and a digest of each result's kind,
name, status and contract hash protect the record. Whether a commit needs any
behavioural contract is S2 policy: a commit backed only by interface obligations is
allowed, and visible as `contract_counts` `io: 0, relational: 0`.

`replay_transaction(manifest, parent, patch, rerun=False)` recomposes from the manifest
and raises `ValueError` if the record was altered, the artifacts differ, the candidate
hash differs or the bound contracts differ. With `rerun=True` the contracts are checked
again and their statuses must match.

```sh
python -m skillforge.wspl_transaction PARENT.json PATCH.json SPEC.json --manifest-output work/manifest.json --artifact-output work/edited.json
```

The spec is `{"format": "wspl.patch_spec.v0", "bindings", "replace", "io_contracts",
"relational_contracts"}`. The manifest is always written; the edited artifact only on
COMMITTED. Exit codes: 0 COMMITTED, 2 REJECTED, 3 UNRESOLVED, 1 error.

## Scope and limits

Composition and contracts are rational real-arithmetic statements about stored
coefficients under trusted solver, encoders and replay evaluators; they are not IEEE
execution or physical guarantees. Only `compose_output` exists. The proposal's
`replace_module` and `add_residual` are not implemented (a residual edit can be
written as a patch reading the parent output), and nor are in-place mask-limited
edits or gradient repair. Augmentation adds depth and parameters, reported in the
manifest; it is not a fixed-architecture edit. A successful transaction does not carry
over certificates of the parent; contracts must be rechecked on the edited network.
