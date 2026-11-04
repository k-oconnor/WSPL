# WSP editor first pass

Historical milestone: the [graph surgery extension](WSP_GRAPH_SURGERY.md) now
supersedes the fixed-topology limit below. This page and its validation receipt
retain the original first-pass scope and test counts.

Implemented November 3, 2026. The first WSPL 0.2 editor supports direct FP32 tensor
programming, explicitly placed affine/ReLU circuits, and descriptions/checks of
the resulting frozen artifact. It is a separate frontend; historical Core PWA
and scientific run implementations are unchanged.

The [design](WSP_TOOLING_DESIGN.md) remains the larger target. This page describes
what ran at that milestone. There are no hosted model calls, training runs, simulator
episodes or model deployment in this implementation check.

## Available operations

| Operation | Implemented behavior |
| --- | --- |
| `inspect` | Artifact identity, graph/layout, tensor shapes, capability flags, paged exact values |
| `plan` | Parse and execute a bounded edit on private storage; record source, parent/candidate identities, complete write footprint, rounding and placement reports |
| `apply` | Replay a plan against the expected immutable parent; refuse stale implementation, stale parent or altered receipt/candidate; return a staged artifact |
| `describe` | Bind views/hypotheses/claims without changing weights; alternatively render equations directly from stored coefficients |
| `check` | Optional Z3 real-graph checks over actual stored coefficients, non-vacuity and independently replayed rational counterexamples |
| `probe` | Evaluate a small explicit fixture and report inputs/outputs/internal taps as empirical evidence |
| `diff` | Paged bit-level changes, counts and tensor identities; no implied behavioral preservation |

These are Python APIs and CLI subcommands, not registered remote/MCP tools. There
is no active-model selection or admission API yet. Applying a plan produces a
candidate; it does not deploy it or weaken any external requirement.

## Run the worked example

Use Python 3.11 and the editor/verification dependencies:

```sh
python -m pip install -e ".[dev,editor,verify]"
python tools/demo_wsp_editor.py --output work/my_editor_demo
```

Choose a new output directory. The demo compares
[explicit placement](../../examples/wsp_editor/deadband.wspl) with
[literal tensor construction](../../examples/wsp_editor/deadband_raw.wspl), then
checks a true equation and a deliberately false claim against the frozen weights.
It is an inspected engineering fixture, not an LLM synthesis or performance test.

The local completed demo in `work/wsp_editor_first_pass_demo` reports identical
raw/placed artifacts, true claim PROVED and false claim REFUTED with witness
replay. Artifact identity:
`db210e7e8b153ae25b0a0dacaa11905e67cc090b26b6cd68850625ffbb2b4940`.

The corresponding step-by-step workflow, after creating `work` if needed:

```sh
python -m skillforge.wsp_editor plan examples/wsp_editor/deadband.wspl --output work/deadband.plan.json
python -m skillforge.wsp_editor apply work/deadband.plan.json --output work/deadband.artifact.json
python -m skillforge.wsp_editor inspect work/deadband.artifact.json --tensor W1
python -m skillforge.wsp_editor describe work/deadband.artifact.json --layer 1
python -m skillforge.wsp_editor describe work/deadband.artifact.json --source examples/wsp_editor/response.description.wspl --output work/deadband.description.json
python -m skillforge.wsp_editor check work/deadband.artifact.json work/deadband.description.json --output work/deadband.check.json
python -m skillforge.wsp_editor probe work/deadband.artifact.json examples/wsp_editor/inputs.json --output work/deadband.probe.json
```

Every output is created exclusively. `apply` can take `--parent` to require an
externally retained parent file; otherwise it uses the exact parent stored in
the plan. Both paths replay the complete plan and check its identity. A successful
CLI `check` exit means a report was produced; inspect its status, which can be
PROVED, REFUTED, UNKNOWN, INVALID, UNSUPPORTED or NO_CLAIMS.

## Tensor programming

The fixed editor handles named finite FP32 tensors of rank one through eight,
including banks whose inference graph is opaque. Dense adapters support either
`in_out` or `out_in` weight orientation and linear, ReLU or tanh activations.
Opaque banks can be edited and diffed; they do not acquire an execution adapter.
Imports require explicit float32 arrays. Shared storage is rejected rather than
silently untied. Whole-model PyTorch import, aliases, mixed dtypes, state and
arbitrary graph execution remain future adapters.

Supported construction features include basic integer/negative/strided slices,
broadcast writes, `= += -= *=`, statement-snapshot copies, immutable `parent.*`
reads, constants, bounded construction loops, arithmetic, matrix multiplication,
`copy`, `transpose`, `reshape`, `zeros`, `full`, `eye` and hashed tensor bindings.
The Python API accepts external bindings; the CLI can use inline constant tensors.
No proposal Python is executed.

`exact_to_storage` parses rational/decimal/scientific constants exactly and rounds
each assignment directly to FP32, nearest-even. There is no intermediate FP64
rounding. `f32bits(...)` and direct copies preserve exact bits, including signed
zero. Arithmetic on stored-bit values uses their exact rational value; rational
zero arithmetic does not retain the sign of zero. Nonfinite writes are rejected.
Literal conversion changes are counted and up to 64 examples are retained.

`native_numpy_v1` supports literal statements using the historical numeric rules
and records the NumPy version. Three complete archived native responses reconstruct
bit-for-bit in this profile. New transactions reject an invalid statement
atomically. `import_legacy(answer)` separately preserves the historical
skip-invalid convention and returns its explicit skipped-statement receipt.

Resource budgets cover tensor elements, expression size, rational precision,
construction work, expanded statements and loop iterations. They are configurable
through the Python API. This is a small local editor, not a hardened public
multi-tenant execution service.

## Circuit placement

The working Deadband source is exactly the form proposed in the design. Each
assigned scalar node is placed into an existing compatible producer. A circuit
body can use its declared scalar ports, earlier local signals and construction
constants. Target signals are connected explicitly at the placement site.

This first lowering profile supports affine expressions and one outer ReLU per
named assignment. It does not automatically lower nested ReLU, clip, min/max or
arbitrary DAGs into the target. Write explicit gates and placements for those
constructions; raw tensors remain available throughout.

Each placed node replaces its complete incoming weight vector and bias, including
old off-region inputs. Other producers retain their stored bits. Existing outgoing
edges to consumers outside the placement are preserved and reported. Missing skip
edges, incompatible activations, missing placements and overlapping placements
fail rather than insert relays or expand the network.

`Real` has no asserted finite bound. Input/output interval refinements are recorded
as UNVERIFIED obligations and are never used to justify lowering. Intermediate
refinements beyond unbounded Real are not supported yet. A later raw edit marks
earlier placement descriptions as potentially stale, conservatively; fine-grained
dependency-based evidence reuse is future work.

## Descriptions and evidence

A description binds to the complete artifact identity and has its own separate
identity. Views can name actual pre/postactivation taps; hypotheses stay
UNVERIFIED. Exact stored-coefficient equations are available without guessing
semantic labels.

The optional checker supports affine/ReLU real-graph formulas, interval assumptions,
Boolean combinations, comparisons and PWA expressions over inputs, outputs and
views. It verifies non-vacuity and replays witnesses through the independent
artifact evaluator. Tanh remains editable and can execute under the NumPy FP32
adapter, while the formal backend reports UNSUPPORTED. Other unsupported claim
syntax is rejected; it is not silently approximated.

Checks record artifact/description/claim/query identities, checker/evaluator source
hashes and the solver version. This is a trusted-solver result over exact rational
interpretations of stored finite coefficients, not an IEEE execution certificate,
a physical guarantee or evidence that a semantic label is correct.

`origin=external_requirement` records caller-supplied provenance; it is not an
authentication mechanism. A future host-owned admission service must retain the
independent requirements. The first-pass editor performs no admission, and a
description with no claims produces NO_CLAIMS rather than a blanket proof.

## Validation and remaining work

All 50 focused editor checks pass. The full suite contains 339 checks: 338 pass
in the primary environment, with one optional native fixture skipped there and
covered by 12 passing replay checks in the separate MinAtar environment.
The focused suite covers exact FP32 rounding (including double-rounding traps and
subnormals), arbitrary tensor writes, native compatibility, statement snapshots,
atomic failures, stale/tampered plans, placement footprint, shared consumers,
immutable descriptions, false/vacuous claims and every CLI subcommand. See
[the validation receipt](wsp_editor_first_pass_validation.json) for exact counts,
environment and source identities.

The next implementation slice is complete graph regions and boundary-preserving
replacement: shared parameters, node splitting, transplant/redirect/residual edits
and explicit topology budgets. Automatic sign-aware allocation, broader circuit
lowering, numerical execution proofs, selective evidence reuse and large-model
adapters also remain open. No new scientific experiment was prepared or launched.
