# WSPL text front end (Core PWA module form)

Added November 3, 2026 (issue #3). `skillforge.wspl_parse` accepts the module form
from the archived proposal's "First executable example" and lowers it to the existing
`wspl.scalar_ir.v0` JSON plus `wspl.io_contract_suite.v0` contracts. This is the
proposal's milestone M0 front end and the G0 "text parsing" item. It accepts only
that module form and Core PWA expressions. It is not a general `wspl` command
suite, and parsing establishes no contract, equivalence or execution claim.

## Current engineering results

| Text | Result |
| --- | --- |
| [brake_patch.wspl](../../examples/wspl/brake_patch.wspl), the proposal text verbatim | IR agrees with `brake_patch.core.json` on a rational grid; compiled layers identical to the committed artifact; all three parsed contracts PROVED |
| [approach_p.wspl](../../examples/wspl/approach_p.wspl) | IR identical to `approach_p.core.json`; bound contract suite identical to `approach.contracts.json` |

The BrakePatch IR trees differ: the proposal writes `clip((0.4 + 0.5*v - d)/0.4, 0, 1)`
where the committed JSON uses `margin(0.4 + 0.5*v - d, 0, 0.4)`. Both lower to the
proposal's stated first-layer weights, rows `[-5/2, 5/4, 0]` with biases `1` and `0`,
plus the `a_base` relay units. The parsed contracts REFUTE the known reduced-gain
and swapped-column regressions.

## Accepted language

```text
module Name {
  input  x: Real[lo, hi] unit_label;   // or: input p: Truth;
  output y: Real[lo, hi] unit_label;
  let t: Truth = margin(x, -0.5, 0.5); // Real lets: Real[lo, hi] with optional unit
  y = clip(20*t - 10, -1, 1);          // each output assigned exactly once

  ensures name:   forall inputs: FORMULA;
  preserves name: forall inputs: FORMULA;
}
```

- **Constants** are exact decimals (`0.4`) or constant expressions (`2/5`). They are
  written to the IR as canonical fraction strings, so `0.4`, `0.40` and `2/5` give
  identical IR and hashes. Scientific notation is rejected.
- **Expressions:** `+`, `-`, multiplication by a constant, division by a nonzero
  constant, `relu`, `clip(x, lo, hi)`, `margin(x, lo, hi)`, `min`, `max`, and the
  Lukasiewicz `not`, `and`, `or`, `implies`. Constant-coefficient sums are merged
  into one exact affine expression; a single unit-coefficient term with no bias is
  emitted bare. `implies(p, q)` has no IR operator; the parser rewrites it to
  `or(not(p), q)`, which equals `min(1, 1 - p + q)` exactly, and records the
  rewrite in the parse report.
- **Scope:** expressions may reference inputs and earlier lets. Outputs cannot be
  referenced in v0 expressions, and forward references are rejected.
- **Contracts:** Boolean combinations (`and`, `or`, `not`, `implies`, `true`, `false`)
  of affine comparisons (`<=`, `<`, `>=`, `>`, `==`) over input and output ports.
  Lets and function calls are not allowed in contracts. A top-level `P implies Q`
  becomes assumptions `P` and ensures `Q`; any other formula has assumptions `true`.
- **`preserves`** is parsed as an ordinary I/O contract, with its kind recorded in the
  parse report. The proposal's baseline-versus-edited relation is not implemented.
- **Rejected with a line:column error:** `requires`, `invariant`, `empirical`,
  `state`, `function`, `import` and `tensor` declarations; products or divisions of
  varying values; unknown functions; non-constant clip/margin bounds; units that
  differ in a sum or comparison; missing bounds or units; duplicate declarations or
  contracts; unassigned or twice-assigned outputs; nesting beyond 64 levels.

The produced IR is validated by `ScalarProgram`, so its interval range
obligations, Truth ranges and naming rules apply unchanged. Those errors are
reported as `IR validation failed: ...` without a source position.

## Units: reading of "explicit normalization"

The proposal says each real type records "physical unit or explicit normalization".
Its BrakePatch labels quantities `normalized_distance`, `normalized_speed` and
`normalized_acceleration`, then adds a Truth value to a normalized acceleration.
The IR only allows sums of identical unit labels. The parser therefore reads a
label starting with `normalized_` as an explicitly normalized, dimensionless
quantity (IR unit `"1"`), keeping the label in the parse report. Any other label
(`m`, `m_per_s`, ...) is a physical unit, kept verbatim and subject to the strict
same-unit rule. Units are required on Real inputs and outputs and optional on lets,
where they are inferred from the expression. This rule lives in `ir_unit()` and is
open for Kevin's decision in issue #3.

## Use

```sh
mkdir -p work
python -m skillforge.wspl_parse examples/wspl/brake_patch.wspl --ir-output work/brake.core.json --report-output work/brake.parse.json
python -m skillforge.wspl_compile work/brake.core.json --output work/brake.relu.json
python -m skillforge.wspl_parse examples/wspl/brake_patch.wspl --ir-output work/brake2.core.json --artifact work/brake.relu.json --contracts-output work/brake.contracts.json
python -m skillforge.wspl_verify work/brake.relu.json work/brake.contracts.json --output work/brake.results.json
```

Contracts bind only to an artifact compiled from the parsed IR: its `source_sha256`
must equal the report's `ir_sha256`. Outputs are created exclusively; the CLI exits
0 on success and 1 on any parse, validation or binding error. The Python API is
`parse_module(text)`, which returns the IR (`program`), `contracts`, `report` and
`contract_suite(artifact)`.

## Limits

The front end is a hand-written recursive-descent parser with a 1 MB source limit.
It executes no Python from the source text. It does not implement tensors, shapes,
functions, stateful modules, baseline/edited `preserves`, `requires` domains or
numerical deployment profiles. The S1 authoring screen in FORMAL_EXPERIMENTS_V1.md
uses the typed scalar JSON IR for its WSPL arm; this text front end does not change
that study's design.
