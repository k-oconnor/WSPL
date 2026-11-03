# First WSPL scalar reference prototype

Implemented November 3, 2026 in src/skillforge/wspl_core.py. This is a typed JSON
IR reader and exact rational reference interpreter, an initial part of G0. It is
not the proposal's textual parser, full type/dimension system, neural compiler,
contract solver, deployed controller or blinded LLM experiment. No wspl CLI is
advertised as available.

## Implemented subset

- Explicit scalar Real/Truth inputs with names, unit labels and exact intervals.
- Ordered acyclic named lets/outputs; no forward references or duplicate names.
- Exact integer, decimal-string and fraction-string constants. Floats and implicit
  Boolean numeric values are rejected, rather than silently treated as exact.
- Constant-coefficient affine expressions, ReLU, clipping, normalization margin,
  min/max, graded not/and/or. No arbitrary code execution, dynamic loops,
  variable multiplication/division, executable Boolean branching or nonlinear
  activation is accepted.
- Conservative interval inference checks declared output/Truth ranges. A range
  not established by this simple analysis is rejected with an unresolved range
  obligation; it is not assumed true. Correlated cancellation such as x-x can
  require a later solver even when mathematically valid.
- Unit labels must match in affine sums/min/max. margin normalizes using bounds
  in the operand's unit and returns dimensionless Truth. This prototype does not
  implement dimensional products, unit conversions, tensors or shape algebra.
- Evaluation returns fractions, independently of PyTorch and any future compiler.
  There is no deployed dtype, numerical error enclosure or serializer yet.

Examples: examples/wspl/brake_patch.core.json and approach_p.core.json. Both are
inspected engineering examples and excluded from final experimental data.
BrakePatch uses dimensionless normalized quantities. ApproachP has the twelve
motor12 input names and explicit position/velocity/aperture units; it normalizes
target deltas before returning bounded dimensionless actions and gripper -1.
It represents the known proportional program, not a newly synthesized skill.

```python
import json
from pathlib import Path
from skillforge.wspl_core import ScalarProgram

program = ScalarProgram(json.loads(
    Path("examples/wspl/brake_patch.core.json").read_text(encoding="utf-8")))
result = program.evaluate({"d": "1/5", "v": "2/5", "a_base": "1"})
assert str(result["a"]) == "-1"
```

The eight new unit tests cover exact arithmetic on finite boundary/interior
fixtures, approach-rule agreement, non-idempotent graded conjunction, unit and
scope failures, invalid constants/normalization, domain violations, conservative
range obligations, document snapshot isolation and unsupported profiles. They do
not establish universal BrakePatch contracts by testing sample points. No solver
status or proof certificate is produced.

Next implement and cross-check the textual module front end or explicitly freeze
the JSON surface for a first language profile; then canonical lowering, neural
weight allocation/serialization and independent scoped verification (G1/G2).
Do not count the old hand-written weight constructors as this compiler.
