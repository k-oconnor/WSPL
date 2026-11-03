# Initial real-graph contract verifier

Implemented November 3, 2026. `skillforge.wspl_verify` checks bounded numeric I/O
contracts against the **serialized ReLU MLP artifact**, not the compiler's source
expressions. It is an initial G2 prototype using optional Z3, tested with
z3-solver 5.3.3.0. No trained-controller import or scientific synthesis study ran.

## Current engineering results

The recorded [fixture results](contract_fixture_results.json) establish under
the declared rational-real graph semantics:

| Artifact | Contract | Result |
| --- | --- | --- |
| BrakePatch | actuator bounds | PROVED |
| BrakePatch | emergency region forces a=-1 | PROVED |
| BrakePatch | clear-path output equals bounded a_base | PROVED |
| ApproachP | actuator bounds and open gripper | PROVED |
| ApproachP | command has the sign of target error | PROVED |

Every assumption domain has a replayed non-vacuity witness. Regressions reducing
the braking gain or swapping input columns return REFUTED with a rational
counterexample independently replayed through `ReLUArtifact.evaluate`. Stale
artifact hashes, empty domains, unit errors and unknown ports return INVALID.
Zero/exhausted budgets, unsupported numerical profiles, missing Z3 or capacity
limits return UNKNOWN; they are not proofs or evidence of physical infeasibility.

These are solver-backed universal I/O statements over the bounded real graph,
not finite random-probe claims. They are **not** source-to-artifact equivalence
proofs, IEEE execution proofs, semantic sensor grounding, robot invariants or
collision avoidance. ApproachP's direction property does not prove convergence.

## Query and trust model

The schema binds a canonical artifact hash, assumptions and conclusion. Each
affine form has exact rational coefficients, a bias, explicit unit label and named
`input.*` or `output.*` references. Only matching units with dimensionless
coefficients are supported; physical conversion factors need explicit supported
normalization, not implicit dimensional multiplication. Boolean combinations use
le/lt/ge/gt/eq, and/or/not/implies. There is no executable Boolean source branch.

The encoder uses exact rational Real terms, affine equalities and ReLU If terms
in QF_LRA. It first asks whether the artifact input box, graph and assumptions
are satisfiable, then asks whether they admit a violation of the conclusion.

- Domain SAT plus violation UNSAT: PROVED under the recorded trust model.
- Violation SAT plus independently replayed assumptions and violated conclusion:
  REFUTED, with validated witness.
- Solver/capability/resource uncertainty: UNKNOWN.
- Malformed/mismatched/vacuous queries: INVALID.

The stored result records artifact/contract digests, verifier and rational
evaluator source hashes, solver version/logic, query hashes, non-vacuity witness,
budget and scoped statuses. SAT/UNSAT explicitly refer to a **violation** query.
The solver, encoder, schema/type interpretation and rational replay remain trusted
components. There is no independently checked UNSAT proof trace; this is not a
fully proof-carrying artifact. Non-vacuity alone does not establish useful domain
coverage, good requirements or physical calibration.

The implementation caps graph size at 10,000 parameters by default and uses one
declared total query budget. It is not a scalable arbitrary-network verifier.
Unsupported operators cannot be silently encoded as ReLU. The separate
intervention evidence path handles empirical investigations without relabelling
them PROVED.

## Use

```sh
python -m pip install -e ".[dev,verify]"
mkdir work
python -m skillforge.wspl_verify examples/wspl/brake_patch.relu.json examples/wspl/brake.contracts.json --output work/brake_contract_results.json
```

Outputs are created exclusively. CLI exit codes: 0 PROVED, 2 REFUTED, 3 UNKNOWN,
1 INVALID/error. In a suite, any invalid/refuted/unknown contract prevents the
suite from being PROVED; inspect per-contract results, not only its aggregate.

Python APIs are `verify_contract(artifact, contract, timeout_ms=...)` and
`verify_suite(artifact, suite, ...)`. JSON examples are in examples/wspl. These
engineering fixture checks make no model calls, collect no scientific test set
and perform no robot execution. New scientific evaluations still use the source
approval gate.

Z3 mathematical Reals are distinct from machine floats
([official arithmetic guide](https://microsoft.github.io/z3guide/docs/theories/Arithmetic/));
the [Z3Py guide](https://ericpony.github.io/z3py-tutorial/guide-examples.htm) describes
exact rational constants, models and solver outcomes. Exact stored coefficients
in Torch do not remove arithmetic rounding. Numerical enclosures, relational
baseline/patch queries and closed-loop dynamics are separate future work. Initial
source equivalence is described below.

## Source-to-artifact equivalence

Added November 3, 2026 (issue #1). `skillforge.wspl_equiv` asks whether a
serialized ReLU artifact computes **exactly** the outputs of its scalar IR source
for every input in the declared box, under rational real semantics. The contract
verifier above checks what the weights do; this check addresses whether the
program is an accurate explanation of those weights.

| Source | Artifact | Result |
| --- | --- | --- |
| BrakePatch | committed `brake_patch.relu.json` | EQUIVALENT |
| ApproachP | committed `approach_p.relu.json` | EQUIVALENT |

See the [fixture results](equivalence_fixture_results.json). Reduced braking
gain, swapped input columns, a 1/10^6 readout bias change, and artifacts compiled
from programs with swapped min/max or and/or connectives all return DIFFERENT
with replayed counterexamples. Of 29 single-coefficient BrakePatch mutations
(+1 each), 26 are DIFFERENT. The other three are EQUIVALENT, and a finite grid
probe agrees with that: these units do not affect the output on the declared domain.

### Query

1. The source is validated by `ScalarProgram`. Its canonical SHA-256 must equal
   the artifact's `source_sha256`, and the artifact's ordered input/output names,
   kinds, units and rational bounds must equal the source declarations. Otherwise
   the result is INVALID. Swapped weight columns under correct port names are not
   an interface error; the query finds them.
2. The source expression tree is encoded into Z3 terms directly from the scalar
   IR semantics (affine sums; `If` terms for relu/clip/margin/min/max and the
   Lukasiewicz connectives). The checker does not import `wspl_compile`; a
   compiler lowering error therefore cannot be reproduced on both sides.
3. The stored layers are encoded exactly as in the contract verifier, over the
   same input variables.
4. One QF_LRA query asks for an input in the box where any source output differs
   from the corresponding artifact output.

- Difference query UNSAT: EQUIVALENT under the recorded trust model.
- SAT with a rational witness on which `ScalarProgram.evaluate` and
  `ReLUArtifact.evaluate` produce different outputs: DIFFERENT, reporting both
  output records and the differing ports.
- Solver/budget/capacity/backend uncertainty or a witness that fails replay: UNKNOWN.
- Malformed source, broken identity binding or interface mismatch: INVALID.

### Scope and trust

Trusted components are the scalar IR validator, both Z3 encoders, Z3 itself and
the two rational interpreters used for replay. There is no independently checked
UNSAT proof trace. The source encoder is a second implementation of the IR
semantics, cross-checked against `ScalarProgram.evaluate` on a grid of a program
using every operator; it is not derived from a formal semantics. Equivalence
covers only the declared input box and the stored rational coefficients. It does
not establish FP32/FP64 execution, correctness of the source map, preprocessing,
contract adequacy or physical behaviour. The contract verifier's
`source_equivalence` field still reports `not_checked`; run this check separately
and keep both reports.

```sh
mkdir work
python -m skillforge.wspl_equiv examples/wspl/brake_patch.core.json examples/wspl/brake_patch.relu.json --output work/brake_equivalence.json
```

Exit codes: 0 EQUIVALENT, 2 DIFFERENT, 3 UNKNOWN, 1 INVALID/error. Outputs are
created exclusively. The Python API is
`check_equivalence(source_document, artifact, timeout_ms=..., max_parameters=...)`.
