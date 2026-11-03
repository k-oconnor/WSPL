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
in Torch do not remove arithmetic rounding. Numerical enclosures, independent
source equivalence, relational baseline/patch queries and closed-loop dynamics
are separate future work.
