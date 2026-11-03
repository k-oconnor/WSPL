# WSPL: Weight Space Programming Language

**A typed programming interface for LLM-authored neural programs.**

WSPL investigates whether an LLM can construct, inspect and repair useful neural
controllers through programs with explicit semantics, rather than manipulating
opaque tensors alone. The intended executable artifacts are neural weights.
**Skillforge** is the robotics and reusable-skill testbed in this repository.

This is an early research prototype led by Kevin O'Connor. We distinguish a
program matching its weights, a controller satisfying an input-output contract,
and a robot satisfying a trajectory property. Each needs its own evidence.

## Current status

| Component | Status |
| --- | --- |
| Typed scalar JSON IR and exact rational interpreter | Implemented; engineering examples and negative checks |
| WSPL text front end (proposal module form) | Initial parser to JSON IR and contract suites; Core PWA only |
| Scalar IR to ReLU MLP weights, source map and independent loader | Initial implementation; fresh architecture only |
| Small real-graph I/O contract verifier | Initial Z3 prototype; non-vacuity and rational witness replay |
| Source IR to stored-weight equivalence check | Initial Z3 prototype; compiler-independent encoding, replayed differences |
| Numerical (IEEE) execution proof | Planned |
| Frozen MLP tensor import and parent/edited relational contracts | Initial G3a prototype; exact hex-float import, Z3 relational checks; no patch composition yet |
| Opaque-network intervention evidence | Paired activation probes, controls, replay and hash-chained empirical logs |
| Blinded LLM authoring and counterexample-feedback harness | Planned |
| Small-alphabet LTLf compiler and strong finite-model planner | Implemented for supplied symbolic effect models |
| Hierarchical task expansion | Implemented at a supplied macro-summary cut; no physical primitive executor |
| Neural MLP/RMV blocks and single-tick LIBERO bridge | Implemented; independent sampled outcome observers |
| Pregrasp positioning pilot | Completed; scoped results below |
| Diverse robot tasks | 16 inspected specifications over 7 audited scene assets; contact/obstacle integration pending |

The proposal's first module form now parses from text into the scalar JSON IR and
contract suites ([text front end](docs/wspl/PARSER.md)); other proposed syntax is not
implemented. The scalar JSON IR compiles neural weights and supports scoped
real-graph contract checks. There is
no numerical execution proof or blinded controller-synthesis harness. The available compiler command is documented below;
the proposal's general `wspl` command suite remains planned.

## Try the exact scalar reference

Python 3.11 or later. The reference core uses the standard library:

```sh
python -m pip install -e ".[dev]"
```

```python
import json
from pathlib import Path
from skillforge.wspl_core import ScalarProgram

program = ScalarProgram(json.loads(
    Path("examples/wspl/brake_patch.core.json").read_text(encoding="utf-8")))
result = program.evaluate({"d": "1/5", "v": "2/5", "a_base": "1"})
assert str(result["a"]) == "-1"
```

The same program can be written in the proposal's text form,
[brake_patch.wspl](examples/wspl/brake_patch.wspl), and parsed into this IR:

```python
from skillforge.wspl_parse import parse_module

parsed = parse_module(Path("examples/wspl/brake_patch.wspl").read_text(encoding="utf-8"))
assert str(ScalarProgram(parsed.program).evaluate({"d": "1/5", "v": "2/5", "a_base": "1"})["a"]) == "-1"
```

Constants and inputs are exact rational strings or integers; floating-point input
is not silently treated as an exact specification. The scalar IR supports affine
arithmetic, ReLU, clipping, normalization, min/max and graded Lukasiewicz logic.
Unit labels, bounds and acyclic references are checked. Conservative interval
analysis can reject a valid program when a stronger proof is needed.

See [the scalar core](docs/wspl/SCALAR_CORE.md),
[BrakePatch](examples/wspl/brake_patch.core.json) and
[ApproachP](examples/wspl/approach_p.core.json).

## Compile neural weights

```sh
mkdir work
python -m skillforge.wspl_compile examples/wspl/brake_patch.core.json --output work/brake_weights.json
```

Create the output directory first and choose a new file. BrakePatch compiles into
the proposed 3 -> 4 -> 2 -> 1 ReLU network, with 29 parameters. ApproachP compiles
into a 12 -> 6 -> 6 -> 4 network, with 148 parameters. Generated examples and the
independent loader are described in [the compiler notes](docs/wspl/COMPILER.md).
Optional Torch emission checks exact coefficient representation; floating-point
execution and universal contracts are not certified by compilation.

## Check contracts and investigate opaque networks

With `python -m pip install -e ".[dev,verify]"`, the initial verifier checks
serialized ReLU graphs using rational real arithmetic. The three BrakePatch
contracts and the approach bounds/direction contracts are solver-backed PROVED;
broken gain/input-order fixtures produce independently replayed counterexamples.
This does not establish source equivalence, IEEE execution or physical safety.
See [verification scope and examples](docs/wspl/VERIFICATION.md).

A separate check asks whether a compiled artifact computes exactly its scalar IR
source over the declared input box. Both committed examples are EQUIVALENT under
rational real semantics, and weight regressions return counterexamples replayed
through both interpreters:

```sh
mkdir -p work
python -m skillforge.wspl_equiv examples/wspl/brake_patch.core.json examples/wspl/brake_patch.relu.json --output work/brake_equivalence.json
```

This is not an IEEE execution proof or a proof-carrying certificate.

For networks outside that verifier, [intervention logging](docs/wspl/INTERVENTION_EVIDENCE.md)
records controlled activation replacements with baseline, no-op and repeat checks.
It supports tensor-valued PyTorch eval/stateless adapters, records identity/RNG
and output effects, and can replay/chain empirical evidence. Observed computational
effects are not semantic concepts, real-world causes or universal guarantees.

## Research program

The first proposed product workflow is **controller change review and repair**.
See the [customer validation plan and collaborator handoff](docs/PRODUCT_VALIDATION.md).
An inspected engineering demo compiles a controller, exposes a deliberate gain
regression with a replayed counterexample, and checks an operator correction:

```sh
python tools/demo_controller_review.py --output work/controller_review
```

Add `--include-opaque` in the tested Torch/NumPy environment to include paired
activation probes. This is a known-answer demonstration, not automatic repair,
customer validation or a scientific performance comparison.

The three studies cover fresh authoring, repair of independently trained artifacts
and frozen composition. They span discrete causal worlds, low-dimensional dynamics and
robot manipulation. Each compares direct weights, ordinary PWA code and WSPL,
with and without verifier feedback. All arms receive final verification.

The [experimental program](docs/wspl/EXPERIMENT_PROGRAM.md) separates mathematical
contracts, deployment numerics and physical outcomes. The proposed first screen
has 36 synthesis runs, but is not registered, funded, approved or launchable yet.
Model identity, monetary cap, frozen tasks and final thresholds remain unset.

The [formal experiment protocols](docs/wspl/FORMAL_EXPERIMENTS_V1.md) now specify
the first finite-domain authoring screen and subsequent learned repair/composition
studies. Implemented: six-arm trial kernel, common 541-parameter target, finite
verification, feedback firewall, budget/selection records, public request export
and source-gated offline grading. Isolated provider collection and frozen launch
settings remain pending; no model calls or scientific trials have run.

- [Structured study plan](experiments/plans/wspl_cross_domain_v1.json)
- [Implementation gates](docs/wspl/WSPL_IMPLEMENTATION.md)
- [Original supplied WSPL proposal](docs/wspl/PROPOSAL_v0_1_2026-11-03.md), archived unchanged with provenance
- [Related work](docs/RELATED_WORK.md); compilation and neural repair have substantial precedents

## Existing robot evidence

A small, unblinded engineering pilot compared six controllers on positioning at
a supplied standoff above an object. Each neural block had about 1,600 parameters.

| Controller | Successful evaluation starts |
| --- | ---: |
| Zero translation | 0/8 |
| Reference proportional controller | 8/8 |
| Analytic MLP weights encoding the reference | 8/8 |
| Analytic RMV weights encoding the reference | 8/8 |
| AdamW MLP, four demonstrations | 5/8 |
| AdamW RMV, four demonstrations | 3/8 |

The complete attempt took 56.266 recorded CPU seconds. RMV/scalar discrepancies
on recorded visited inputs were below 3e-8. This is empirical numerical agreement,
not a universal floating-point guarantee. Earlier failed attempts and costs are
retained in the [audited report](docs/APPROACH_V1_RESULTS.md).

These were inspected analytic encodings of a known controller, not a blinded LLM
synthesis result. The task was pregrasp positioning, not grasping, pickup, general
transfer or physical hierarchy execution. Four demonstrations did not reliably
teach the same closed-loop behavior; a general LLM-over-gradient advantage is not
established.

## Robotics testbed

The selected stack is MuJoCo for physics, robosuite for robot/control infrastructure,
and LIBERO for tasks and assets. The local adapter was checked against MuJoCo
3.2.7, robosuite 1.4.1 and a pinned LIBERO checkout. Simulation is optional and
must be installed/configured separately; reference tests do not launch it.

- [System specification](docs/SYSTEM_SPEC.md) and [adapter contracts](docs/SIM_ADAPTER.md)
- [Sixteen robot task specifications](docs/ROBOT_TASK_SUITE.md): repeated grasps,
  sorting, drawer/door chains, barriers, blocker clearance, timed holding and recovery
- [Symbolic hierarchy fixtures](docs/HIERARCHICAL_TASKS.md)
- [LTLf compilation and strong planning](docs/M1_ENGINEERING.md)

Obstacle/articulation/orientation inputs, contact-ready skills, independent
physical observers and the primitive executor remain explicit integration work.
An existing BDDL scene is not a working Skillforge task.

## Checks and reproducibility

```sh
python -m unittest discover -s tests -v
```

The current local suite has 183 checks (14 added for source equivalence); all pass
with Z3 installed, with 37 skipped when PyTorch/NumPy are absent. Optional neural/adapter checks
skip when PyTorch or NumPy is absent, and solver checks skip without Z3; full local results used a separate CPU
research environment. No model download, simulator rollout, scientific training
run or LLM API call occurs in the unit suite. A single optimizer transaction is
used only as a checkpoint/gradient unit fixture.

[Validation receipt](docs/validation.json) and [test output](docs/unit_tests.txt)
record the checked sources. Scientific runs require an exact user-approved source
manifest through [the source gate](experiments/source_pins/README.md). Historical
pins are preserved, not refreshed to match later development. Model weights,
datasets, credentials and machine-specific bindings are excluded from Git.

The [archive check](docs/github_archive_check.json) also records 132 passing tests
from a fresh GitHub clone, using the existing external CPU dependencies.

Install simulator environments **outside this checkout** for source-gated runs;
the local-source gate rejects unpinned package code inside the repository. Use
ignored `artifacts/local_paths.json` for machine bindings.

## Project record

- [Working context](docs/CONTEXT.md)
- [Evidence and limitations](docs/EVIDENCE.md)
- [Research roadmap](docs/ROADMAP.md)
- [Provenance](provenance/README.md)
- [Research coordination contract](AGENTS.md)

This repository preserves code, proposals, results and negative findings for the
ongoing research program. It is separate from legacy Gravemind, RBL-RMV and TNST
projects; their artifacts are not silently relabelled as WSPL results.

GitHub: https://github.com/k-oconnor/WSPL (private research archive).
No redistribution license has been selected. Nothing in the proposed experiment
program authorizes new API spending, training, deployment or physical hardware use.
