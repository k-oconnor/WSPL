# WSPL ? Weight Space Programming Language

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
| WSPL-to-neural compiler and serialized weights | In development |
| Universal contract verifier / numerical execution proof | Planned |
| Blinded LLM authoring and counterexample-feedback harness | Planned |
| Small-alphabet LTLf compiler and strong finite-model planner | Implemented for supplied symbolic effect models |
| Hierarchical task expansion | Implemented at a supplied macro-summary cut; no physical primitive executor |
| Neural MLP/RMV blocks and single-tick LIBERO bridge | Implemented; independent sampled outcome observers |
| Pregrasp positioning pilot | Completed; scoped results below |
| Diverse robot tasks | 16 inspected specifications over 7 audited scene assets; contact/obstacle integration pending |

The public-facing WSPL text syntax in the proposal is not implemented. The scalar
prototype is a JSON IR; it does not yet compile neural weights, prove universal
contracts or synthesize controllers. No WSPL CLI is advertised as available.

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

Constants and inputs are exact rational strings or integers; floating-point input
is not silently treated as an exact specification. The scalar IR supports affine
arithmetic, ReLU, clipping, normalization, min/max and graded Lukasiewicz logic.
Unit labels, bounds and acyclic references are checked. Conservative interval
analysis can reject a valid program when a stronger proof is needed.

See [the scalar core](docs/wspl/SCALAR_CORE.md),
[BrakePatch](examples/wspl/brake_patch.core.json) and
[ApproachP](examples/wspl/approach_p.core.json).

## Research program

Three studies?fresh authoring, repair of independently trained artifacts, and
frozen composition?span discrete causal worlds, low-dimensional dynamics and
robot manipulation. Each compares direct weights, ordinary PWA code and WSPL,
with and without verifier feedback. All arms receive final verification.

The [experimental program](docs/wspl/EXPERIMENT_PROGRAM.md) separates mathematical
contracts, deployment numerics and physical outcomes. The proposed first screen
has 36 synthesis runs, but is not registered, funded, approved or launchable yet.
Model identity, monetary cap, frozen tasks and final thresholds remain unset.

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

The current local suite has 122 passing checks. Optional neural/adapter checks
skip when PyTorch or NumPy is absent; full local results used a separate CPU
research environment. No model download, simulator rollout, scientific training
run or LLM API call occurs in the unit suite. A single optimizer transaction is
used only as a checkpoint/gradient unit fixture.

[Validation receipt](docs/validation.json) and [test output](docs/unit_tests.txt)
record the checked sources. Scientific runs require an exact user-approved source
manifest through [the source gate](experiments/source_pins/README.md). Historical
pins are preserved, not refreshed to match later development. Model weights,
datasets, credentials and machine-specific bindings are excluded from Git.

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
