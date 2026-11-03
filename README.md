# Skillforge

**LLM-programmed neural and interpretable RMV skills, composed through finite-trace temporal programs.** Working name, confirmed by Kevin O'Connor; revisit branding later.

An LLM proposes a task specification in LTLf and implementations of small closed-loop skills. A checked compiler and planner select skills under explicit contracts. An independent observer verifies what happened. Validated skills become a reusable library; changing one mechanism should permit local repair.

```mermaid
flowchart LR
    N[Task description] --> L[LLM task proposal]
    L --> T[Typed LTLf specification]
    T --> A[Automaton and contract-based planner]
    K[Validated skill library] --> A
    A --> S[Neural or RMV action block]
    S --> E[Robot simulator]
    E --> O[Independent outcome observer]
    O --> A
    W[LLM weight proposal or gradient learning] --> V[Skill validation]
    V --> K
```

The diagram describes the proposed system. This repository currently contains the **specification, schemas, examples and reference semantics**, not a robot system or a completed LTLf compiler.

## Start here

- [System specification](docs/SYSTEM_SPEC.md): interfaces, temporal semantics, execution, RMV fidelity and guarantees.
- [First experiment](docs/EXPERIMENT_01.md): LLM/gradient acquisition × MLP/RMV, costs, held-out composition and repair.
- [Roadmap](docs/ROADMAP.md): implementation stages and evidence required to advance.
- [Prior evidence](docs/EVIDENCE.md): completed toy-world experiments and their limits.
- [Related work](docs/RELATED_WORK.md): what is established and where this proposal could contribute.
- [Task example](examples/pick_place.task.json), [skill contract](examples/lift.skill.json), [RMV graph](examples/lift_gate.rmv.json).

## What exists

| Component | Status |
| --- | --- |
| Typed JSON task/skill/RMV schemas | Provided |
| Direct LTLf semantics on completed finite traces | Implemented reference; unit-tested |
| Scalar grounded RMV evaluation including affine motor readout | Implemented reference; unit-tested |
| Source-identity gate and empty approval registry | Provided; gate tests copied with provenance |
| LLM task translation / weight-authoring harness | Planned |
| DFA compiler, realizability check and planner | Planned |
| Differentiable MLP/RMV skills and robot adapter | Planned |
| Robotic acquisition, composition or repair results | Not run |

## Reference checks

Python 3.11 or later. Reference evaluators use only the standard library. Schema tests additionally use `jsonschema`:

```sh
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

No simulator, model download, training, LLM API call or hardware control occurs in those tests. LTLf evaluation applies to a **closed, nonempty trace**; a false unfinished prefix is not a failure verdict. Logical propositions describe observed state, commands or verified skill outcomes; action blocks themselves are executable policies.

## Research questions

1. Can LLM-authored weights acquire reliable skills using fewer interactions or less total time than gradient learning?
2. Can grounded RMV implementations retain useful performance while providing exact computational explanations of motor outputs?
3. Do independently validated skills transfer to unseen temporal compositions without internal retraining?
4. Can changing one mechanism be handled by local repair while preserving still-valid behavior?

LTLf planning, LLM-generated robot programs and skill libraries already have substantial prior work. Novelty must be assessed around weight-authored, faithfully compiled RMV skills, their acquisition cost and compositional repair. The toy-world evidence supports supplied factorization; it is not robotic validation or evidence that LLMs outperform gradient-trained skills.

## Repository boundaries

This is a new local repository, separate from Gravemind and legacy RBL/TNST robot models. [Provenance](provenance/README.md) links the antecedents without copying datasets or checkpoints. No GitHub remote, publishing configuration or redistribution license has been selected. Future scientific runs require exact source approval under [AGENTS.md](AGENTS.md); the current specification authorizes no run automatically.
