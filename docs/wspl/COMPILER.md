# First WSPL neural compiler and weight artifact

Implemented November 3, 2026. The scalar JSON IR now compiles into an ordinary
dense ReLU MLP with exact rational coefficients, explicit named ports and a
deterministic source map. `ReLUArtifact` independently loads/evaluates the stored
matrices without importing the compiler or WSPL source interpreter. Optional
Torch emission produces an ordinary `nn.Sequential` of Linear/ReLU layers.

This is an initial G1 implementation. Text syntax, fixed-architecture editing,
RMV allocation, universal equivalence/contract verification, error enclosures,
learned-model imports and the scientific proposer harness remain incomplete.
Compilation and finite cross-checks do not establish a solver-backed certificate.

## What the compiler does

Type-check the scalar source, combine affine expressions using rational linear
forms, then lower nonlinear expressions using real-arithmetic identities:

```
clip(z,l,u) = l + ReLU(z-l) - ReLU(z-u)
min(a,b)   = a - ReLU(a-b)
max(a,b)   = b + ReLU(a-b)
and(p,q)   = ReLU(p+q-1)
or(p,q)    = p+q-ReLU(p+q-1), for Truth inputs
not(p)     = 1-p
```

Normalize margins before clipping; constant multiplication remains linear. Common
ReLU preactivations are shared and unused nonlinear units are removed. Then
allocate layers: signed inputs crossing a hidden layer use positive/negative
ReLU channels; existing nonnegative ReLU outputs use identity relay neurons.
This preserves skip dependencies without assuming a signed value survives ReLU.
The final layer is affine, not silently clipped by the runtime.

The compiler has explicit layer/unit/parameter limits and only `fresh_mlp` mode.
Allocation-limit errors are not infeasibility proofs. Unsupported fixed/augmented
target modes are rejected instead of silently changing architecture. This compiler
chooses its own fresh MLP; it does not target the pilot's 1,604-parameter model.

## Concrete examples

| Source | Allocated widths | Dense parameters, including bias |
| --- | --- | ---: |
| BrakePatch | 3 -> 4 -> 2 -> 1 | 29 |
| ApproachP | 12 -> 6 -> 6 -> 4 | 148 |

BrakePatch reproduces the proposal's explicit construction: first-layer urgency
units plus signed baseline relays, two final clipping units, then linear output.
ApproachP compiles the known standoff P controller from source; these are inspected
examples, not independent LLM synthesis, learned models or a new physical run.
The graph is not aggressively minimized; capacity counts include allocated zeros
and relay parameters.

The generated JSON examples are tracked for inspection:

- examples/wspl/brake_patch.relu.json
- examples/wspl/approach_p.relu.json

They contain matrices and metadata, not executable Python, pickled models or
training checkpoints. Source maps associate expression paths with exact linear
forms and allocated ReLU/relay locations. They are emitted annotations with unit
cross-checks, not independently verified universal explanations yet.

## Compile and inspect

After the README editable installation, choose a new output path:

```sh
python -m skillforge.wspl_compile examples/wspl/brake_patch.core.json --output work/brake_weights.json
```

The parent directory must exist; existing outputs are not overwritten. The command
reports the allocation and explicitly says universal verification was not performed.
It constructs no simulator, invokes no optimizer and makes no LLM/API calls.

```python
from skillforge.wspl_artifact import ReLUArtifact

artifact = ReLUArtifact.load("examples/wspl/brake_patch.relu.json")
assert artifact.evaluate({"d": "1/5", "v": "2/5", "a_base": "1"})["a"] == -1

# Optional, if PyTorch is installed:
model, numerical_profile = artifact.to_torch()
assert numerical_profile["stored_coefficients_exact"]
assert numerical_profile["execution_guarantee"] == "not_established"
```

The artifact's SHA-256 is over canonical JSON content, including the source map,
named I/O and coefficients. It is distinct from a pretty-printed file byte hash.
The source hash likewise binds canonical source IR, not arbitrary text whitespace.
Load with an independently retained expected artifact hash to detect mutation.
Hash agreement establishes identity, not semantics. No proof/certificate is stored.

## Exact weights versus numerical execution

The serialized mathematical artifact accepts rational coefficients, including
ones not representable exactly in a machine dtype. Torch emission currently
allows only FP32 or FP64 and checks every stored weight/bias for exact
representability. A coefficient such as 1/3 is rejected rather than rounded.
There is no approximate compilation mode yet.

Even exact coefficient representation does not imply exact floating-point
arithmetic, source equivalence or a deployed contract. Torch's profile records
that execution guarantees are not established. The caller is responsible for
binding/ordering named inputs, units, domain checks and input quantization. The
bare emitted tensor module does not enforce physical units or a safety domain.
Input quantization, operator rounding and full preprocessing/postprocessing need
a separate numerical profile/error enclosure before transferring universal claims.

## Validation and next gate

Ten new tests check the explicit BrakePatch weight matrices, exact source versus
independent weight evaluation on finite boundary/interior fixtures, all supported
operator compositions, deep signed/nonnegative relay paths, dead-unit removal,
shape/hash mutations, output overwrite refusal, target/capacity errors, CLI
serialization, optional Torch coefficient checks and FP32/FP64 finite probe
agreement. Global Torch RNG is preserved when emitting the deterministic network.
These are engineering checks, not a scientific benchmark or universal proof.

Next: G2 independently encode the serialized graph and source/contract semantics,
test non-vacuity, validate counterexamples, and scope PROVED/REFUTED/UNKNOWN/INVALID
results to mathematical real graphs. Only then proceed to learned-controller
imports and the blinded cross-domain harness.
