# WSPL 0.2 graph surgery

Implemented November 3, 2026, extending the [tensor/circuit editor](WSP_EDITOR_FIRST_PASS.md).
The editor can now replace complete graph regions, transplant pinned modules,
redirect connections, add residual branches, and separate shared parameters.
It operates on explicit stateless vector DAGs and produces immutable candidates.
Historical Core PWA and scientific runners retain their original semantics.

## Executable scope

| Capability | Implementation |
| --- | --- |
| Graph | Named vector input/output ports, stable node IDs, explicit ordered connections |
| Operators | `affine`, `relu`, `tanh`, `identity`, `add`, `scale`, `concat`, `slice` |
| Storage | Exact finite FP32 bits; canonical tensor names explicitly represent sharing |
| Construction | Direct tensor edits and graph operations in one replayable transaction |
| Inspection | Complete boundary cuts, parameter consumers, shapes, topology and resource counts |
| Diff | Nodes, edges, tensor metadata/bits and downstream dependency closure |
| Execution | NumPy FP32; independent rational interpretation for the supported real-graph subset |
| Formal checking | Actual candidate DAG, non-vacuity and independent rational witness replay; `tanh` is unsupported |

There is no recurrent state, convolution/attention operator, mixed dtype, physical
tensor-view aliasing, arbitrary framework importer or automatic placement in this
extension. General subgraph topology is supported within the listed operator set.
Node splitting is expressible by replacing a node with an explicit multi-node
module, for example two affine branches followed by concatenation. Such edits
do not automatically preserve behavior or floating-point reduction order.

## Graph identity and boundaries

`input:x` names an input signal; `node:left` names a node output. `left.x` names
the incoming edge at node `left`'s `x` port. `output.y` names a public output
edge. `left.weight` names a parameter reference, which is distinct from a signal
edge. An affine node declares `in_out` or `out_in` storage orientation.

The graph validator resolves all connections, rejects cycles/dangling references,
infers widths, checks parameter shapes, and forbids implicit broadcasting. A
deterministic topological order does not depend on node declaration order.
Dense artifacts enter graph mode only through an explicit conversion:

```sh
python -m skillforge.wsp_editor graphify dense.json --output dag.json
python -m skillforge.wsp_editor inspect dag.json
```

`graphify` preserves every tensor bit and the existing layer/operator order. It
changes the artifact's graph identity and requires rebinding descriptions. It
does not import an arbitrary PyTorch model or prove IEEE equivalence universally.

A region cut enumerates **every crossing edge**, including fan-out, skip edges,
and public outputs, plus all tensors shared between region and outside nodes.
The declared aliases must cover that exact cut once each. An edit cannot hide a
consumer by omitting it from its declaration. Query the actual cut first:

```sh
python -m skillforge.wsp_editor boundary examples/wsp_graph/parent.json --nodes left
```

## Replacement syntax

This is the [runnable replacement](../../examples/wsp_graph/replace.wspl):

```wspl
wspl 0.2;
target base=artifact("base",sha256="692b928fb6ed8e194bfe5db305e8da226412e8f260da8bff00ca92cf40402776");
region steering in base {
  nodes=[node("left")];
  inputs={obs:edge("left.x")};
  outputs={contribution:edge("sum.a")};
  shared_parameters=["W"];
}
edit base as candidate mode=graph budget="repair" {
  replace steering with artifact("patch",sha256="396f378554c949740c761dc56e41e4b8f59aca14809b68f48aa47bb6382ec736") as fixed {
    inputs={x:steering.inputs.obs};
    outputs={steering.outputs.contribution:y};
  }
}
```

The module's input `x` receives the old cut input. Its output `y` replaces the
specific consumer edge. Every outgoing cut edge needs a mapping. Every incoming
cut alias must be used or explicitly listed in `drop_inputs=[alias]`. Multiple
consumers can map to the same new module output. The replacement's declared
input widths are checked even if a module does not use an input internally.

The module is supplied separately by the host, and its SHA-256 must match the
source. Imported nodes and tensors receive the `fixed__` prefix. Default imports
copy storage while preserving sharing *within* the module. An optional
`share={W1:W}` reuses parent storage only when its shape, dtype and every bit match
the module tensor. Tensor equality alone never triggers implicit sharing.

Replacing a region removes its nodes and only those region parameters that no
remaining node uses. Shared or unrelated storage remains intact. A region is
declared against the original parent; changing its cut before replacement makes
the declaration stale. Coordinated changes that cannot be valid individually
must be packaged into one region replacement.

## Other edits

These fragments belong inside a graph-mode edit; module hashes must be full
actual hashes, as in the runnable examples:

```wspl
transplant artifact("patch",sha256="396f378554c949740c761dc56e41e4b8f59aca14809b68f48aa47bb6382ec736") as branch {
  inputs={x:signal("input:x")};
}
redirect edge("output.keep")=signal("node:branch__y");
untie parameter("left.weight") as privateW;
privateW[:,:]=5;
```

`transplant` imports without rewiring existing consumers; `redirect` changes one
edge. `untie` copies only the named parameter reference to fresh canonical storage.
Without it, a raw edit to `W` updates the parameter seen by **all** its consumers;
those consumers are included in the write report and dependency diff.

An explicit residual edit imports a module and inserts scale then add:

```wspl
add_residual artifact("patch",sha256="396f378554c949740c761dc56e41e4b8f59aca14809b68f48aa47bb6382ec736") as boost {
  inputs={x:signal("input:x")};
  output=y;
  into=edge("output.y");
  scale=1/2;
}
```

The ordered computation is `original + (branch * stored_gain)`. The requested
rational gain is rounded directly to FP32; its bits, stored value, new nodes and
resource costs appear in the receipt. This is architecture augmentation and is
counted as such. Auxiliary names are deterministic and collision-free.

## Host budget and transaction

`mode=graph budget="repair"` requires a separately supplied `GraphBudget`. Its
JSON limits nodes, edges, stored parameter elements, depth, cumulative activation
values and allowed operators. The `max_live_values` field counts all input and
node-output values retained by the reference trace, not measured peak VRAM.
The original public input mapping and output names/widths must remain unchanged.
Source cannot introduce or enlarge the budget. Hard parser/artifact limits also
apply. These bounded engineering limits do not establish denial-of-service safety
for arbitrary hostile deployments.

```sh
python -m pip install -e ".[editor,verify]"
python -m skillforge.wsp_editor plan examples/wsp_graph/replace.wspl --parent examples/wsp_graph/parent.json --module patch=examples/wsp_graph/patch.json --budget repair=examples/wsp_graph/budget.json --output work/replace.plan.json
python -m skillforge.wsp_editor apply work/replace.plan.json --parent examples/wsp_graph/parent.json --output work/replaced.json
python -m skillforge.wsp_editor diff examples/wsp_graph/parent.json work/replaced.json
python -m skillforge.wsp_editor describe work/replaced.json --node fixed__a1
```

Create `work` first and choose unused output filenames. Construction occurs on
private storage. Any invalid operation aborts the plan without changing its
parent or publishing a candidate. `apply` reconstructs pinned modules and budget
from the receipt, then replays source and checks the entire resulting plan.
Parent, implementation or receipt changes invalidate it. Hashes establish
identity, not caller authentication or deployment authorization. Applying stages
an artifact; no active-model switch or admission service is implemented.

## Evidence and validation

`describe` can bind `view signal=tap("node_id")[0]` and claims to actual candidate
inputs/outputs. Z3 encoding and rational witness evaluation traverse the entire
stored graph. A successful real-graph claim is about real arithmetic with stored
FP32 coefficients; it does **not** prove IEEE execution or physical behavior.
Tanh returns `UNSUPPORTED`; resource-limited formal checks may return `UNKNOWN`.
Probe reports remain empirical. All descriptions bind a particular artifact
identity and become stale after an edit.

The diff's `unaffected_dependency_nodes` means topology and transitive storage
dependencies did not change. It is not a numerical equivalence certificate or
permission to reuse an old claim automatically. Edited paths require fresh checks.

Run the supplied engineering example:

```sh
python tools/demo_wsp_graph.py --output work/wsp_graph_surgery_demo
```

It starts from `y=4*x+1, keep=2*x`, performs three independent edits, and checks
`y=5*x`, `y=(11/2)*x+1`, or `y=7*x+1` while preserving `keep=2*x`. It proves both
claims for each candidate on `[-4,4]`, then independently replays witnesses for
deliberately false claims. Plans, candidates and receipts are retained.

All 50 graph checks pass. The full suite runs 389 checks: 388 pass and one native
fixture is skipped in the primary runtime; all 12 native fixture checks pass in
the separate MinAtar runtime. Source identity remained unchanged through testing.
Validation is recorded in [the receipt](wsp_graph_surgery_validation.json) and
[unit-test log](wsp_graph_surgery_unit_tests.txt). This is supplied-answer software
engineering evidence. No LLM authoring, training, scored game episodes, robot
execution or new scientific comparison was performed.
