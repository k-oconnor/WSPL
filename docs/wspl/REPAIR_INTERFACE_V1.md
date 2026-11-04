# Shared neural repair interface packet

The task supplies an existing neural DAG, its exact parameter values, input box,
two externally evaluated requirements and a common resource budget. Change its
drive behavior and retain its keep behavior. This is a synthetic circuit repair
screen. No simulator, training, tools or evaluation feedback is available. All
information below is given to both interfaces. Only SELECTED ARM differs.

Return a short explanation and exactly one fenced `json` object. Maximum complete
reply size is 200,000 UTF-8 bytes. Duplicate JSON keys, multiple proposal blocks,
nonfinite values and unknown fields invalidate the proposal. All valid repairs
within the common budget are allowed, including adding nodes or replacing the
entire internal computation. Input names/widths and output names/widths stay fixed.

## Common raw access

Both arms may submit `{"raw": {...}}`. Raw fields are optional:

| Field | Meaning |
| --- | --- |
| `put_tensors` | Map tensor name to a complete new/replacement tensor record |
| `drop_tensors` | List existing tensor names to delete |
| `put_nodes` | List complete new/replacement node records, identified by `id` |
| `drop_nodes` | List existing node IDs to delete |
| `outputs` | Map public output names to replacement signals; omitted outputs retain their mapping |
| `tensor_source` | Ordered direct tensor assignments, after structural updates |

A tensor record is `{"shape":[rows,columns],"values":[...flat row-major values...]}`.
Use JSON integers or quoted decimal/rational strings, e.g. `"1/8"`. JSON decimal
literals are read as their written decimals. Values round directly to finite
FP32, nearest/even. Alternatively use `{"shape":[...],"bits":["3f800000",...]}`
with exact eight-digit lowercase FP32 hex words. No implicit tied array views.
Reusing one tensor name in multiple nodes explicitly shares its storage.

A node record has exactly `id`, `op`, `inputs`, `params`, `attrs`:

```json
{"id":"u","op":"affine","inputs":{"x":"input:x"},"params":{"weight":"A","bias":"c"},"attrs":{"layout":"in_out"}}
```

This is schema illustration, not a task solution; actual widths must match.
Signals are `input:port_name` or `node:node_id`. Edges are `node_id.input_port`
or `output.output_name`. Parameter references are `node_id.parameter_port`.

| Operator | Inputs | Parameters | Attributes and meaning |
| --- | --- | --- | --- |
| `affine` | `x` | `weight`, `bias` | `layout`: `in_out` (input-by-output matrix) or `out_in`; vector bias |
| `relu` | `x` | none | none; coordinatewise max(0,x) |
| `identity` | `x` | none | none |
| `add` | `a`,`b` | none | none; equal widths, ordered a+b |
| `scale` | `x` | `gain` | none; gain tensor has shape [1] |
| `concat` | `a`,`b` | none | none; vector a followed by b |
| `slice` | `x` | none | integer `start`,`stop`,`step`; Python slice semantics, nonempty output |

Empty `params` and `attrs` objects are required when unused. All graph nodes are
stateless. Cycles, dangling connections, implicit broadcasts and budget violations
invalidate construction. Every named stored tensor counts toward the budget,
including unreferenced tensors. Every node counts, including disconnected nodes.

`tensor_source` uses direct array syntax with semicolon-terminated statements:
`T[0,0]=1/8; T[:,1]+=2; T[:,:]=parent.T[:,:];`. Reads use current statement-start
storage; `parent` is immutable storage at the start of this tensor phase. Slices,
ordinary numeric/array arithmetic, scalar broadcasting, `f32bits(0x...)`,
literal rectangular lists, `zeros(shape)`, `full(shape,value)`, `eye(size)`,
`copy(...)`, `reshape(value,shape)`, `transpose(...)` and bounded
`for i in range(...) { ... }` construction are
available. No arbitrary Python execution, imports, file access or named circuits
are available in this field. New storage must be declared with `put_tensors`.
Raw construction is atomic; the final graph must be valid before tensor edits.

## Structured WSPL extension

The `wspl` arm can use all common raw access, and may additionally submit
`"modules": {...}` and `"wspl": "source text"`. The direct arm permits only `raw`.
There is no requirement for the WSPL arm to use its additional features.

A module is a complete `{ "graph": ..., "tensors": ... }` object. Tensor records
use the same shape/values-or-bits format as raw access. Its graph has exactly:
`{"kind":"dag","inputs":{"port":width},"nodes":[...],"outputs":{"port":"signal"}}`.
Up to eight module declarations are accepted. Modules are authored in the reply;
no ready-made task-solution module is supplied. Their contents are validated and
hashed before source expansion. There is no automatic expression allocator.

WSPL runs after the optional raw phase. The source begins with:

```wspl
wspl 0.2;
target base=artifact("base",sha256="{{base_sha256}}");
```

The transport substitutes `{{base_sha256}}` with the current raw-phase artifact
hash and `{{module_sha256:alias}}` with that submitted module's hash. It saves
expanded source and the exact modules in the construction receipt. These are
literal hash placeholders, not code execution or inferred edits.

The edit has the form `edit base as candidate mode=graph budget="repair" { ... }`.
Use fixed mode for tensor-only changes. Raw tensor statements are available inside
the edit as well. Graph operations are:

```wspl
transplant artifact("alias",sha256="{{module_sha256:alias}}") as prefix {
  inputs={module_input:signal("node:existing_node")};
}
redirect edge("output.output_name")=signal("node:prefix__module_node");
untie parameter("node_name.weight") as fresh_tensor;
```

Transplant copies nodes/tensors under `prefix__` without connecting existing
outputs. The optional `share={module_tensor:parent_tensor}` field reuses storage
only when shape, dtype and all bits are identical. Untie clones only the named
node's parameter reference; other consumers retain their original storage.

A region declaration precedes the edit:

```wspl
region region_name in base {
  nodes=[node("selected_node")];
  inputs={input_alias:edge("selected_node.x")};
  outputs={output_alias:edge("consumer.x")};
  shared_parameters=["shared_tensor"];
}
```

List every edge entering or leaving the selected nodes, including public output
edges and skip connections. List every tensor referenced both inside and outside
the region; use `[]` if there are none. Inside the edit:

```wspl
replace region_name with artifact("alias",sha256="{{module_sha256:alias}}") as prefix {
  inputs={module_input:region_name.inputs.input_alias};
  outputs={region_name.outputs.output_alias:module_output};
}
```

All outgoing aliases must be mapped. Incoming aliases must be used or listed
explicitly in `drop_inputs=[unused_alias]`. Replacement removes the old nodes and
only their now-unreferenced tensors. Module outputs can feed multiple consumers.
Regions describe the original raw-phase parent; stale cuts fail.

For residual augmentation:

```wspl
add_residual artifact("alias",sha256="{{module_sha256:alias}}") as prefix {
  inputs={module_input:signal("node:existing_node")};
  output=module_output;
  into=edge("output.output_name");
  scale=1/8;
}
```

This explicitly computes original + (module_output * stored FP32 scale). Added
nodes and tensors count toward the same budget. No parent mutation is published
when any operation fails. Candidate descriptions or replacement requirements are
not accepted; the evaluator owns the task's claims.

## Common evaluation

Both repair and preservation equalities are checked over the entire supplied
input box with real arithmetic and exact interpretations of stored FP32 values.
Both must return PROVED; REFUTED, UNKNOWN, UNSUPPORTED or invalid construction
cannot count as success. A common FP32 execution panel must also have maximum
absolute error <= 0.000005 against the requested functions, with finite outputs.
That panel does not establish universal IEEE or physical behavior. Final claims,
counterexamples, resource usage, complete candidate artifacts and failures are
recorded. There is no evaluator feedback or retry during this one-shot screen.
