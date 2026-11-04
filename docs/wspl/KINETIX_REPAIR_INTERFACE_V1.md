# Common Kinetix repair interface

Both arms receive this entire guide. Only SELECTED ARM differs. Diagnose the
parent's failure before your candidate, retain useful computation where possible,
but prioritize success over minimizing changed weights. No tools, execution,
training, score feedback or retries. One candidate, not multiple alternatives.

The supplied rounded actor is cast once to FP32 (nearest representable), then
both arms edit and deploy FP32 tensors using identical NumPy inference. Physics
and observations use FP32; categorical sampling uses stable FP64 probabilities.
The original parent trace is supplied historical information, not a guarantee
of exact FP32 deployment behavior. The architecture and all 171,664 parameters
are available to both arms. No hidden-unit reservation or feature adapter.
The score is success fraction on 1,024 sampled-action episodes, horizon 256.

## Names

Native aliases correspond to WSPL names as follows:

| Native | WSPL | Shape of W / b |
| --- | --- | --- |
| embed.W / embed.b | W1 / b1 | 679×128 / 128 |
| actor0.W / actor0.b | W2 / b2 | 128×128 / 128 |
| actor1.W / actor1.b | W3 / b3 | 128×128 / 128 |
| actor2.W / actor2.b | W4 / b4 | 128×128 / 128 |
| actor3.W / actor3.b | W5 / b5 | 128×128 / 128 |
| actor4.W / actor4.b | W6 / b6 | 128×128 / 128 |
| actor_head.W / actor_head.b | W7 / b7 | 128×16 / 16 |

The first hidden activation is ReLU, the next five are tanh; output is linear.
Rows are input units, columns output units. No graph/topology changes here.

## Native

Exactly one `edits` fence. Literal assignments, one per line; comments beginning
`#` and multiline lists allowed. Explicit indices for each axis; half-open
NumPy slices, negative indices and steps allowed. Updates `=`, `+=`, `-=`, `*=`;
scalar/list broadcasting. Plain decimal literals only, no expressions/loops.
`ZERO actor0.W` is shorthand for setting that whole tensor to zero.
Every invalid statement rejects the entire candidate; no silent skips.

Example syntax only:

```edits
embed.W[9, 2] = 0.25
actor_head.b[4:6] += [0.5, -0.5]
```

## WSPL

Exactly one `wspl` fence with a complete program. The parent artifact SHA below
identifies the same supplied FP32 actor; import it, do not create a zero network.

```wspl
wspl 0.2;
target base=artifact("parent",sha256="{PARENT_SHA256}");
edit base as candidate mode=fixed numeric=native_numpy_v1 {
  W1[9,2]=0.25;
  b7[4:6]+=[0.5,-0.5];
}
```

`native_numpy_v1` uses literal assignments only and matches native FP32 updates.
Alternatively `numeric=exact_to_storage` supports exact rational expressions,
constants and bounded loops; each assignment rounds to FP32. Arithmetic reads
exact rational values of current FP32 tensors. This can differ from native
intermediate rounding and is reported separately. Construction loops are not
runtime recurrence.

```wspl
wspl 0.2;
target base=artifact("parent",sha256="{PARENT_SHA256}");
const gain=1/4;
edit base as candidate mode=fixed numeric=exact_to_storage {
  for j in range(3) { W2[j,j]=gain; }
  W3[0:3,0:3]=copy(parent.W3[0:3,0:3]);
}
```

Exact expressions: decimal/scientific/rational numbers, lists, unary signs,
`+ - * / @`, indexing/slices, `copy`, `transpose`, `reshape(a,[...])`,
`zeros([...])`, `full([...],v)`, `eye(n)`, `f32bits(bits)`. `parent.W1` reads
immutable initial storage; `W1` reads current storage. RHS reads use statement
snapshots. Declare constants before the edit block. Bounded `range(stop)`,
`range(start,stop)` and `range(start,stop,step)` loops are supported.
No external tensors, Python execution, circuits, regions, descriptions or graph edits.

## Common limits

Whole reply ≤1,000,000 UTF-8 bytes. 4,096 expanded statements/loop entries;
4,096 loop iterations; 100,000,000 charged construction operations;
4,096-bit rationals; 171,664 tensor elements. Stored weights finite and
absolute value ≤1,000,000. Invalid candidates fail in full. A zero-edit-equivalent
valid assignment is allowed and evaluated. Explanations are hypotheses; tanh
formal proofs are unsupported. No checker feedback is supplied.
