# Common fixed-tanh authoring interface

Both arms receive this entire guide. Only the final selected-arm line differs.
Write a short explanation of your strategy and intended hidden-unit roles, then
exactly one fenced candidate block for the selected interface. Do not use tools.

## Target and scoring

Every weight starts at zero. You can set every entry of all six FP32 tensors.
The fixed network has 20,486 parameters and no input feature adapter:

```
h1 = tanh(x @ W1 + b1)    W1: [600,32], b1: [32]
h2 = tanh(h1 @ W2 + b2)   W2: [32,32],  b2: [32]
logits = h2 @ W3 + b3     W3: [32,6],   b3: [6]
```

Rows index source units, columns index destination units. The six output logits
are sampled through softmax, with the game's sticky-action probability 0.1.
Score is mean reward on 1,000 episodes, each capped at 10,000 steps. No score,
execution or checker feedback is returned. There is no gradient training.
The architecture, activations and observation/action mapping cannot change.

All deployed values must be finite FP32 with magnitude at most 1,000,000.
One whole reply is limited to 1,000,000 UTF-8 bytes. There is no best-of selection:
an invalid proposal fails in full for BOTH arms. Nothing is silently skipped or
repaired. This strict rejection rule replaces the historical native skip rule.
Explanations are archived as hypotheses; they do not establish circuit semantics.
Tanh is outside the available formal verifier. Game scores are empirical.

## Native arm

Return exactly one `edits` block. One assignment per line; `#` comments and
multiline lists are allowed. Targets are W1, b1, W2, b2, W3, b3 with explicit
indexing for every axis. RHS values are plain decimal literals or nested lists,
with optional signs; no exponent notation, expressions, function calls or loops.
Supported updates: `=`, `+=`, `-=`, `*=`. NumPy broadcasting applies. Slices
are half-open, can be negative or strided, and follow ordinary NumPy clipping.
Native literal and intermediate stored values must have magnitude <=1,000,000.
Statements apply in order using the historical NumPy FP32 update semantics.
Maximum 4,096 statements. Example syntax only, not a suggested game strategy:

```edits
W1[12, 0] = 2.5
W1[0:6, 3] = [1, 0, -1, 0, 2, 0]
b1[0:4] = -1
W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]
W3[:, 2] *= 2
```

## WSPL arm

Return exactly one `wspl` block containing a full WSPL 0.2 tensor program.
This directly programs the same tensors; there is no scalar-source compiler,
relay allocation or hidden-unit reservation. The basic form is:

```wspl
wspl 0.2;
target base=dense(widths=[600,32,32,6],activations=[tanh,tanh,linear],layout=in_out,storage=f32,init=zeros);
edit base as candidate mode=fixed numeric=native_numpy_v1 {
  W1[12,0]=2.5;
  W1[0:6,3]=[1,0,-1,0,2,0];
  b1[0:4]=-1;
  W2[0:2,0:3]=[[1,-1,0.5],[0,2,0]];
  W3[:,2]*=2;
}
```

`numeric=native_numpy_v1` accepts the same literal tensor assignments as the
native arm, with semicolon terminators. It preserves the native FP32 arithmetic
on valid programs. This profile does not support constants, loops or expressions.

Alternatively use `numeric=exact_to_storage` (the default when omitted) for
constants, arithmetic and bounded construction loops. Decimal, scientific and
rational literals are interpreted exactly. Each assignment rounds directly to
FP32 nearest-even. Arithmetic reads the exact rational value of current FP32
storage. This profile may differ numerically from native intermediate rounding;
the selected profile and changed-rounding counts are recorded.

```wspl
wspl 0.2;
target base=dense(widths=[600,32,32,6],activations=[tanh,tanh,linear],layout=in_out,storage=f32,init=zeros);
const gain=1/4;
edit base as candidate mode=fixed numeric=exact_to_storage {
  for j in range(4) { W1[j,j]=gain; }
  W2[0:4,0:4]=copy(W1[0:4,0:4]);
  b3[0]=1/3;
}
```

Exact construction expressions support scalar/array `+ - * / @`, unary signs,
parentheses, literal lists, indexing/slices, `copy(a)`, `transpose(a)`,
`reshape(a,[...])`, `zeros([...])`, `full([...],v)`, `eye(n)`, `f32bits(bits)`.
Values broadcast on assignment; `@` is matrix multiplication. `parent.W1[...]`
reads immutable initial zero storage. `W1[...]` reads current storage. RHS copies
use statement snapshots. Nested `for name in range(stop)` / `range(start,stop)` /
`range(start,stop,step)` loops are bounded construction, not runtime control flow.
Constants are declared before the edit block. Do not shadow tensors, builtins,
`parent`, `x`, `y`, or names `h` followed by digits. There is no external tensor
binding in this screen and no proposal Python execution.

Limits: 4,096 expanded statements including loop entries, 4,096 total loop
iterations, 100,000,000 charged construction operations, 4,096-bit rational
numerators/denominators, 20,486 storage elements, 200,000 lexer tokens,
50,000 AST nodes per expression and nesting <=64. Full-program source <=1 MB.
All final stored weights have the common magnitude bound above. Whole-program
failure is atomic. Full tensor assignment does not consume hidden activations.

Only fixed tensor construction is in this screen. Do not submit `circuit`,
`region`, `place`, `describe`, graph edits or topology changes. The current
circuit-placement subset supports affine/ReLU, not the fixed tanh hidden units;
direct tensor construction remains fully available. The harness saves the source,
replayed plan, deployed weights, storage-derived equations and empirical checks.
Neither writer gets any of that evidence as feedback.
