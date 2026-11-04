# Public shared interface packet: matched Breakout ReLU screen

This entire packet is given unchanged to every arm. The selected arm is identified
at the end of the task prompt. Use only that arm's submission fence. One final
reply, brief design description then one final candidate block. No tools, running
experiments, follow-up questions or evaluation feedback. The game rules/source
and raw observation layout are supplied separately and identically to every arm.

## Common target and scoring

Construct the weights of a 600 -> 32 -> 32 -> 6 neural policy from scratch.
There are exactly 20,486 stored parameters, including zeros and biases. Hidden
activations are ReLU, NOT tanh. The final layer is linear:

```
h1 = max(0, x @ W1 + b1)    W1:600x32, b1:32
h2 = max(0, h1 @ W2 + b2)   W2:32x32,  b2:32
logits = h2 @ W3 + b3       W3:32x6,   b3:6
```

All arrays and arithmetic at evaluation are FP32; actions are sampled from the
same FP64 softmax sampler. All arms use the same raw 600 binary observation
values and the same six actions. No features, geometry or control logic are
computed by the adapter. Mean game return over 1,000 episodes, capped at 10,000
steps each, is the performance endpoint. No gradients or development scores.
Stored coefficients must be finite with magnitude <=1,000,000. Whole reply size
<=1,000,000 UTF-8 bytes. Every final policy has the same two 32-wide hidden layers.
Source programs that cannot be allocated in that target fail; no expansion or
extra interpreter executes during gameplay.

## Common names and operators

Inputs are x000,x001,...,x599, where xNNN is raw observation index NNN under the
supplied row/column/channel mapping. Their declared domains are Real[0,1], with
dimensionless normalized_value units. Output logits a0,...,a5 are ordered actions
0,...,5. Output declarations use Real[-1000000000000000000000000000000,
1000000000000000000000000000000] normalized_value. These broad bounds are not
winning-game or safety contracts. Propose a memoryless policy; state comes only
from the supplied observation channels.

Source operators: +,-, multiplication/division by constants; relu(x)=max(0,x),
clip(x,l,u)=min(u,max(l,x)), min(x,y), max(x,y). No varying-by-varying products,
nonlinear arithmetic, branches, loops, imports, attributes, functions or indexing
in the source arms. WSPL also has its existing margin and Truth connectives:
margin(x,l,u)=clip((x-l)/(u-l),0,1), not(p)=1-p,
and(p,q)=max(0,p+q-1), or(p,q)=min(1,p+q),
implies(p,q)=min(1,1-p+q). Truth operands must have proved interval [0,1].
These can also be written using the common PWA operators in Python source.

Plain decimals and integer/rational constant expressions in source are exact
mathematically. The compiler explicitly rounds final coefficients through FP64
to FP32, reports any coefficient changes, and runs a final-only source/weight
equivalence diagnostic. Prefer integers and dyadic fractions (halves, quarters,
etc.) when exact coefficient fidelity matters. Native edits use FP32 updates.
This is not a floating-point execution proof; no diagnostic is fed back to you.

## Weights arm: native compact edits

Use an edits fence. All six arrays start at zero. Row i of W1 is input i;
column j of any matrix belongs to output unit j of that layer. Accepted tensors:
W1,b1,W2,b2,W3,b3. One statement per line; long literal lists may span lines.
Only =,+=,-=,*= with literal decimal values/arrays and integer indices or slices.
No variables, expressions, array constructors or executable Python.

```
W1[3,0] = 1
b1[0] = -0.5
W2[0,0] = 1
W3[0,1] = -1
W3[0,3] = 1
```

Slices a:b are half-open, a:b:s uses stride s, : uses the whole axis. NumPy
broadcasting for literal scalar/list writes is supported. For example
W1[3:600:60,0]=1 writes channel 3 / column 0 at every row to hidden unit 0.
Statements apply in order. Last complete edits fence wins; unparseable
statements are logged and skipped, preserving the native baseline convention.
An all-zero complete candidate is evaluated; a missing fence is invalid.

## Code arm: restricted straight-line Python body

Use a python fence containing scalar assignments to fresh lower-case names and
one final return list/tuple of six logits. The xNNN names and operators above are
already in scope. Do not write def, input declarations or executable scripts.
This is parsed as data into the same scalar IR/compiler as WSPL, not executed.
Intermediate intervals/types are inferred. Example of the SAME tiny function:

```
h = relu(x003 - 0.5)
return [0, -h, 0, h, 0, 0]
```

No reassignment, loops/comprehensions, branches, arbitrary calls, attributes or
indexing. AST <=200,000 nodes; lowered scalar IR <=50,000 nodes, expression
nesting <=32. Long affine sums are flattened; earlier scalar names can be reused.
This control is a source-language subset, not unrestricted Python weight scripts.

## WSPL arm: Core PWA text body with a fixed signature

Use a wspl fence containing typed lets and the six output assignments. The
harness inserts an existing Core PWA module header with all 600 named input
ports and six output declarations described above, then calls the existing text
parser. Do not repeat module/input/output declarations. This saves boilerplate;
no computational features or policy code are inserted. Example, same function:

```
let h: Real[0,0.5] normalized_value = relu(x003 - 0.5);
a0=0; a1=-h; a2=0; a3=h; a4=0; a5=0;
```

Each let needs an explicit Real[lo,hi] range (optional normalized_value unit) or
Truth type, and must be declared before use. Names are lower-case letters/digits/
underscores beginning with a letter. Outputs must each be assigned exactly once.
Use // comments and semicolons. Candidate ensures/preserves contracts are rejected;
the study owns all requirements. Unit labels starting normalized_ denote
dimensionless inputs/values; no new physical-unit interfaces are permitted here.
Existing parser limits apply (1 MB expanded module, nesting <=64); lowered scalar
IR <=50,000 nodes, expression nesting <=32. Invalid source fails the proposal.

## Source allocation, shared by code and WSPL

Exact rational affine/ReLU lowering shares common units and removes unused ones.
relu is one gate; clip(z,l,u)=l+relu(z-l)-relu(z-u); min/max similarly use a gate.
Compilation supports at most two nonlinear dependency layers; every allocated
hidden layer must fit 32 units INCLUDING relays. A source with a raw-input skip
across a hidden layer needs positive/negative relay channels; count that space.
No skip connection is added to the fixed target. A purely affine output is
carried by positive/negative output relays. Nonnegative hidden states can use
identity ReLU relays; remaining neurons/weights are zero padding. All such
parameters count. Compiler construction guard:128 source/relay units and100,000
dense parameters before final fixed-target fit, at most two hidden layers.
Allocation errors are not mathematical proofs that no such network exists.

The primary comparison is useful behavior plus proposal/usage cost. Source
equivalence, construction/solver time and rounding diagnostics are reported
separately. They do not certify task success, and there is no iterative repair.
