# Common diffusion coordinate-edit task

Write one reusable program to transform an arbitrary supplied denoiser of the
architecture below. The same program will be executed independently on three
released trained models. No generation, execution feedback, tools, retries or
training is available to you. The execution happens only after your final reply.
Both interfaces below are supplied to both arms. Only SELECTED ARM differs.

## Required behavior

Pixels form a 12x12 image, row-major. Define P to move each pixel two columns to
the right, wrapping at the right edge: old (r,c) becomes (r,(c+2) mod 12).
Time channels must not be shifted. For every finite real pixel vector x and time
t, the edited network g must satisfy `g(P*x,t) = P*f(x,t)`, where f is the parent.
Preserve the parent's appearance through this coordinate change; do not train,
substitute a shape template, or call the parent at inference. Use the same fixed
architecture and six tensors, all finite FP32. Every tensor is editable.

```
input = concatenate(noisy_pixels[144], time_embedding[16])
h1 = relu(input @ W1 + b1)    W1 [160,64], b1 [64]
h2 = relu(h1 @ W2 + b2)       W2 [64,64], b2 [64]
x0 = h2 @ W3 + b3            W3 [64,144], b3 [144]
```

Rows index incoming units; columns outgoing units. Time embedding is eight sine
then eight cosine values with frequencies 10000**(-k/8), k=0..7. The sampler uses
x0 predictions clipped to [-1,1]. For paired generation comparisons, all initial
and subsequent pixel noise is shifted by P for the candidate. No external shape
classifier is used. Fixed functional probes and paired samples must agree within
absolute tolerance .001. A sufficient exact tensor-structure certificate is also
reported independently; absence of that certificate is not automatic failure.

Your program reads actual parent tensors at construction time. Their values and
shape-class labels are not in this packet because the requested transformation
must work without depending on their numerical values or the generated class.
No solution program is supplied. Return a brief explanation and exactly one
fenced block, tagged `python` or `wspl` for SELECTED ARM. Entire reply ≤200,000
UTF-8 bytes. Invalid programs fail in full; no repair or best-of selection.

## Python interface

Full Python 3.12 with NumPy 1.26.4, not a literal-assignment subset. Read the six
arrays from `/parent.npz` using `np.load(...,allow_pickle=False)`. Save exactly
W1,b1,W2,b2,W3,b3 to `/output/candidate.npz` with `np.savez`; same shapes and
FP32 dtype. Import NumPy explicitly. Use loops, indexing and construction
arithmetic freely. Do not run forward passes, fit weights or sample the network.
The code runs once per parent in a fresh sandbox. No network/GPU, subprocesses,
host files or returned execution feedback. CPU limit 15s, wall 20s, address
space 2GiB, candidate file 2MiB, scratch output tmpfs 2MiB. Stdout discarded,
stderr archived but not returned. Temporary files under /tmp are sandbox-local.

## WSPL interface

Full WSPL 0.2 fixed-dense editing. Supply this target binding, with the literal
placeholder replaced by the harness using each actual parent identity:

```
wspl 0.2;
target base=artifact("parent",sha256="{{PARENT_SHA256}}");
edit base as candidate mode=fixed numeric=exact_to_storage {
  // your edits
}
```

Unwritten coefficients retain parent storage. `parent.W1[...]` etc read immutable
parent tensors. `W1[...]` etc read current storage; statement-start snapshots.
Assignments `=`, `+=`, `-=`, `*=`; scalar/array `+ - * / @`; literal nested lists;
indexing and half-open slices; broadcasting; `copy`, `reshape`, `transpose`,
`zeros`, `full`, `eye`, and `f32bits`. Constants before the edit use `const a=...;`.
Bounded nested loops: `for r in range(12) { ... }`, also range(start,stop,step).
There is no modulo, floor division, if statement or NumPy advanced indexing;
split ranges or write equivalent arithmetic. Exact rational arithmetic rounds
each write directly to FP32. Pure copies retain nonzero FP32 values exactly.
Maximum 23,824 tensor elements, 4,096 statements, 4,096 loop iterations,
100,000,000 charged construction operations. The fixed dense graph cannot grow.

Optional scalar affine/ReLU circuit declarations and explicit placement are
available, but not required. A syntax-only example on hypothetical compatible
dimensions is:

```
circuit Gain(a:Real)->(b:Real) { b = 2*a; }
```

Such a circuit requires explicit placement at a compatible next-layer producer;
there is no automatic allocator or implicit relay. Ordinary tensor edits are
equally valid. Report the mechanism you actually used, without claiming that
writing a loop demonstrates general graph surgery.

Declare the circuit before `edit`. An example placement inside `edit` is
`place Gain(a=h2[0]) at { b=y[0]; } writes=replace_producers;`.
This replaces that output's entire incoming column and bias. Bind every circuit
input, place every assigned local/output exactly once, and use only preceding
layer signals for each affine producer. ReLU assignments must occupy ReLU layers;
linear assignments must occupy linear layers. These examples specify syntax,
not the requested transformation.
