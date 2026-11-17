# Implemented WSPL syntax

This reference describes the parsers and interpreters shipped with this repository.
It separates syntactic acceptance from semantic validation and verification.
An accepted source file can still fail shape, type, placement, resource-budget or
proof checks. The [README](../../README.md#what-the-implementation-actually-accepts)
contains the core statement grammars and function whitelists.

## 1. Core PWA modules

Authority: [wspl_parse.py](../../src/skillforge/wspl_parse.py) and
[wspl_core.py](../../src/skillforge/wspl_core.py).

A file contains one `module Name { ... }`, with no trailing program. Statements
end in semicolons. Whitespace and `//` line comments are accepted; `#` comments
are not. Module/contract identifiers match `[A-Za-z_][A-Za-z0-9_]*` and must not
be reserved words. Ports, lets and unit labels match `[a-z][a-z0-9_]*`.

Numbers are integers or decimals such as `12`, `0.25`, `.25`; signs are unary
operators. Scientific notation and trailing-dot literals such as `1.` are not
accepted. Fractions use division, e.g. `1/4`. Constant expressions support
parentheses and `+ - * /`. Arithmetic precedence is unary signs, multiplication/
division, then addition/subtraction.

```wspl
module Rectifier {
  input x: Real[-1,1] normalized_value;
  output y: Real[0,1] normalized_value;
  let positive: Real[0,1] = relu(x);
  y = positive;
  ensures Nonnegative: forall inputs: y >= 0;
}
```

Inputs and lets must exist before executable references to them. Outputs cannot
be read by executable expressions. Duplicate declarations, undeclared assignment
targets, repeated output assignments and unassigned outputs are rejected.
Let units may be inferred; input/output Real units are mandatory. Labels starting
with `normalized_` all denote dimensionless values. Other labels denote their
named units. Addition/min/max require compatible units. Multiplication requires
a constant operand; division requires a nonzero constant denominator.

The executable functions are exactly `relu`, `clip`, `margin`, `min`, `max`,
`not`, `and`, `or`, `implies`, with the arities in the README. In particular,
`abs`, trigonometric functions, indexing, tensors, function definitions and
control-flow statements are not module expressions. Absolute value can be
expressed as `max(x,-x)`. Graded truth is numerical: `not(p)=1-p`,
`and(p,q)=max(0,p+q-1)`, `or(p,q)=min(1,p+q)`, and
`implies(p,q)=min(1,1-p+q)`. `margin(x,lo,hi)` clips `(x-lo)/(hi-lo)` into `[0,1]`.

Contracts have exactly the prefix `ensures NAME: forall inputs:` or
`preserves NAME: forall inputs:`. Their expressions are affine in input/output
ports, with constant multiplication/division. They cannot reference lets or call
functions. Boolean precedence is `not`, `and`, `or`, then right-associative
`implies`. Comparisons are `< <= > >= ==`; there is no `!=` in this frontend.
`forall inputs` is a fixed form, not a general quantifier language. Both contract
keywords describe the current module's I/O behavior. Verification is separate
and must bind the contract to the compiled artifact.

Default limits include 1 MB source, nesting depth 64 and 1,024 scalar IR nodes;
the parser API permits an explicitly chosen node budget. Interval checks are
conservative, so a mathematically valid function can need a different expression
or stronger proof support.

## 2. Tensor and graph editor, version 0.2

Authority: [wsp_syntax.py](../../src/skillforge/wsp_syntax.py),
[wsp_editor.py](../../src/skillforge/wsp_editor.py),
[wsp_values.py](../../src/skillforge/wsp_values.py) and
[wsp_circuit.py](../../src/skillforge/wsp_circuit.py).

### File structure and declarations

```text
wspl 0.2;
target NAME = TARGET_CONSTRUCTOR;
const NAME = CONSTRUCTION_EXPRESSION;
circuit NAME (PORTS) -> (PORTS) { ASSIGNMENTS }
region NAME in TARGET { GRAPH_FIELDS }
edit TARGET as CANDIDATE mode=fixed numeric=exact_to_storage { STATEMENTS }
describe CANDIDATE { EVIDENCE }
```

This is a structural template: replace uppercase placeholders; omit unused
declarations. Target/constant/circuit/region declarations precede the edit; their
kinds may be interleaved. Exactly one target is required. At most one edit is
allowed, and zero or more descriptions follow it. Declaration names must be
unique and cannot be `parent`, `x`, `y`, or `W`/`b`/`h` followed by digits.
Additional evaluator checks prevent collisions with built-ins and tensor names.

Names use `[A-Za-z_][A-Za-z0-9_]*`. Both `//` and `#` line comments are accepted.
Strings use double quotes. Decimal/scientific numeric literals are supported in
exact construction arithmetic. `f32bits(0x3f800000)` supplies an explicit finite
FP32 bit pattern. Python-looking expression syntax is interpreted through an AST
whitelist; it does not authorize arbitrary Python execution.

Target constructors:

```wspl
target base = dense(widths=[2,3,1], activations=[relu,linear],
                    layout=in_out, storage=f32, init=zeros);
```

Or `target base = artifact("name", sha256="FULL_64_HEX_DIGEST");`, with the actual
parent supplied by the host. Dense targets require `widths` and `activations`;
optional defaults are `layout=in_out`, `storage=f32`, `init=zeros`. Layout can
also be `out_in`; storage and initialization currently have only those single
values. Activations are `linear`, `relu` or `tanh`, one per affine layer.

Fixed edits use `mode=fixed`. Graph edits use
`mode=graph budget="HOST_BUDGET_NAME"`. Optional `numeric=...` and `budget=...`
can appear in either order after the mode. Graph mode requires a host-provided
budget and the exact profile; fixed mode forbids a graph budget. The parent must
be the declared target and the candidate name must be fresh.

### Construction expressions and writes

| Form | Constraint |
|---|---|
| Numbers, rectangular lists/tuples | Finite values; bounded size |
| `+a`, `-a`, `a+b`, `a-b`, `a*b`, `a/b` | Exact rational arithmetic with compatible broadcasting |
| `a @ b` | Compatible rank-two matrices |
| `W1`, `parent.W1`, `W1[i,j]`, `W1[:,0:4]` | Candidate or immutable-parent reads; basic indexing/slices |
| `copy(a)` | Snapshot copy |
| `transpose(a)` | Rank-two transpose |
| `reshape(a, shape)` | Same element count |
| `zeros(shape)`, `full(shape, value)`, `eye(n)` | Explicit bounded construction |
| `f32bits(integer_literal)` | Finite FP32 storage word |
| `tensor("binding", sha256="DIGEST")` | Host-supplied, identity-checked tensor binding |

These are the complete construction calls. There are no imports, arbitrary
attributes/methods, comprehensions, conditionals, random-number calls, `while`,
exponentiation or general function definitions. `relu` is a circuit operation,
not a construction-array function.

Writes use `=`, `+=`, `-=`, `*=` on declared tensor selections. Under the exact
profile, `W1 = value;` is full-tensor slice sugar. Explicit indexing must name
every axis: use `W1[0,:]`, not `W1[0]`, for a matrix row. Integer indices and
`start:stop:step` slices are supported, with negative indices and nonzero strides;
out-of-range integers fail and slice clipping is recorded. No advanced indexing
or ellipses. Right-hand sides are read before the statement writes. Subsequent
statements see updated candidate storage, while `parent` remains unchanged.

`for i in range(stop)`, `range(start,stop)` and `range(start,stop,step)` expand
construction statements within braces. Bounds must evaluate to integers and fit
the host limits. Nested loops are allowed; shadowing bindings is rejected. They
finish before neural inference begins.

The default `numeric=exact_to_storage` rounds exact arithmetic to FP32 at each
write. `numeric=native_numpy_v1` is a narrower compatibility profile: literal
slice assignments and the three augmented assignments, plain decimal literals,
historical NumPy arithmetic and value limits. It rejects construction loops and
circuit placement. Selecting this profile does not enable NumPy source code.

### Circuits and graph operations

A circuit declares nonempty named input/output ports with `Real`, `Real[lo,hi]`
or `Truth` types, then single-assignment signals. Each signal is an affine
expression or one outer `relu(affine_expression)`. Products of varying signals,
nested ReLUs and direct `clip`/`min`/`max` calls are rejected by placement; expand
them into explicit affine/ReLU signals. Intermediate annotations support only
plain `Real`. Port refinements are recorded as **unverified obligations**, not
used as established bounds during lowering.

`place C(input=signal, ...) at { local=h1[0]; output=y[0]; }
writes=replace_producers;` explicitly locates every assigned signal. Destinations
must have matching activations and available edges from the preceding layer.
There are no implicit relay neurons or automatic capacity increases.

The graph operation forms are `replace`, `transplant`, `add_residual`, `redirect`
and `untie` as listed in the README. Their field schemas, region declarations,
boundary aliases, sharing rules and runnable examples are in the
[graph surgery reference](WSP_GRAPH_SURGERY.md). Fields are not arbitrary options:
the surgery evaluator checks them and the host budget. Supported graph operators
are affine, ReLU, tanh, identity, add, scale, concat and slice. This graph adapter
does not provide recurrent, convolutional or attention graph imports.

### Descriptions and claims

```text
describe CANDIDATE {
  view NAME = EXPRESSION;
  hypothesis "Human interpretation, not a proved fact";
  claim NAME semantics=real_graph {
    assume EXPRESSION;
    ensure EXPRESSION;
  }
}
```

Views and claims have unique names within a description. Claims need at least
one `ensure`; assumptions are optional. Evidence expressions are a separate
whitelist supporting affine arithmetic, comparisons, Boolean combinations,
`relu`, `min`, `max`, `clip`, and graph observations such as `pre`, `post`, `tap`.
See [wsp_evidence.py](../../src/skillforge/wsp_evidence.py) for observation binding
and interval-membership rules. The checker supports `semantics=real_graph` for
its supported operators; other requested semantics yield `UNSUPPORTED`.
Parsing, placing or describing a circuit does not certify FP32 execution or its
task performance. Host budgets and the checker's supported operator set still
apply.

## 3. Robot executive action language

Authority: [control_dsl.py](../../src/skillforge/control_dsl.py). This is separate
from both weight-construction frontends. It accepts exactly these instructions,
one per line, with optional `#` comments and blank lines:

```text
move_to OBJECT_OR_POINT
grasp OBJECT
release OBJECT
transfer OBJECT to POINT
```

Names must exist in the supplied, disjoint object/point namespaces and match
`[A-Za-z_]\w*`. No semicolons, loops, branches or embedded WSPL. `transfer`
expands to move-to-object, grasp, move-to-point, release-object. The default
expanded instruction budget is 128; empty programs fail. Runtime completion,
safety monitoring and bounded failure are interpreter behavior, not extra
source-language statements.
