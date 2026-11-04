# WSPL tensor and circuit programming design

Design, November 3, 2026. Kevin requested language/tool alignment before another
experiment. A [first implementation pass](WSP_EDITOR_FIRST_PASS.md) now covers
direct FP32 edits, explicit affine/ReLU placement and frozen descriptions.
[Graph surgery](WSP_GRAPH_SURGERY.md) now adds stateless vector DAG replacement,
transplantation, rewiring, residuals and explicit parameter sharing. This document
remains the larger design; those implementation pages record the exact supported
subsets. The historical Core PWA parser is unchanged.

The [syntax companion](WSP_SYNTAX_0_2.md) specifies the initial grammar, expression
contexts, write ordering and tool request shapes.

## The interface I would want to use

I need to inspect an actual network, select the part I intend to change, write
that change without losing control of placement, and see exactly what artifact
results. I should be able to start with literal weights, discover a useful
pattern, name it, package it as a circuit, reuse it, and later attach checked
claims. None of those steps should require pretending an unexplained neuron has
an established meaning.

**WSPL should make the full target network addressable.** Direct tensor editing
is a first-class language feature. Circuit placement and expression compilation
are additional ways to produce edits to the same artifact. The verifier examines
the artifact produced by any path; verifier coverage does not define which
networks the editor can express.

The initial target is stateless dense neural networks. The artifact design must
also accommodate explicit graph rewrites and opaque operators. Execution support
and verification support are separate capability tables. ReLU is the current
formal backend's boundary, not a language-wide ban on tanh or other activations.

## Decisions

| Decision | Reason |
| --- | --- |
| Every tensor element can be read and written directly | The allocator must never restrict the native editing path |
| Preserve the MIT-style slice syntax | It already gives concise, useful control of fixed networks |
| Add reusable circuits with explicit placement | An editor can control where computation lives and how it is connected |
| Permit graph changes under a separately declared topology budget | Replacing a subgraph is different from editing a fixed network's weights |
| Infer intermediate shapes and ranges; annotations are optional obligations | Boilerplate bounds should not obstruct construction |
| Keep descriptions and claims outside executable identity | Naming a circuit must not alter its weights or make a hypothesis true |
| Preserve staged candidates even when verification is unresolved | Construction, evaluation and admission have different purposes |
| Make every operation inspectable before application | An editor needs usable diagnostics, an exact diff and reversible branches |

These are proposed defaults. The fixed-versus-graph distinction and independently
owned admission requirements are explicit configuration, not implicit compiler
choices.

## One artifact, three authoring levels

1. **Tensors:** literal edits, slice arithmetic, copies and bounded construction.
   No expression allocator runs.
2. **Circuits:** reusable computation with named internal nodes, connected to
   existing signals and explicitly placed into existing neurons or graph nodes.
3. **Expressions:** ordinary affine/PWA expressions with automatic allocation
   when requested. The plan exposes generated neurons, relays and all costs.

All three produce a common edit plan containing exact storage writes and/or graph
changes. They can be mixed in one ordered transaction. The canonical artifact is
the graph and stored tensors, not a scalar source program reverse-engineered
from them. A directly authored artifact never needs to survive a source compiler
round trip in order to be executable.

For a supported fixed architecture, every allowed finite tensor assignment must
be representable by literal or raw-bit writes. This is the expressivity floor.
It is about access to parameter space; it is not a claim that a small network can
represent every controller or that automated circuit placement is complete.

## Target declarations and familiar raw edits

Proposed complete example:

```wspl
wspl 0.2;

target policy = dense(
    widths=[600, 32, 32, 6],
    activations=[relu, relu, linear],
    layout=in_out, storage=f32, init=zeros
);

edit policy as tracking mode=fixed numeric=exact_to_storage {
    W1[3:600:60, 0] = 1;
    b1[0] = -0.5;
    W2[0, 0] = 1;
    W3[0, 1] = -1;
    W3[0, 3] = 1;
}
```

`dense` supplies `W1,b1,...,Wn,bn`, input signal `x`, hidden signals
`h1,...,h(n-1)`, and output signal `y`. Under `in_out`, inference at layer k is
`h_previous @ Wk + bk` followed by its declared activation. `pre(h1)` and
`post(h1)` distinguish preactivation from activation; `h1` alone means post.
`out_in` is also representable but must be declared and appears in every view.
The storage adapter never guesses a transpose from a tensor's shape.

For an existing model, the equivalent declaration is:

```wspl
target policy = artifact("policy", sha256="<64 hex digits from inspect>");
```

The quoted name resolves through the caller's artifact registry, not an arbitrary
network download or executable import. Placeholders in documentation are not
valid hashes. An imported target supplies its actual tensor names, graph, dtypes,
aliases, buffers, preprocessing and output semantics. Adapters can provide the
convenience W/b names, with the mapping visible. A graph hash and a storage hash
bind both the computation and its parameters.

### Literal editing semantics

- Support integer indices, negative indices, half-open slices and positive or
  negative nonzero strides, scalar/list/nested-list values, `=`, `+=`, `-=`,
  `*=`, and comments. NumPy-compatible basic slicing and broadcasting are the
  floor; empty/clipped slices are reported prominently rather than hidden.
- Statements execute in source order on a private candidate. Each right-hand
  side reads a snapshot taken at the start of that statement. Overlapping copies
  therefore have defined behavior. The parent remains immutable.
- Add tensor reads, `copy`, `transpose`, `reshape`, `zeros`, `full`, `eye`, `@`,
  constants, and compile-time `for ... in range(...)`. These construct weights;
  they do not add inference-time code. Shapes and expansion/work limits are
  checked before expensive materialization. No arbitrary Python execution,
  imports, filesystem access or hidden optimizer is implied.
- Integer, decimal, scientific-notation and rational constants are accepted.
  `f32bits(0x3f800000)` specifies exact stored bits, including signed zero where
  relevant. Nonfinite writes are rejected in the first profile. Other dtypes
  need their own bit constructors and conversion rules before support is claimed.
- With `exact_to_storage`, constants and construction arithmetic are rational;
  each assignment rounds once, directly to the destination dtype, nearest-even.
  An augmented assignment starts from the exact stored value. Rounding changes
  are reported. This specifies weight construction, not inference arithmetic.
- A separately named `native_numpy_v1` profile uses the pinned historical
  assignment implementation/runtime. It preserves valid legacy edit behavior,
  including its numeric conversions. Do not claim that exact arithmetic and
  NumPy augmented-assignment arithmetic always give the same bits.
- A new WSPL transaction fails atomically on an invalid statement. The historical
  skip-invalid-lines rule is available only through an explicit legacy importer
  with its complete diagnostics. Failed statements must never disappear silently.
- Broadcasting does not imply parameter sharing. Tied storage is listed in the
  target manifest; editing it reports every alias and affected consumer. An
  independent copy requires explicit untie/copy permission in graph mode.

The native importer should accept the original `edits` fence unchanged, retaining
its original text, line numbers and profile. Semicolon insertion belongs in a
lossless wrapper, not in a reinterpretation of coefficients or placement.

### Useful weight programs beyond literals

These construction operations should be ordinary programs in the same language:

```wspl
// Copy a block from the immutable parent, not from a changing candidate view.
W2[0:8, 16:24] = copy(parent.W2[0:8, 0:8]);

// A caller-supplied low-rank update; U and V are pinned construction tensors.
W2[:, :] += U @ V;

// Positive rescaling of one ReLU producer and all of its next-layer consumers.
W1[:, 3] *= 2;
b1[3] *= 2;
W2[3, :] *= 0.5;
```

The rescaling illustrates a circuit-preserving transformation under real ReLU
semantics when those are all its consumers. It needs a complete consumer check
and a separate numerical analysis before claiming deployed preservation. Similar
library recipes can permute units with all corresponding consumer columns/rows,
clone a circuit, or splice a learned replacement. These are transparent edit
expansions with provenance, not privileged operations that bypass the diff.

Construction constants and pinned tensors are declared in the file or supplied
as named bindings in the plan request. Their content hashes enter the plan.
`parent.*` is read-only, tensor expressions without that prefix read the current
candidate at statement entry, and activations such as `h1` are unavailable in
weight-construction arithmetic. Activation-dependent behavior belongs in the
network being constructed. This prevents accidental runtime computation in an
apparently static weight edit.

## Circuit placement that the author controls

A circuit is a reusable acyclic computation, with named intermediate nodes and
ports. It can have a learned frozen tensor implementation or a source-defined
implementation. A name describes a location; its interpretation remains a claim.

Here is a complete small example of source-defined placement:

```wspl
wspl 0.2;
target control = dense(
    widths=[2, 4, 1], activations=[relu, linear],
    layout=in_out, storage=f32, init=zeros
);

circuit Deadband(e: Real) -> (u: Real) {
    p = relu(e - 0.25);
    n = relu(-e - 0.25);
    u = p - n;
}

edit control as deadband mode=fixed numeric=exact_to_storage {
    place Deadband(e=x[0]-x[1]) at {
        p = h1[0];
        n = h1[1];
        u = y[0];
    } writes=replace_producers;
}
```

This placement must yield exactly:

```text
W1 = [[ 1, -1, 0, 0],
      [-1,  1, 0, 0]]
b1 = [-0.25, -0.25, 0, 0]
W2 = [[1], [-1], [0], [0]]
b2 = [0]
```

There is no hidden allocation choice: two existing ReLU units and one existing
linear output are assigned. `replace_producers` writes the full incoming weight
vector and bias of every placed node, zeroing unmentioned inputs to those nodes.
It preserves other producers. Outgoing edges from the hidden nodes to consumers
outside the placement remain unchanged and are listed in the plan. The caller
must explicitly edit or redirect those if different behavior is intended.

The same circuit's output could be bound to an existing graph signal rather than
a final output if the operator and placement constraints match. Binding a signed
linear output to a post-ReLU neuron is invalid without an explicit representation
change; the compiler must not silently add a clamp.

The circuit body infers shapes and conservative ranges. `Real` means no optional
range refinement was supplied; it is not an unproved finite bound. A declared
`Real[lo,hi]` or `Truth` creates an obligation. Failure to establish a refinement
does not make a direct weight edit inexpressible, but a check or transformation
that requires that refinement cannot use it without evidence.

### Allocating versus placing

`place ... at {...}` specifies all computational nodes and requires no unrequested
neurons. It fails with an exact missing-edge/operator/slot diagnostic if the
placement cannot work. An explicitly selected `allocate ... within {...}` may
search for placement and insert relays inside the declared region. It must first
return a plan; it cannot expand the network automatically.

The expression compiler must use established sign/range facts to avoid redundant
relays. For example, `relu(-x)` is zero when `x >= 0`, so carrying both signed
rails is unnecessary on that domain. An assumption used for this optimization
becomes part of the artifact's validity domain and evidence. It cannot be inferred
from a few observed samples.

Allocation failure means the current algorithm did not produce a valid placement.
It is not proof of nonrepresentability. Literal editing remains available even
when the automatic path fails. The full-target direct encoding and an explicit
placement should be attempted before attributing a capacity limit to the network.

## Editing complete circuits

The useful operations are parameter edits, replacing producers, transplanting a
bounded subgraph, composing a residual path and rewiring selected consumers.
They need different semantics, rather than one ambiguous `replace` operation.

| Operation | Exact meaning |
| --- | --- |
| Tensor assignment | Changes the addressed storage, preserving topology |
| `place ... writes=replace_producers` | Rewrites complete incoming tensors/biases for the placed neurons; preserves their external outgoing edges |
| `replace region ...` | Substitutes a declared subgraph with explicit mappings for every incoming and outgoing boundary edge |
| `add_residual ...` | Adds a branch and an explicit add node, with declared output shape, numerical order and scale |
| `redirect consumer.port = signal` | Replaces one edge; does not implicitly redirect every use of the old signal |
| Copy/transplant | Copies immutable circuit parameters with a port/node mapping; parameter sharing must be requested separately |

Graph surgery is permitted only in `mode=graph`, under a host-owned budget for
operators, nodes, parameters, state and interfaces. Fixed mode permits storage
edits and placements into existing compatible producers only. A graph mode edit
that adds a branch is reported as augmented even if its trainable parameter count
is unchanged. Removing a node still requires resolving every consumer.

A proposed region declaration and replacement looks like this:

```wspl
region steering in policy {
    nodes = [node("hidden_a"), node("hidden_b")];
    inputs = { obs: edge("input_to_hidden_a") };
    outputs = { command: edge("hidden_b_to_readout") };
}

edit policy as repaired mode=graph budget="repair_budget" {
    replace steering with artifact("new_steering", sha256="<64 hex digits>") {
        inputs = { input: steering.inputs.obs };
        outputs = { steering.outputs.command: output };
    }
}
```

This is a topology example, using node/edge IDs returned by inspection. Boundary
declarations must exhaust the actual graph cut, including skip edges, tied
parameters and shared consumers. A partial boundary is an error with the missing
edges listed. Removing the old region and installing the replacement happens
atomically after shape, dtype, operator and acyclicity checks. A replacement's
assumed input domain is a separate obligation, not established by equal shapes.

For the initial implementation, regions span complete operator nodes. Neuron
slices are handled by explicit placement and tensor edits. Splitting a dense
operator into graph nodes changes the execution graph and may change floating
point reduction order; it is a later, explicit rewrite, never a free annotation.

Opaque supported operators may be retained or edited with an execution adapter
even when the formal backend cannot encode them. Their checks return UNSUPPORTED
or empirical results, not PROVED. Stateful/recurrent graph editing is deferred
until state, clock, reset and transition semantics are part of the interface.

## Names, explanations and post hoc specifications

An annotation binds a storage/graph identity to views and claims. It never writes
weights. Views can overlap, combine multiple signals and label hypotheses:

```wspl
describe deadband {
    view positive_error = h1[0];
    view negative_error = h1[1];
    hypothesis "positive_error responds to error above the deadband";

    claim response semantics=real_graph {
        assume x[0] in [-1, 1] and x[1] in [-1, 1];
        ensure y[0] == relu(x[0]-x[1]-0.25)
                     - relu(x[1]-x[0]-0.25);
    }
}
```

Claims reference the frozen target, not its author's claimed source map. The
checker resolves the actual taps and coefficients. An equation involving hidden
signals needs an internal-signal encoding, which the current I/O-only contract
parser does not implement. Names and natural-language hypotheses remain unproved
even when a narrower arithmetic claim succeeds.

The same description mechanism must work on directly authored, compiled and
gradient-trained networks. Partial explanations are allowed: the report identifies
which outputs/paths/regions remain undescribed and which claims were checked.
An exact dump of every matrix is faithful but is not automatically an informative
explanation. No single opaque "interpretability percentage" should combine these.

An external requirement set supplies the primary behavior and preservation
obligations. Candidate-authored claims can supplement it, never replace it.
Post hoc explanations cannot change the artifact being explained. A simplifying
rewrite produces a new artifact and requires an explicit relation to its parent.

## Numerical semantics and identities

Maintain four separate records:

1. Executable graph identity: operators, edge order, ports, preprocessing,
   postprocessing and state, if supported.
2. Storage identity: canonical tensor names, shapes, dtypes, aliasing and exact
   bytes. Preserve signed zero; reject undeclared nonfinite edits.
3. Runtime profile: backend/version, operation and reduction behavior, device,
   precision and execution settings. A generic dtype label is not a complete
   bitwise execution specification.
4. Description/evidence identity: source, names, domains, claims, checker profile
   and receipts, all referring to the executable/storage identities.

The candidate ID binds graph, storage and runtime requirements. Annotation changes
create a new description ID, not new weights. A direct-edit receipt records
literal rounding and exact changed elements. The formal adapter interprets actual
stored finite coefficients as exact rationals when supported. That supports
real-graph statements; it does not establish IEEE execution equivalence.

Resource settings belong to a host profile: source bytes, statement count,
construction arithmetic size, tensor elements, expanded loops and time limits.
The language does not hard-code Breakout's 32-unit layers, coefficient cap or
1,024-node scalar limit. Larger profiles are still bounded and inspectable.

## Tools designed for an editor

The tool workflow is **inspect -> plan -> apply -> check/probe -> diff**. All
responses include artifact identities, machine-readable results and a concise
human-readable rendering. Large tensors are paged or referenced by hashed files;
do not flood the model context with dense arrays.

| Tool | Input | Useful output |
| --- | --- | --- |
| `wsp.inspect` | Artifact ID, selection, detail/byte budget | Topology, tensor orientation, aliases, pre/post taps, fan-in/fan-out, execution/verifier capabilities, paged values |
| `wsp.plan` | Parent ID, versioned WSPL text, host profile ID | Resolved writes, placement, graph cut, rounding, added capacity, affected consumers, diagnostics, immutable plan ID |
| `wsp.apply` | Plan ID and expected parent ID | New immutable candidate ID and edit receipt; parent untouched |
| `wsp.describe` | Artifact ID and optional description text | Exact graph-derived equations/views, or a separately stored proposed description; no weight changes |
| `wsp.check` | Artifact/parent IDs, requirement set, semantics, budget | Per-obligation results, assumptions, witnesses, unsupported coverage and evidence IDs |
| `wsp.probe` | Artifact IDs, pinned input fixture, taps/interventions, seed | Paired outputs/activations and empirical differences; no implicit training |
| `wsp.diff` | Parent and candidate IDs, selection | Touched/changed weights, graph edits, rounding, structural dependency changes, evidence made stale |

Tool names are proposed, not registered services. The implementation should expose
the same schemas through a local Python API and CLI so it is usable without a
particular LLM provider. `check` does not automatically run a game or simulator.
Scientific evaluation/model calls retain the repository's existing launch rules.

### A useful planning failure

An editor needs diagnostics like:

```json
{
  "status": "INVALID_PLACEMENT",
  "source_span": {"line": 14, "column": 9},
  "node": "u",
  "requested": "post(h2[0])",
  "reason": "linear output assigned to ReLU output",
  "required": "linear signal or explicit signed representation",
  "parent_unchanged": true
}
```

An allocation failure should distinguish computational units from relays, list
the actual overfull layer and state which simplifications were attempted. Invalid
syntax, insufficient construction budget, unsupported lowering and solver timeout
are distinct. The tool may suggest edits, but must not apply its suggested repair
without a new plan.

### Transactions and evidence

A valid plan can create a STAGED candidate with no behavioral proof. A separate
host-owned admission policy decides whether that candidate becomes the selected
version. Policies can require exact properties, empirical thresholds, both, or
simply permit exploratory retention. The editor cannot reduce those requirements
inside its source. An unverified experimental artifact stays labelled as such.

Track validation, claim results and admission separately. For example, a candidate
can be structurally VALID, have a claim UNKNOWN due to timeout, and be UNRESOLVED
under an admission policy requiring that claim. An empirical probe pass is never
substituted for PROVED. A failed requirement can reject admission while retaining
the candidate for diagnosis. No policy-free COMMITTED status should imply safety.

Plans bind the parent, text, profiles, compiler version, expansion and resulting
writes. Applying a stale plan fails. An empty diff is reported as NO_CHANGE.
Undo selects a retained parent; it is not an approximate inverse edit. Independent
branches can be compared without touching the active artifact. Changes outside
the write mask are checked bitwise, but unchanged weights do not imply unchanged
behavior if their inputs or consumers changed. Evidence is invalidated by its
full dependency set; uncertain dependencies cause conservative invalidation.

## What I deliberately would not build first

No autonomous optimizer hidden behind `repair`. No arbitrary Python plugin loaded
from a proposal. No automatic semantic labels based on a circuit's chosen name.
No general neural program decompiler promise. No dynamic controller execution in
the weight-construction language. Static construction loops and inference-time
recurrence are separate features. No silent architecture growth to make a source
program fit. These choices keep the initial tool understandable and debuggable.

## Implementation order and completion criteria

The existing Core PWA parser remains a versioned frontend. Do not silently change
historical numeric, type or allocation semantics. New syntax uses a separate
parser entry point. Existing historical manifests and frozen runs stay intact.

| Slice | Concrete deliverable | Completion criterion |
| --- | --- | --- |
| A: complete parameter access | Generic tensor manifest, versioned edit parser, immutable candidate and diff | Every permitted stored value addressable; valid legacy transcripts reproduce exact historical tensors in compatibility mode |
| B: placed circuits | Circuit definitions, explicit placement, producer replacement, source-to-storage mapping | Deadband source and raw edits yield identical tensors; cross-consumer effects and missing edges visible |
| C: descriptions and checks | Frozen-artifact descriptions, internal taps, independent requirement binding | Descriptions cannot mutate weights; stale and false descriptions produce explicit failures; unknown remains unknown |
| D: complete graph edits | Regions, cut validation, transplant/redirect/residual, topology budgets | Every boundary/alias resolved; shape/cycle/resource failures atomic; augmentation counted |
| E: automatic allocation | Sign-aware relays, source inference, opt-in placement search | Requested architecture preserved; no redundant negative rail for a proved nonnegative input; failure does not block direct editing |

A, B and C are the next implementation unit. D is designed now so the artifact
and transaction format will not need to be replaced later. E improves convenience;
it is no longer a prerequisite for native weight-space programming.

Engineering conformance cases before another scientific test:

1. Entire permitted fixed target populated by arbitrary finite literal/bit writes;
   no compiler invocation, added units or inaccessible tensor cells.
2. Legacy positive/negative/strided slices, broadcast and augmented assignments
   reproduce stored bytes, including an existing archived valid native transcript.
3. Invalid late statement leaves parent and selected artifact unchanged; candidate
   rejection lists the error without losing earlier source text.
4. Overlapping copy uses a statement snapshot; aliases and repeated writes are
   reported; zero-size/clipped writes cannot look like successful repairs.
5. Values around FP32 rounding boundaries distinguish exact and compatibility
   profiles, and raw bits preserve signed zero.
6. Raw Deadband and explicit circuit placement have identical complete tensors;
   a pre/post-ReLU placement mistake fails with a useful location.
7. A preexisting nonzero producer is fully replaced, including forgotten biases
   and off-region inputs; unrelated storage is unchanged.
8. A shared consumer appears in the footprint even when its weights are unchanged.
9. Graph cut with a missing skip edge or tied parameter fails atomically; explicit
   replacement succeeds without dangling edges or accidental state creation.
10. Tanh tensor editing can succeed with formal checks UNSUPPORTED; execution
    capability and verifier capability remain separate.
11. A frozen-artifact description adds zero executable changes; a false arithmetic
    description is refuted and a natural-language hypothesis stays unproved.
12. Claim/domain/runtime/hash changes invalidate the appropriate evidence; an
    independently required behavior cannot be removed by a candidate annotation.
13. Automatic allocation respects a declared nonnegative input domain and exposes
    relay costs; original historical artifacts are not rewritten.
14. Huge construction loops, resource exhaustion, malformed shapes and stale plan
    application fail before partial mutation or uncontrolled expansion.

These are acceptance specifications for the full design; the
[first-pass report](WSP_EDITOR_FIRST_PASS.md) distinguishes implemented and
checked features from remaining work. There is no new experiment proposal in
this design. The next research question can be selected after the authoring
interface reaches the required expressivity floor.

## Relationship to the current implementation

The current [native reader](../../src/skillforge/native_weight_edits.py) already
provides data-only tensor slices but is hard-coded to six Breakout tensors and
historical skip-invalid behavior. Generalize through a new frontend; retain it
for reproducibility. [Core PWA](PARSER.md) and the [compiler](COMPILER.md) remain
useful expression tools, with their current limitations explicit.

[Frozen imports](../../src/skillforge/wspl_import.py),
[composition](../../src/skillforge/wspl_compose.py) and
[transactions](../../src/skillforge/wspl_transaction.py) provide important pieces.
They currently target hidden-ReLU/final-linear artifacts and output-patch
augmentation. Complete in-place tensor edits, arbitrary graph regions and the
proposed tool APIs are new work. Adapt the existing independent checkers to the
actual candidate; do not recompile imported weights through a smaller language.
