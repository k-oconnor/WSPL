# WSPL 0.2 syntax alignment

Design syntax with a [first implemented subset](WSP_EDITOR_FIRST_PASS.md) and
[implemented graph extension](WSP_GRAPH_SURGERY.md). Read those implementation
pages for supported operators and current limits, and the
[tooling design](WSP_TOOLING_DESIGN.md) for the larger intended interface.
Historical Core PWA modules continue to use their existing parser and meaning.

## Source forms

| Form | Purpose | Runs during inference? |
| --- | --- | --- |
| `target` | Declare or reference an exact network | Declares the executed artifact |
| `const` | Bind construction data or a scalar | No |
| `circuit` | Define reusable acyclic signal computation | Only its placed neural implementation |
| `edit` | Ordered storage changes or explicit graph changes | No |
| `place` | Map circuit nodes onto specified existing producers | Resulting producers execute |
| `region` | Identify a complete operator subgraph and boundary | No new computation |
| `describe` | Attach views, hypotheses and mathematical claims | No |

One plan has one edit transaction and one parent. Several candidates require
several plans; a description-only plan has no edit. Circuit and constant names
are immutable. An edit's `as` name denotes its resulting candidate. No annotation
can assign tensors, change a graph, or alter an externally required property.

## Minimal grammar for the first implementation unit

This initial grammar covers raw edits and explicit placement. The subsequent
implemented `region`/graph-edit grammar is documented with runnable syntax in
[graph surgery](WSP_GRAPH_SURGERY.md). The new editor has its own claim parser
with scalar indexing and internal taps; the old Core PWA contract parser keeps
its original meaning. General quantifiers remain outside the implemented subset.

```ebnf
program       = "wspl" "0.2" ";", { declaration }, [ edit ], { description } ;
declaration   = target | constant | circuit ;
target        = "target", name, "=", target_constructor, ";" ;
constant      = "const", name, "=", construction_expr, ";" ;
circuit       = "circuit", name, "(", input_ports, ")", "->",
                "(", output_ports, ")", "{", { signal_assignment }, "}" ;
port          = name, ":", type ;
type          = "Real" | "Real", "[", constant_expr, ",", constant_expr, "]"
              | "Truth" ;
edit          = "edit", name, "as", name, "mode=fixed",
                [ "numeric=", numeric_profile ], "{", { edit_statement }, "}" ;
edit_statement = tensor_write | placement | construction_loop ;
tensor_write  = tensor_selection, ( "=" | "+=" | "-=" | "*=" ),
                construction_expr, ";" ;
construction_loop = "for", name, "in", "range", "(", constant_arguments, ")",
                    "{", { edit_statement }, "}" ;
placement     = "place", name, "(", named_signal_arguments, ")", "at", "{",
                { name, "=", producer_selection, ";" }, "}",
                "writes=replace_producers", ";" ;
signal_assignment = name, [ ":", type ], "=", signal_expr, ";" ;
```

`input_ports` and `output_ports` are comma-separated ports. Constructor and
expression names are whitelists, not arbitrary calls. This structural grammar is
paired with the lexical/operator rules below; it is not a claim of a shipped
parser. The default new-program numeric profile is `exact_to_storage`.

### Names and delimiters

Names use ASCII letters/underscore followed by letters/digits/underscore; they
are case-sensitive. Keywords and target aliases cannot be shadowed. References
resolve lexically; no dynamic name evaluation. Semicolons terminate simple
statements; braces delimit blocks; newlines are whitespace. `//` and `#` introduce
line comments outside quoted strings. Quoted strings use JSON escaping.

Duplicate target, circuit, constant, local or output declarations fail. A circuit
assigns every output exactly once; its intermediate signals are single-assignment
and acyclic. Each `place` creates a fresh circuit instance. Every computational
node/output must have one compatible placement; different circuit nodes cannot
silently alias the same target producer.

### Three expression contexts

**Constants and construction expressions** operate on numbers and finite tensors.
Their operators are `+ - * / @`, parentheses, lists, tensor reads and approved
construction functions. Multiplication is elementwise, `@` is rank-two matrix
multiplication, scalar arithmetic broadcasts, and the remaining broadcasting
uses trailing-axis compatibility. Division by zero, ragged lists and incompatible
shapes fail. `copy`, `transpose`, `reshape`, `zeros`, `full`, `eye` are explicit
builtins; external `tensor("name", sha256="...")` resolves a pinned binding.
No activations, runtime inputs or stochastic functions are visible here.

**Signal expressions** operate on inference values inside a circuit or at its
input bindings. The initial profile supports affine combinations, ReLU, clip,
min/max and named earlier signals. This intentionally reuses the useful Core PWA
operations. Tanh is permitted as a target activation for direct tensor editing;
support for placing source-defined tanh nodes requires a separately implemented
signal operator. A missing compiler operator must not block literal tensor edits.

**Claim expressions** operate on frozen artifact inputs, outputs and named taps.
They never modify inference or weight construction. Exact equation claims can
reuse supported signal operations; Boolean connectives/affine comparisons have
their own grammar and checker profile. Domain assumptions are recorded explicitly.
Unsupported claims are retained as unsupported rather than misparsed as executable
programs. Ordinary prose is a `hypothesis`, never a theorem by syntax alone.

For both numerical expression contexts, postfix indexing binds tightest, then
unary signs, then `* / @` left-to-right, then `+ -` left-to-right. Parentheses
override precedence. No implicit multiplication or chained assignment.

### Selections and placement

`W1[3,0]`, `W1[:,0:4]`, `W1[3:600:60,0]` and `b1[-1]` are storage selections.
Integer selection removes an axis; slice selection retains it. Each axis is
explicit. Advanced indexing/masks are deferred; all cells remain addressable
through basic indexing and construction loops. Negative strides are allowed;
zero strides fail. The plan lists empty/clipped selections and actual indices.

`h1[0]` and `y[0]` are producer selections for placement; `pre(h1[0])` is an
inspection/claim tap, not an independent writable producer. Placement at a hidden
unit must match its existing activation; binding a source output to a tap cannot
override the target's activation. Vector placement requires an explicit shape
mapping and is a later extension; the first circuit-placement profile is scalar.

Producer replacement clears the full selected fan-in and bias, then writes the
new coefficients. Other outgoing consumers persist and are reported. Two placements
that write the same producer in one transaction are an error unless explicitly
marked as an override in a future extension. An explicit raw assignment following
a placement is legal and has ordinary sequential meaning; the final description
of the placed circuit becomes stale if that assignment changes its computation.

### Domain and numerical behavior

`Real` carries no finite bound. The first type profile is dimensionless; physical
units and shaped circuit ports require explicit extensions. Existing physical
Core PWA modules keep their old unit rules. `Truth` refines a value to [0,1]; it
is neither a probability interpretation nor an automatic Boolean observation.

Inferred bounds are facts only when justified by declared domains and implemented
analysis. User refinements create obligations. Unverified refinements cannot be
used to remove a relay, satisfy an interface contract or prove another claim.

`exact_to_storage` and `native_numpy_v1` are separate construction semantics.
The legacy importer records its selected profile explicitly. A bit-exact literal
has to match its destination dtype; casting it requires an explicit conversion.
Source values, rounded values and changed bits remain available in the receipt.
In the first implementation, arithmetic on a bit literal uses its exact rational
value; assignment/copy preserves its stored bits, including signed zero.

## Example editor session

The following are proposed request shapes, with documentation placeholders:

```json
{
  "tool": "wsp.plan",
  "parent_id": "sha256:<candidate identity>",
  "source": "<versioned WSPL text>",
  "profile_id": "sha256:<host construction and target profile>",
  "bindings": {"U": "sha256:<tensor U>", "V": "sha256:<tensor V>"}
}
```

A successful plan returns an immutable `plan_id`, exact resolved writes,
`parent_id`, proposed `candidate_id`, resource counts, numeric profile and
diagnostics. Applying it uses `plan_id` and `expected_parent_id`. If the active
selection has advanced, application can still create an independent branch from
the pinned immutable parent, but cannot update that active selection without
matching its expected version. Applying a plan never selects a new active model
implicitly. A separate admission action is governed by caller policy.

`wsp.check` takes `candidate_id`, an optional `parent_id`, an independent
`requirements_id`, the requested semantic profile and a resource budget. Its
response contains one result per obligation with dependencies, assumptions,
solver status, witness/replay details and scope. It does not accept a new
candidate-provided requirement set as a replacement for the independently pinned
one. Exploratory description claims may be checked in an explicitly separate set.

This separation lets an editor construct freely, inspect the actual changes,
attach proposed meaning, and learn from precise failures without conflating
construction validity with task success.
