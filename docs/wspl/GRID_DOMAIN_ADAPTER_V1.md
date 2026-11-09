# Domain adapter: supplied object movement

Objective: one reusable small ReLU adapter operating on native hidden states,
first trained by gradients to establish reachability, then authored by Astra with
logged diagnostics under the same interface. The target is a transformation domain,
not one edit per puzzle. No ASP or task-specific inference dispatcher is included.

The initial domain uses4x4 grids. Color0 is background; one connected monochrome
object has color1 or2; a single marker has color3. Move the object exactly one cell
toward the marker, preserving shape/color and the marker. The marker is outside
the object's bounding box and aligned with its row or column span, giving exactly
one cardinal direction. Cases with output collisions or out-of-bounds movement are
excluded by the preparation-time generator. The rule is supplied, not inferred.
This is an ARC-style generated domain, not an ARC-AGI-1 benchmark result.

The generator enumerates560 distinct inputs using single cells, two-cell bars and
four three-cell L orientations. Evaluation splits and demonstrations are disjoint
by whole input. Shape, position, direction and color vary. This finite domain is
a first feasibility fixture, not evidence of unrestricted spatial reasoning.

## Prepared adapter interface

At layer27 output, at every token:

`h' = h + B ReLU(Ah + a) + b`

Hidden width576, bottleneck32,37,472parameters. A starts random, B and biases zero;
the unchanged model is recovered exactly at initialization. The base model,
tokenizer and output head remain frozen. No external object detector, grid solver,
answer-ID router or symbolic runtime. The module is piecewise linear, but no
semantic certificate or grounding claim follows from that fact alone. Formal
module properties and end-to-end task correctness remain distinct.

Synthetic engineering checks establish insertion/restoration and gradient flow.
They do not establish learnability. No adapter training or authoring has occurred.

## Stage 0: input/output gate (the prepared scientific run)

Use the existing frozen SmolLM2-135M-Instruct to establish whether the chosen
representation is workable. Fixed eight demonstrations cover directions/colors.
Copy prompts show copied outputs; movement prompts show actual moved outputs.
The task instruction is explicit. Input/output grids are row-major16-digit strings.
No grid answer is parsed into objects during inference.

Evaluate64 copy inputs (32 simple objects,32 L shapes),64 familiar-shape movement
inputs and64 novel-L-shape movement inputs. Each condition is tested in two modes:

- Unconstrained: greedily choose the highest-probability vocabulary token for
  exactly16 output positions. Any non-palette token makes the output invalid;
  do not trim, repair, retry or extract a grid from prose.
- Palette-only: choose the highest-probability token among the four color digits
  at each of16 positions. This constrains syntax only; it has no knowledge of the
  target grid, object location, marker or requested displacement. It is an explicit
  output-format aid, so passing it is not unrestricted generation capability.

384 grid generations total,6,144 autoregressive output positions. Full-grid exact
match is primary, cell accuracy secondary. Invalid outputs score zero. Preserve
tokens, decoded text, per-case exact/cell scores and direction summaries.

The copy prerequisite is at least61/64 exact under palette-only decoding. Report
unconstrained copy separately. If this fails, resolve model/representation first;
do not train an adapter or buy authoring calls against an unusable interface.
If it passes, prepare a separate, source-approved gradient-trained positive control
with the same model and codec. No automatic training or Astra session follows.
Movement scores establish headroom, not an authoring result. Passing copy alone
does not prove that the later adapter can learn the movement transformation.

GPU FP32 inference only. No paid/model-author calls, gradients, fitting or downloads
in Stage0. Expected5–20minutes; two-hour guard, no automatic retry. Existing model
and runtime pinned. Source/model/resource hashes checked before/after. Output is
evaluation records only; no additional model checkpoints.

## Subsequent stages, not yet approved

After the gate, train only the prepared ReLU adapter on a generated construction
pool, with exact-grid transfer and copying-retention evaluation. Only if that
establishes reachability should Astra author a domain adapter, using aggregate
feedback, native diagnostics and the same capacity/interface. Reserve new inputs
before authoring; no per-example runtime editing. The learned adapter's weights
should not simply be supplied for copying. Compare data, queries and compute
explicitly, and distinguish analytic supervised construction from gradient-free
or zero-shot claims. Subsequent budgets and source pins require separate approval.
