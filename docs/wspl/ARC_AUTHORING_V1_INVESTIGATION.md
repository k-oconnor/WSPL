# ARC authoring investigation — November 5, 2026

Read-only analysis of the completed nine-session screen. No new model calls,
forward predictions, training or benchmark evaluation. Original pin and archive
remain unchanged. Machine-readable accounting: arc_authoring_v1_investigation.json.

## Findings from programs and saved records

1. **The proposals were generic weight transformations.** Native sessions tried
   scaling the four late attention output matrices up/down and modifying their
   singular spectrum. Direct WSPL sessions tried scalar rescaling, then restored
   the original matrices. LoRA sessions used fixed channel-group averages or
   directions derived from selected weight rows, then changed their magnitude,
   sign or affected layers. None constructed an identifiable ARC-specific feature.
   This last statement describes source structure, not a proof about latent semantics.

2. **No session used INSPECT.** All36 responses were programs. The32 supplied
   training examples and16 detailed validation records supplied task information,
   but authors had no activations, feature attribution, gradients or model-forward
   probes. Weight access alone did not expose how to correct a particular error.
   Each inspection would have consumed one of only four responses.

3. **Edits were applied and often substantial.** There were32 accepted proposals,
   including seven restorations to the original model, leaving25 accepted nonzero
   proposals. Relative matrix update norms often reached6–10%, so this was not
   merely numerical noise or universally tiny edits. No proposal was rejected
   for exceeding the norm bound. Saved likelihoods changed; restored candidates
   returned exactly to base weights.

4. **Almost no wrong validation answers became right.** Across the25 nonzero
   proposals, just one proposal fixed one baseline error: native draw2 step2,
   a dominant-singular-mode attenuation. It also broke two previously correct
   answers. Every other accepted nonzero proposal fixed zero baseline errors.
   These counts compare each checkpoint with the original baseline, not its
   immediate predecessor. They are repeated observations on the same299 examples.

5. **Syntax friction was real but cannot explain the whole failure.** All three
   direct WSPL sessions and one LoRA session first used bare whole-tensor assignment
   (e.g. `W26 = parent.P26 * ...`). The parser required a subscript. Four of36
   response slots were lost, all in WSPL arms. Native had no such failures and
   still found no net improvement. The interface packet should have supplied
   explicit, executable assignment examples; accepting full-tensor shorthand
   is also a reasonable ergonomic repair.

6. **Final retention differences were tiny.** Seven sessions restored the model.
   Each remaining LoRA fixed one Challenge test example and broke three, net -2
   of1172. On Easy test, one fixed7/broke9 and the other fixed9/broke7, net -2/+2
   of2376. These are not compelling adaptation gains. No intermediate test scores
   were exposed during authoring.

## Engineering repair, distinct from the archived result

Development WSPL now accepts a declared tensor name on the left side under
`exact_to_storage`, lowering it to a full slice. This is shape-preserving write
syntax, not arbitrary variable rebinding. Unknown names, parent writes and
invalid shapes still fail. The historical `native_numpy_v1` literal profile is
unchanged. Full-tensor/explicit-slice comparisons cover bit identity, signed-zero
copies, augmented snapshots, touched indices and resource accounting. All251
WSPL-focused tests pass, including three new regression tests. No study was rerun
and no rejected historical proposal was relabelled as successful.

## Interpretation and next decision

The strongest supported conclusion is that **this information-limited, four-step,
late-attention editing protocol did not find useful adaptation**. It does not
establish that LLM-authored adapters cannot work. It also does not establish that
more author tokens or a larger weight budget would help.

Recommended next diagnostic: establish reachability with a gradient-trained
rank8 adapter on exactly these same matrices, the same norm restriction and
training-only examples, then evaluate validation and retention. This is a new,
explicit gradient control—not a replacement for the native authoring arm, and
not a gradient-free result. If this scope cannot learn, reconsider target layers
before spending more author calls. If it can, compare native/WSPL authors with
matched, train-only activation or error-localization diagnostics. Any supplied
gradients must be reported as gradient-assisted authoring, not silently added.

The contrast with Go is concrete: Go inputs had explicit board coordinates and
occupancy semantics, allowing spatial algorithms to be written directly. These
SmolLM edits address learned latent matrices without such a mapping. That makes
grounding the edit directions a plausible bottleneck, but it remains a hypothesis
until the diagnostic-access and reachable-scope experiments are run.

No follow-on benchmark or training has been prepared, approved or launched here.
