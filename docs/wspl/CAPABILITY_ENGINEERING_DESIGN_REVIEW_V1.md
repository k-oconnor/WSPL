# Capability engineering: Astra design review

Status: conceptual review completed. No training, benchmark evaluation, external
authoring collector call, experiment pin or scientific launch in this review.
The user explicitly requested Astra-led reconsideration of the experimental
setting. An Astra/xhigh reviewer inspected local evidence read-only. This is
collaborative experimental design, not a blinded authoring result.

## Research question

What must an LLM engineer to install a reusable new capability in another model,
and does WSPL improve that process? Weight modification alone, a small net number
of corrected examples, or lower training loss is not capability acquisition.
Success requires substantial transfer, retention, a fixed deployed artifact and
explicit accounting of training and inference computation.

Evidence reviewed: GRID_COPY_V1_RESULTS.md, GRID_MOVEMENT_V1_RESULTS.md,
GRID_REPAIR_V1_RESULTS.md, GRID_REPAIR_INTERFACE_V1.md, WSP_GRAPH_SURGERY.md and
WSP_SYNTAX_0_2.md. Copying succeeded. Movement and repair did not meet their
criteria. The observed failures do not identify whether representation, adapter
placement/capacity, optimization or data distribution is responsible.

## Astra's recommended change

Replace repair of the late rank32 ReLU adapter with fresh capability acquisition
from the frozen copy-trained foundation. Default mechanism: rank16 low-rank
branches on attention query/key/value/output projections across all30 layers.
These let learning change information routing as well as representations.
This is a design hypothesis, not demonstrated superiority to the late adapter.

Astra recommends direct autoregressive outputs for this single-transformation
assay, without a scratchpad. This time it is an explicit reviewer design choice,
not an inherited requirement that all capability engineering must avoid reasoning.
Inference deploys only the model and declared neural modules, not a grid solver.

Proposed next domain: generated6x6 object translation, with whole shape families
held out after canonicalizing rotations/reflections, larger-object transfer and
whole-input exclusions. Prior panels become development evidence only. Keep
fresh4x4 copying as a retention measure. This is a bounded synthetic capability
assay, not ARC-AGI-1, general reasoning or a causal-world-model result.

## Proposed single bounded screen

Five trained arms: one fixed ordinary-gradient recipe; two independent native
Astra authors; two independent native-plus-WSPL authors. Both author groups have
the same Python/PyTorch, diagnostics, training and candidate resource capabilities.
Within a shared capacity/operator envelope, authors may change placement, rank
allocation, initialization, losses and curriculum. Freeze the envelope before
authoring rather than letting one interface receive more expressive power.

Reviewer-proposed, not approved, budgets per arm:16,384 training examples,
1,024 development examples, at most8,192 updates or60 active GPU minutes,
whichever comes first. Four responses per author. All attempts/restarts must
count against the same per-arm budget. Five hours is a training GPU ceiling,
not a measured total-runtime estimate; evaluation and authoring add time.

Proposed descriptive success criteria after all artifacts freeze:

- Familiar-shape exact transformations:487/512 or better.
- Unseen canonical shape families:461/512 or better.
- Larger-object transfer:461/512 or better, separately.
- Retained copying:507/512 or better and at most five paired regressions against
  the unchanged foundation.
- Two-call compositions:231/256 or better, using the same frozen model twice.

These are proposed engineering thresholds, not power-derived statistical tests.
The unchanged foundation receives identical prompts and inference budgets.
Compare with the ordinary-gradient recipe to separate successful learning from
value contributed by LLM-directed design. Two author pairs are only descriptive.

## WSPL's role and implementation boundary

Native versus native-plus-WSPL is the credible comparison. Do not handicap native
tools or prohibit training to create an apparent language advantage. Record actual
WSPL use; merely having the language available is not evidence it helped.

Current stateless WSPL DAG operations can express affine low-rank branches,
residual additions and explicit tensor edits. The language does not currently
represent full transformer attention, recurrent state, arbitrary framework imports
or automatic placement. A shared host bridge must attach named modules to model
projections, preserve sharing, expose training and export frozen artifacts.
Existing Core dense artifacts have limited Torch emission; a general trainable
graph-to-framework integration must be implemented and checked, not assumed.

Use identical deployed semantics across interfaces. Verify branch-level shape,
identity, round-trip, deployment fidelity and declared contracts separately from
end-to-end task performance. A module certificate is not proof of semantic ability.
Measure tokens, calls, optimizer targets, GPU time, construction failures, debugging
effort, retained capability and transfer. Describe numerical training honestly as
LLM-directed neural engineering, not gradient-free weight invention.

## Coordination review: unresolved items before preparing a source pin

1. The copy-trained model has demonstrated4x4 copying, not6x6 copying. Changing
   grid size creates a new I/O uncertainty. Include6x6 copying diagnostics and an
   explicitly budgeted copy-acquisition curriculum in the common design; do not
   repeat the assumption that a representation is already usable. Any shared
   prerequisite training must be identical for every arm and fully reported.
2. Absolute accuracy alone is insufficient. Predeclare a substantial paired gain
   over the unchanged model with the same prompt/budget; if the unchanged model
   already solves the task, it cannot demonstrate newly installed capability.
3. Define token/sequence limits, batch size, precision, parameter cap and allowed
   module types before launch. The reviewer budgets are not yet a runnable spec.
4. Freeze composition serialization: the bridge may parse/re-encode the model's
   grid output but must not calculate movement, repair invalid outputs, choose a
   successful retry or reveal target-derived information. Generate examples where
   both requested moves remain valid. Count both model calls in inference cost.
5. Choose fresh train/development/test seeds and canonical shape exclusions before
   authoring. Keep final predictions inaccessible until all candidates freeze;
   count every development interaction and report all arms.
6. A negative native-plus-WSPL result may reflect little actual language use. Log
   what operations it supplies and distinguish syntax/debugging/verification value
   from the value of a more expressive architecture or extra training.

No new experiment is approved by this document. Next implementation should make
this shared design concrete and verify its engineering interfaces, then request
one exact source approval for the bounded screen. Avoid another long sequence of
small repair pilots that never addresses domain-level acquisition.
