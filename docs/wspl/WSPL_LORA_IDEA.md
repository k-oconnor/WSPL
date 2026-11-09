# WSPL-LoRA: LLM-authored adapter edits for specific skills

Status: user-proposed research idea; logged November 4, 2026 (America/Los_Angeles).
No implementation, experimental approval, demonstrated result or novelty claim.

Kevin's proposal: **"WSPL-LoRA... LLMs making wspl edits to lora adaptors for specific skills."**

## Core proposal

Use WSPL to let an LLM construct or revise low-rank adapter weights for a
specified skill while keeping the base network frozen. The output is a
versioned, executable adapter edit with a reproducible parent identity and
measured before/after behavior. Begin with an existing adapter; construction
from zero is a separate, harder hypothesis.

For a declared linear-layer convention, expose the adapter factors A and B
and scale s, with effective update delta-W = s B A. Fix target layers, shapes,
rank and scaling in the task packet. Give the editor bounded access to those
factors through WSPL transactions; prohibit changes to base weights. Whether
useful edits can be authored from the supplied information is the research
question, not an assumed capability.

## Hypotheses worth testing

- LLM-authored WSPL edits can improve a specific skill in an existing adapter.
- Restricted adapter edits can retain unrelated behavior while improving the
  target skill, as measured rather than presumed from freezing the base.
- Successful and unsuccessful edit trajectories can train a better editor,
  with transfer to held-out tasks, adapters and eventually base models.
- Separately authored skill adapters may be reusable components. Composition,
  routing and interference need their own experiments; a LoRA is not inherently
  an isolated or interpretable skill module.

## Suggested first experiment — not approved

Choose a small open-weight base model and one existing adapter with an objectively
scored, bounded skill, such as a constrained output transformation with exact
answers. Freeze the base, tokenizer, adapter rank, target layers and task splits.
Reserve hidden final examples and unrelated retention tasks before editing.

Compare unchanged adapter, bounded random edits, gradient-based adapter
continuation, Python/NumPy LLM edits, and WSPL LLM edits. Match the information
and feedback opportunities of the two authoring interfaces. Report gradient
data/updates and authoring tokens/calls separately instead of calling them
compute-matched. Start with independent proposals; feedback-driven revision
should be a declared additional condition.

Measure target-task improvement, retained skills, edit validity, effective
delta-W magnitude, adapter storage, source length, authoring usage, construction
time and execution cost. Evaluate every proposal, retaining failures and costs.
Hash frozen base tensors and untouched adapter tensors before and after.

Key design choice: what information does the proposer receive about the parent
adapter and its failures? Supply identical information to both interfaces and
record whether edits are informed by examples, activations, gradients or only
weights/specification. Do not imply task-only synthesis when stronger signals
were supplied.

## Interpretation and verification boundaries

Shapes, allowed edit scope, finite weights and unchanged base identity are
engineering checks. They do not certify a skill, retention or semantic locality.
Adapter factor coordinates are not automatically grounded concepts. Different
factorizations can represent the same effective update; evaluate effective
weight changes and behavior, not factor magnitudes alone.

An adapter can affect many tasks even with the base frozen. Retention and
composition claims therefore require behavioral tests. Full-network guarantees
do not follow from a valid WSPL program or low-rank structure.

## Connection to the broader program

This extends WSPL from complete small controllers toward compact edits of larger
networks. Candidate training records pair the parent adapter, skill request,
WSPL transaction, resulting adapter, validity evidence and measured skill/
retention outcomes. That supports the proposed editor-improvement and RTI
direction without establishing recursive improvement today.

Next decision: select a tractable parent model/adapter and scoring task, then
prepare a concrete protocol and source pin. Do not launch from this idea record.
