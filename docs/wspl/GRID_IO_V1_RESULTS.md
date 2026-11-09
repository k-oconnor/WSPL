# Grid I/O prerequisite: failed

Run and source receipt complete in166.391seconds (2.77minutes). All384 planned
frozen-model generations saved. No adapter training or author calls occurred.

| Task | Unconstrained exact /64 | Palette-only exact /64 | Palette cell accuracy |
|---|---:|---:|---:|
| Copy | 0 | 48 | 96.19% |
| Familiar-shape movement | 0 | 2 | 80.96% |
| Novel-L movement | 0 | 0 | 74.02% |

The predeclared copy requirement was61/64 exact with palette-only decoding;48/64
fails. Cell accuracy does not override exact-grid failure. Accordingly no movement
adapter training or Astra authoring follows automatically.

Unconstrained failures must be interpreted carefully: the model emits a leading
space, while the protocol requires exactly16 color tokens with no extra token.
The space consumes one position under the fixed16-token budget. This strict-format
result is not evidence of zero underlying grid ability. No post-hoc trimming or
extra-token regeneration was used to rescore it. Counts of leading-space starts
are preserved in the companion JSON. Palette-only decoding removes this syntax
issue and still fails the copy gate, so formatting alone does not explain failure.

9 of sixteen palette-copy errors have15/16 cells correct, with the remainder
more substantial. Copy exactness by marker direction:down5/16,left15/16,right16/16,
up12/16. These are descriptive correlated fixtures, not an independently designed
position-bias study. Movement exactness remains poor. This is an interface/base
capability limitation in the chosen frozen SmolLM setup, not evidence against the
ReLU adapter: the adapter was not deployed or trained in this run.

Archived source and all saved exact-grid scores independently verified read-only.
No new model calls or evaluations in audit. Preserve protocol/results unchanged.
Next decision: establish reliable copying before testing movement acquisition,
through a separately declared model/interface preparation or a different base.
No new scientific run is prepared or launched by this report.
