# Copy foundation: prerequisite passed

The run and source receipt completed successfully in366.875seconds (6.11minutes).
512 updates used2,048 distinct copy examples,32,768 answer targets. The final
training update was recorded at97.125seconds. No movement labels or Astra calls.

| Copy panel | Palette-only exact | Unrestricted exact |
|---|---:|---:|
| Fresh random holdout | 256/256 | 256/256 |
| Previously observed domain copy panel | 64/64 | 64/64 |

Both predeclared gates pass (244/256 and61/64 palette-only). Unrestricted
generation also produces every grid correctly, without the palette constraint.
The historical panel improves from48/64 palette-only and0/64 strict unrestricted
before fine-tuning. The prior unrestricted failure included a leading-space
format mismatch; that historical comparison is not purely a copying comparison.

This establishes copy transfer to unseen inputs under this fixed codec and
eight-demonstration prompt. It does not establish movement acquisition, ARC-AGI
performance or general-language retention. Full-model copy fine-tuning changes
the experimental foundation. Freeze this exact foundation and share it across
all subsequent movement controls and adapter arms.

Read-only audit verified source identity, all640 saved generations and their
exact scores, train/holdout disjointness,512 training records,2,048 distinct
selected examples, checkpoint source identity and exact checkpoint tensor hash
matching the evaluated model. No new inference or training during the audit.
Peak allocated GPU memory5,438,509,568bytes (5.07GiB). One last.pt checkpoint,
including optimizer state, is1,614,487,862bytes; no duplicate final checkpoint.

Next: separately prepare gradient-trained movement-adapter reachability against
this frozen foundation, with unchanged-foundation movement and copy-retention
controls. Copy training could itself strengthen a copy bias, so movement ability
must be measured again on this foundation. No follow-on run has launched.

Evidence: grid_copy_v1_results_summary.json; isolated output
Skillforge-grid-copy-v1/results/grid_copy_v1.
