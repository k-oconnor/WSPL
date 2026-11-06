# Movement adapter: partial learning, reachability gate failed

Run and source receipt completed in573.14seconds (9.55minutes). All1,024 updates
completed; final training update at85.797seconds. All168 movement training inputs
were sampled. Only the37,472 adapter parameters trained; runner verified frozen
foundation tensor identity and absence of base gradients.

| Panel | Frozen copy foundation | Trained adapter |
|---|---:|---:|
| Training diagnostic | 0/32 | 11/32 |
| Unseen familiar shapes | 0/64 | 8/64 |
| Unseen L shapes | 0/64 | 3/64 |
| Domain copying | 64/64 | 60/64 |
| Random copying | 64/64 | 63/64 |

Palette-only and unrestricted decoding produced identical grids in every case.
The movement gate required48/64 and at least16 improvements on each movement
panel. Both fail. Retention also fails because domain copying drops below61/64.
No Astra authoring or follow-on training has launched.

This is partial movement learning, not a successful reusable domain adapter.
Even the32 training-case diagnostic reaches only11 exact grids, so failure
cannot be attributed solely to unfamiliar shapes. Final minibatch teacher-forced
NLL0.05298 does not establish exact autoregressive performance; it is a mixed
movement/copy batch, not an aggregate evaluation loss.

The copy-trained foundation repeats the input on58/64 familiar and54/64 novel
movement prompts. The adapter repeats none of those inputs, and improves cell
accuracy from81.15% to86.13% familiar and75.20% to81.84% novel. These secondary
metrics show behavior changed but do not override exact-grid failures. High
cell accuracy is also influenced by unchanged background cells.

Read-only audit verified archived source, all1,152 saved outputs against their
fixtures and exact scores, identical decoding-mode outputs, complete update logs,
and checkpoint source/adapter tensor identity. No new inference during audit.
Evidence: grid_movement_v1_results_summary.json and isolated output
Skillforge-grid-movement-v1/results/grid_movement_v1.

Interpretation: this particular late-layer rank32 adapter and training budget
have not established reachability. This does not establish impossibility or an
Astra failure. Before authoring, investigate training fit and whether a stronger
gradient-trained interface can learn the movement transformation while retaining
copying. Any new scientific run requires a separate protocol and approved pin.
