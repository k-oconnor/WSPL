# Diffusion coordinate-edit comparison: completed

The six-proposal screen completed in 242.25 seconds (4.04 minutes), at
2026-11-04 20:57:28 UTC. Both source receipt and study report complete.

| Outcome | Python/NumPy | WSPL |
|---|---:|---:|
| Successful generic programs | 3/3 | 3/3 |
| Successful parent-model applications | 9/9 | 9/9 |
| Exact real-graph permutation certificates | 9/9 | 9/9 |
| Input tokens | 5,907 | 5,916 |
| Reported output tokens | 3,260 | 2,710 |
| Of those: reasoning output tokens | 2,488 | 1,892 |
| Output minus reported reasoning | 772 | 818 |
| Total local construction time, all three programs/parents | 9.139 s | 54.797 s |

For each parent, every proposal produced bit-identical FP32 weights. Maximum
functional-probe error was 9.54e-7; worst coupled-sampling error was 8.37e-6,
well below the predeclared .001 tolerance. The structural witness establishes
the real-arithmetic coordinate identity; it is not a universal IEEE proof.

All WSPL programs used bounded loops, slices and immutable parent tensor reads,
with exact-to-storage arithmetic. None used named circuit placement or topology
surgery. Python used NumPy index permutations. Each correctly transformed both
the first-layer pixel inputs and final-layer output weights/biases while leaving
time channels and intermediate computation unchanged.

WSPL reported 16.9% fewer output tokens, driven by fewer reasoning tokens.
Non-reasoning output was 818 versus 772 (6.0% more). Total input plus output was
8,626 versus 9,167 (5.9% fewer). Three draws on one generic algebraic task cannot
establish an efficiency advantage. Local WSPL construction was about six times
slower; the implementations perform different checks and arithmetic, so this is
observed end-to-end construction cost, not a controlled runtime microbenchmark.

Read-only audit verified 116 source files, 1,366 bound Windows resources and
1,579 Linux runtime resources, six fresh single-turn/no-tool collections,
normalized prompt transport hashes/byte counts, extracted sources, 18 saved
candidate hashes, all archived parent weights, independent inverse-index
reconstruction of the structural identity, and all 1,152 stored paired samples.
No new authoring, submitted-code execution or generation during the audit.
Auditor: tools/audit_diffusion_edit_v1.py. Evidence: diffusion_edit_v1_audit.json.
Full archive: sibling Skillforge-diffusion-edit-v1/results/diffusion_edit_v1.

Interpretation: a clean feasibility ceiling. Astra can express a correct generic
weight transformation through either interface, transferring unchanged to all
three denoisers. This is not semantic feature discovery or general graph surgery.
The three parent models are transfer checks, not additional author replicates.
A harder task should require internal computation changes or module composition;
repeating this coordinate-permutation task is unlikely to distinguish interfaces.

Approved source: 494c130f55b69d0705885e15ea5c3af4a49fe31883c4319ac870e8dec939a4ac.
