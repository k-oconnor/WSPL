# Completed direct versus WSPL repair screen

November 4, 2026. **Both arms succeeded on all six proposals.** The prespecified
decision is `feasible_but_ceiling_no_advantage_established`. The task panel is too
easy to distinguish these interfaces on repair success; the broader structured
surgery operations were not exercised by the proposer.

| Task | Direct editing | WSPL available | Structured feature actually used |
| --- | --- | --- | --- |
| Shared gain repair | 2/2 | 2/2 | `untie` in both WSPL proposals |
| Clipped drive | 2/2 | 2/2 | None; both WSPL proposals chose raw graph edits |
| Isolated deadband | 2/2 | 2/2 | None; both WSPL proposals chose raw graph edits |
| Total | **6/6** | **6/6** | 2/6 WSPL proposals used structured surgery |

All twelve replies were valid on their first submission. All 24 external repair
and preservation claims were PROVED over the specified real input box. All 249
FP32 probe inputs per candidate passed, with zero maximum observed error on that
panel: 2,988 candidate-input evaluations. No collection failures, harness retries,
tool use, training or scored environment episodes. Twelve fresh threads and one
completed turn/message per thread were verified. The question tool remained
advertised by the CLI but was never used.

## What was learned

The complete proposal-to-artifact-to-check workflow works on these supplied
circuits. Models can introduce nonlinear computation, remove cross-channel
dependence, and separate shared weights while preserving the other output.
Both arms have adequate expressivity here; no scalar allocator blocked authoring.

The two WSPL shared-gain replies use the same short edit:

```wspl
untie parameter("n0.weight") as W_drive;
W_drive[0,0]=3;
```

All four shared-gain candidates, across both arms and draws, have identical
artifact identities. Thus the convenience operation and direct tensor/node edit
produced the same actual stored network. This supports the implemented operation's
usefulness as concise authoring syntax on this example; it does not establish
human-time savings.

Four WSPL-assigned proposals submitted only the common raw JSON path. No proposer
used region replacement, transplant, residual insertion, or authored module
bundles. This was permitted prospectively. The comparison measures **access to**
structured WSPL alongside raw access, not mandatory use of its features. It does
not establish that general graph surgery improves repair. The reason for choosing
raw edits is not measured; low complexity or interface overhead are hypotheses.

## Usage and timing

| Metric | Direct | WSPL available |
| --- | ---: | ---: |
| Reported input tokens | 20,598 | 20,616 |
| Cached input tokens | 0 | 0 |
| Reported output tokens, including reasoning | 5,188 | 5,051 |
| Reasoning tokens, included above | 3,346 | 3,318 |
| Collection time across six calls | 214.72 s | 187.98 s |
| Construction time | 0.016 s | 0.033 s |
| Evaluation time | 0.259 s | 0.245 s |

Total reported usage: 41,214 input and 10,239 output tokens, including 6,664
reasoning tokens. WSPL output usage was 137 tokens (2.64%) lower overall, a small
descriptive difference across six proposals. It is not reliable efficiency
evidence. Collection latency includes service variability; both arms received
the same full interface packet. Human setup/review effort and hosted compute or
dollar cost were not measured. No API-key fallback was used.

Study active time was **405 seconds (6m45s)**, substantially below the conservative
preparation estimate. The source-gated process ran from 00:11:28 to 00:18:15 PST
(08:11:28–08:18:15 UTC). Final models contain 3 parameters for shared gain,
9 for clipping, and 8 for deadband. These are tiny constructed circuits, not
learned controllers or the archived 20,486-parameter integration fixtures.

## Audit and provenance

Approved source SHA-256:
`807c27b5bc0d769969a6bae14dda809fc68b4b185cccfad704b491658e2a28fe`.
Run and source receipt both complete; 12 checked constructor calls. Isolated
checkout: `Skillforge-wsp-repair-v1`, output `results/wsp_repair_v1`, frozen source
snapshot over base commit `29eb1cd`. The base commit alone is not the source identity.

[Completion audit and trial data](wsp_repair_v1_results.json) records:

- All 90 inventoried Python files, 14 auxiliary bindings, task/prompt identities,
  numeric/solver dependencies and CLI executable reverified unchanged.
- All twelve reply/event/usage records matched; exact prompt previews and explicit
  Astra/xhigh commands checked. An auxiliary CLI metadata warning is preserved in
  stderr and is not used as evidence of proposer identity.
- All replies replay to exactly the saved candidate and construction receipt.
- All formal/numeric evaluations reproduced, including the external description
  identities and statuses. A separate recursive rational evaluator also agrees
  with the requested functions on 81 fresh diagnostic points per candidate. That
  post-run diagnostic is not substituted for the original primary score.
- Trial denominators, paired zero success differences, usage totals and the
  prespecified ceiling decision recomputed. Raw evidence hashes are retained.

Reproducer: `tools/audit_wsp_repair_v1.py --root <isolated-checkout> --output <new-json>`.
The audit is read-only with respect to the scientific run and made no model calls.
[Byte-preserved replies](../../artifacts/wsp_repair_v1/index.json) are archived
alongside the code; full collection logs, plans and deployed candidates remain
in the isolated output. The frozen protocol and all prior experimental archives
are unchanged.

## Next decision

Proceed to designing a harder task panel, rather than claiming a WSPL advantage
or simply repeating these tasks. Useful pressure points are repairs involving
multiple consumers, shared parameters, reused submodules and several preservation
obligations on a larger existing graph. Keep expressive raw graph/code access
and matched budgets, and record spontaneous use of abstractions. A required-use
mechanism test would answer a separate question and must be labeled accordingly.

No follow-on experiment is prepared or approved by this completion. These results
do not demonstrate learned-policy repair, general causal discovery, robotics,
universal IEEE execution proof or recursive improvement.
