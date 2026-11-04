# Graph surgery engineering fixtures

The parent contains two affine branches sharing `W = [[2]]`:
`left = 2*x + 1`, `right = 2*x`, `y = left + right`, `keep = right`.
The module in `patch.json` computes `3*x`. These tiny JSON artifacts contain
exact FP32 storage bits and explicit stateless DAGs. They are supplied known
answers, not learned controllers or model-authored experimental outputs.

Each source starts from the same pinned parent:

| Source | Change | Result |
| --- | --- | --- |
| `replace.wspl` | Replace the left branch; retain shared storage used on the right | `y = 5*x` |
| `residual.wspl` | Add half of the module's output at the public `y` edge | `y = (11/2)*x + 1` |
| `untie.wspl` | Clone only the left branch's weight, then set it to five | `y = 7*x + 1` |

All retain `keep = 2*x`. `budget.json` is the host's explicit topology/storage
budget. It is supplied separately from the source; a plan records its identity.
Run from the repository root with a **new** output directory:

```sh
python -m pip install -e ".[editor,verify]"
python tools/demo_wsp_graph.py --output work/wsp_graph_surgery_demo
```

The demo saves replayable plans, candidates, descriptions, dependency diffs,
real-graph proofs and independently replayed counterexamples for deliberately
false claims. See [scope and syntax](../../docs/wspl/WSP_GRAPH_SURGERY.md).
