# Scientific run identities

approach_v1 was user-approved and launched; it failed before fitting on native-horizon bookkeeping. Preserve its manifest/spec, approval and output. approach_v1_horizonfix.spec.json and .manifest.json prepare the minimal replacement; it is unapproved and not queued. See docs/APPROACH_V1_LAUNCH.md. Never transfer the old approval to the replacement digest. Both identities include project Python and external package/asset inventories.

After implementing a concrete constructor and runner, prepare a spec naming their module/source/symbol, constructor arguments, full runner configuration and Python source roots. The runner must accept constructor, output_dir and config, instantiate the supplied constructor, and save the approved manifest digest in checkpoints. Include the gate itself, all local model/runner/dependency code and tests in the declared inventory.

```sh
python experiments/source_consistency.py prepare --spec experiments/source_pins/example.spec.json --output experiments/source_pins/example.manifest.json
```

This creates an unapproved candidate, not permission. Review/test the complete implementation and ask Kevin to approve that exact SHA-256. Register only actual user approvals. Then launch only through:

```sh
python experiments/source_consistency.py run --model-id APPROVED_ID --output results/NEW_OUTPUT
```

The commands above are templates, not existing runnable experiments. Never refresh an approved historical identity to hide changes, auto-approve a candidate or bypass the gate. Unit checks and read-only audits do not require experiment approval.
