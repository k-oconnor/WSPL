# Kinetix replay preparation record

November 4, 2026. Ready for exact-source approval; not registered or launched.

Manifest SHA-256: `ef3ca419ffae13e156be92a3eb2d220762c035cc2eb00839e4d8c3c8d0b0046a`.
Model ID: `kinetix-replay-v1`. Isolated checkout:
`C:/Users/kevin/OneDrive/Desktop/Skillforge-kinetix-replay-v1`, snapshot over
`9eda86877acff953725149041f33d62a44b31b31` plus the inventoried preparation files.
Source inventory: 99 Python files. Resources: 11,287 files including external
runtime Python/native libraries, released references and protocol. Local runtime
identity: `results/kinetix_preparation_v1/runtime_identity.json`; isolated audit:
`results/kinetix_preparation_v1/isolated_preflight.json`.

Development runtime venv-sim: 433 checks, 431 passed/two environment-specific
skips. Isolated WSL: seven Kinetix checks passed, including native three-tick
fixture; pip check clean. Earlier suite attempts under incomplete Python
installations exposed missing jsonschema/Z3 dependencies; those logs are retained.
No source changes were made to bypass those failures. Read the protocol for
float64 storage/inference, signed-zero reconstruction and RNG limitations.

Prepared run: 1,024 episodes each, parent/released repair, CPU only, fixed paired
seeds, 256 steps maximum, two-hour guard. No APIs, optimizer or GPU. Save both
actors, every episode and first three complete trajectories per policy.
Expected duration remains an estimate (tens of minutes, with the two-hour guard);
no scored timing pilot was performed. No automatic subsequent authoring.

Only after exact approval: register the digest in the isolated and main approved
registry, reverify source/resources, then run the approved-source launcher in
WSL with JAX_PLATFORMS=cpu and OPENBLAS/OMP/MKL_NUM_THREADS=1. Use hidden process
launch, persistent logs and results/kinetix_replay_v1. Do not modify the isolated
source or bound files after launch. Inspect study status independently of the
source receipt after a controlled exit.
