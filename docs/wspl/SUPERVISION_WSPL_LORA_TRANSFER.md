# Superseded: original-machine transfer inventory

Status: superseded November 23 by Kevin's explicit instruction to engineer a
fresh local stack. Original artifacts are not available and are not required
for the new prerequisite in FRESH_ROBOT_PARENT_V1.md. The inventory below is
historical provenance only, not an outstanding request to the user.

Previous transfer plan: the current host has an RTX 3090,
Python 3.13 with CUDA PyTorch, and the Git source, but no prior Qwen checkpoints,
robot runtime, or WSL installation. Do not substitute new checkpoints or silently
retrain the old baselines.

An existing configured SSH host or network share is enough to inspect the original
files. Do not copy credentials into the repository. Prefer running on the intact
original runtime to rebuilding the environment unnecessarily.

Needed inputs (paths below are historical suffixes, not verified current paths):

- Base model/tokenizer: `skillforge-runtime/models/Qwen2.5-Coder-1.5B-Instruct`.
  Preserve the exact snapshot, config, tokenizer and weight shards; derive their
  resource hashes from the original approved manifest and verify them.
- Ordinary-practice parent:
  `WSPL-executive-teaching-repair-v2/results/executive_teaching_v2/rehearsal/adapter_final.pt`.
  The final author provenance records parent SHA-256
  `84f2148dd14d46eb50abe60832cad9881fa645f9586b9f9e39d1061f7d40af59`;
  verify its binding against the original run before adopting it.
- Final supervised adapter: under
  `results/astra_supervisor_resume_v1/supervisor/submissions/a5b1d71c2321cc10cc97819c8a522027ac9b04bace62ee49aa48579bcb240b83`.
  The archived `adapter.pt` SHA-256 is
  `d3a67a7d84c53d0041d8f6886709acac59723ca31b4c7bd784bd88962bc6c50c`.
- Frozen motor artifact:
  `WSPL-factory-v1/results/factory_pilot_v1/artifact_final.json`.
  Author provenance records motor SHA-256
  `0b98ed9c1f0296d2a9b15ad832ded7da7e758a686dd8a0343b386507032591e5`;
  verify the original runtime binding.
- Original recovery `approved_manifest.json`, `source_receipt.json`, run status,
  `frozen.json`, runtime manifest and deployment receipts. These locate and verify
  resources; their historical approval is not transferred to this new run.
- Existing isolated Linux Python/CUDA/MuJoCo/robosuite runtime, its assets,
  bubblewrap isolation and supervisor CLI configuration. Preserve it in place
  when possible rather than copying a virtual environment across machines.

Useful for authentic presentation replays: archived final evaluation
`mission_*.json.gz` trajectories and requested camera images. These are optional
for running the new experiment, and should not enter the author's workspace as
fresh final evaluation data.

Tracked policy/interface sources and compact outcomes are already available in
Git commit `10614ea716bcde8631ce8eb7a32bde1446849d60`. The new isolated worktree is
`WSPL-supervision-wspl-lora-v1`, branch `research/supervision-wspl-lora-v1`.
