# Provenance

This project is a fresh repository. index.json records the exact origin/hash of copied source and small evidence artifacts. No checkpoint, corpus, simulator dataset or external credential is copied.

- experiments/source_consistency.py was copied unchanged from Kevin's local Gravemind repository. Its tests were copied with one adaptation: the direct-entry test invokes the guard API instead of a Gravemind runner absent here. Original/destination hashes and that change are recorded. The approval registry starts empty; no old run is authorized here.
- toy_world/program_blocks.json and toy_world/adamw_comparison.json are unchanged prior summaries, retained for research context rather than as Skillforge empirical results.
- toy_world/adamw_comparison.png is the antecedent plot, with the old study's assumptions, not a robotic-control result.
- Legacy robot models and the reported remote-workstation RBL-RMV run remain external evidence with the scope described in docs/EVIDENCE.md.

Machine-specific sibling paths are recorded descriptively, not imported as runtime dependencies. A public GitHub remote and project redistribution license have not been selected.
