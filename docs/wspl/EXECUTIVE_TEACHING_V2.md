# Executive teaching v2: fresh collection after arrival repair

This replaces the interrupted v1 run, not its archived results. See
EXECUTIVE_TEACHING_V1.md for unchanged design and
EXECUTIVE_TEACHING_ARRIVAL_REPAIR.md for the diagnosed failure and engineering checks.

## Explicit restart boundary and budget

Repeat the four student acquisition checks under the repaired interface. If all
three adaptation checks succeed, stop before any teacher calls or training.
Otherwise collect eight fresh Astra demonstration missions with at most sixteen
responses each (128 new calls maximum). All eight are fresh because the public
completion specification has changed, even before physical trajectories diverge.
Do not reuse historical replies as labels under new prompts. The previous forty
completed calls and one capacity-interrupted invocation (unknown usage) remain
archived and count toward total project cost, separately from this run.

This intentionally avoids a complex simulator-state recovery mechanism. No saved
training is discarded: v1 never trained a student. The exact same saved parent
executive, motor modules, acquisition seeds, decision budgets and deployment
panels are retained. This is a revised exploratory experiment on reused acquisition
conditions; reserved deployment panels remain unscored.

## Only behavioral interface change

Require three consecutive sensor-derived distances within 4 mm before returning
observed_reached. Preserve independent scoring at three true samples within 5 mm,
ten ticks per reach chunk, noise, budgets and all action choices. Tell both models
the new measured completion rule. No automatic retry, goal advancement or hidden
state disclosure. This margin is not a mathematical guarantee under noisy sensing.

Require successful teacher coverage of every condition before any fitting. Then
run the original two 128-update matched LoRA continuations and 96-episode frozen
evaluation. Half-hour rolling checkpoints plus final adapters; no base copies.
No automatic retry, expansion, or follow-on. Capacity errors stop collection.

Expected 2–5 hours; 12-hour study guard, 20-minute author-call guard. Existing Codex
usage; no hard dollar/output-token cap or API fallback. Local GPU student inference
and training, CPU physics. Exact replacement-manifest approval is required before
execution; historical approval is not transferred.
