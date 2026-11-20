# Teaching pilot: arrival-interface repair

The original approved run (manifest
`52e43287bc87b5ff51315bbd395a9cca1f6a84a082342b996de4c45666d5a539`)
remains unchanged in `../WSPL-executive-teaching-v1`. Collection stopped on an
Astra capacity error during teacher mission 4, response 5 (zero-based). Forty
completed replies and four completed mission traces survive. No student training
or reserved evaluation occurred.

## Failure diagnosis from saved traces

Both ordinary demonstrations succeeded. Both changed-calibration demonstrations
failed the physical scorer despite correct probing and selection of the repaired
controller. Public `observed_reached` feedback caused the executive to advance or
stop one tick before the independent dwell requirement was met.

For teacher_2 goal_8452, the last three true distances were 5.0561, 4.2062,
3.4987 mm. For teacher_3 goal_9858 they were 5.0710, 4.2391, 3.5483 mm.
Each therefore had only two consecutive samples within 5 mm. No scores have
been relabeled. The first also incurred an ordering violation because its first
goal never registered physically.

## Implemented repair

The sensor-facing reach action now requires three consecutive measured samples
within 4 mm, retaining the ten-tick action bound and all controller weights.
The independent scorer still requires three true samples within 5 mm. Public
documentation states the measured criterion. No true state is exposed to the
executive. The 1 mm margin is an engineering precaution, not a deterministic
guarantee under unbounded sensor noise.

Twenty focused unit tests pass, including a boundary-noise regression and a
test rejecting arrival outside the conservative margin. The existing physical
engineering fixtures (seeds 991–994, not acquisition or reserved scenes) pass
in ordinary, changed, switch and return conditions. They use an explicit
engineering oracle, not Astra or Qwen: 11/11/13/12 decisions and 75/74/86/75
physical ticks respectively. Receipt: `teaching_physics_preflight.json`.

## Recovery constraints

Do not restart the original runner or reuse its source approval. This checkout
contains an unapproved source change. No replacement scientific run, model call
or training has launched.

Preserve original replies and usage. Longer reach actions change physical states,
sensor samples and subsequent prompts. Old later replies cannot be silently
treated as responses to those new observations. A recovery must either verify
exact prompt/state equivalence before reusing a response, or explicitly collect
fresh demonstrations under the revised interface. Historical trajectories may
remain diagnostic evidence but are not repaired successful imitation labels.

Before launch, prepare a replacement protocol and pin with explicit additional
call accounting, validate recovery boundaries, and obtain exact digest approval
as required by AGENTS.md. The original teacher-coverage gate remains intact.
