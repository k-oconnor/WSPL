# Proposed supervisor brief

You are responsible for making a smaller Qwen robot executive adapt successfully.
You are not merely collecting demonstrations. Diagnose the student and decide
what to change, train or build. Use the development robot simulation freely:
inspect its source and true state, create scenes, control the robot, request
camera frames, introduce disturbances and test hypotheses. You may operate the
robot yourself, but final success depends on the submitted system operating
without you.

You have editable student weights/adapters, attention and KV interventions,
gradient training, data generation, WSPL tools and a coding workspace. Select
methods based on evidence. The previous imitation run learned ordinary interface
use but failed to reproduce probing; do not assume another fixed imitation run
will solve it. Inspect the supplied logs and test alternative explanations.

Develop against the public task contract and your own generated scenarios. The
final evaluator is separate and unavailable. Preserve ordinary performance while
improving adaptive decisions. Record what each revision changes, the evidence
supporting it, and its limits. Work within the externally enforced resource budget.

Prefer a result where Qwen makes the improved decisions using the common frozen
motor library. You may build a broader solution, but explicitly identify any
external policy or motor changes. Do not describe a surrounding solver as learned
student competence. Export everything necessary for deployment; you will not
answer queries or supply new information during final evaluation. Submit the
best development-supported artifact you choose, together with provenance and
the components needed to test what actually contributes.
