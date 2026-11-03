# Examples

- pick_place.task.json is a proposed typed temporal specification, with explicitly declared observed/controllable propositions. It is not a generated or executed robot result.
- pick_place.trace.json is a handwritten positive finite trace for that specification. Unit tests also construct ordering, collision and release-location counterexamples.
- lift.skill.json is an unvalidated motor-skill contract showing the proposed 12-input/4-output RMV interface. Its physical ranges and observers require simulator calibration; weights are absent.
- lift_gate.rmv.json is a deliberately tiny two-input/one-output arithmetic example for the scalar reference. It is not an implementation of lift.skill.json and cannot be deployed as its controller. At height_error_m=0.05 and contact=1.0, its normalized conjunction is 0.5 and motor output is 0.1.

The initial task/skill/RMV examples check structure and reference arithmetic. They do not establish physical feasibility, success or controller admission.

The discrete_pickup.task.json / discrete_pickup.model.json pair additionally exercises real finite-model strategy synthesis and abstract execution. It uses an abstract_tick clock and supplied bounded-failure contracts, not a robot simulator. The example admits at most one approach timeout and one grasp failure before guaranteed recovery. Changing those outcomes to allow infinite adversarial failures can make the same task unrealizable. Use the README plan command to inspect the generated strategy/trace. This does not establish physical success or controller admission.

discrete_pickup.expected.json records the M1 fixture's generated strategy and last-outcome execution trace for inspection. It is an engineering example output, not a robot or learned-skill benchmark score.
