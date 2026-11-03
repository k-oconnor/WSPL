# Examples

- pick_place.task.json is a proposed typed temporal specification, with explicitly declared observed/controllable propositions. It is not a generated or executed robot result.
- pick_place.trace.json is a handwritten positive finite trace for that specification. Unit tests also construct ordering, collision and release-location counterexamples.
- lift.skill.json is an unvalidated motor-skill contract showing the proposed 12-input/4-output RMV interface. Its physical ranges and observers require simulator calibration; weights are absent.
- lift_gate.rmv.json is a deliberately tiny two-input/one-output arithmetic example for the scalar reference. It is not an implementation of lift.skill.json and cannot be deployed as its controller. At height_error_m=0.05 and contact=1.0, its normalized conjunction is 0.5 and motor output is 0.1.

The validators check structure and reference arithmetic. No example proves feasibility, physical success, controller admission or task synthesis.
