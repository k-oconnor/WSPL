"""Opt-in LIBERO / robosuite 1.4.1 Panda sensor and single-tick command bridge.

No legacy motor routines are imported. Instantiate after reset, before stepping.
The caller owns environment construction/lifetime and supplies object/goal IDs.
"""
from .motor import MotorFrame, OSCCommand, SensorUnavailable, vector
import math


class LiberoMotorAdapter:
    def __init__(self, env, object_name, target, command=OSCCommand(), max_steps=1000):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.env, self.object_name, self.command = env, object_name, command
        self.max_steps, self.step_calls = max_steps, 0
        self.target = vector(target, 3, "bound target")
        self._ready = False
        self._done = False
        base = env.env
        if len(base.robots) != 1 or base.control_freq != 20 or base.action_dim != 7:
            raise ValueError("requires one Panda, 20 Hz and 7D OSC_POSE actions")
        robot = base.robots[0]
        if type(robot.gripper).__name__ != "PandaGripper":
            raise ValueError("only audited PandaGripper supported")
        controller = robot.controller
        import numpy as np  # optional dependency, never imported by reference tests
        if (type(controller).__name__ != "OperationalSpaceController" or
            not controller.use_delta or not controller.use_ori or
            controller.impedance_mode != "fixed" or
            not np.allclose(controller.input_min, -1) or
            not np.allclose(controller.input_max, 1) or
            not np.allclose(controller.output_max, [.05, .05, .05, .5, .5, .5]) or
            not np.allclose(controller.output_min, [-.05, -.05, -.05, -.5, -.5, -.5])):
            raise ValueError("controller differs from audited OSC_POSE configuration")
        if command.native_translation_m != .05:
            raise ValueError("native translation scale must match the audited controller")
        self.robot, self.sim = robot, base.sim
        obj = base.get_object(object_name)
        self._object_geoms = {self.sim.model.geom_name2id(n) for n in obj.contact_geoms}
        self._finger_geoms = tuple({self.sim.model.geom_name2id(n) for n in robot.gripper.important_geoms[k]}
                                   for k in ("left_finger", "right_finger"))
        self._pad_ids = tuple(self.sim.model.geom_name2id(robot.gripper.important_geoms[k][0])
                              for k in ("left_fingerpad", "right_fingerpad"))
        if not self._object_geoms or not all(self._finger_geoms):
            raise SensorUnavailable("required object/finger collision geometry missing")
        self._site = robot.gripper.important_sites["grip_site"]
        self._validate_pad_geometry()

    def _validate_pad_geometry(self):
        # MuJoCo boxes: local y is the pad normal in the audited Panda asset.
        # Compute inner faces, not a joint-coordinate proxy for jaw opening.
        for i in self._pad_ids:
            if int(self.sim.model.geom_type[i]) != 6:  # mjGEOM_BOX
                raise ValueError("finger pad is not the audited box geometry")

    def _aperture(self):
        import numpy as np
        a, b = self._pad_ids
        data, model = self.sim.data, self.sim.model
        direction = np.asarray(data.geom_xpos[b]) - np.asarray(data.geom_xpos[a])
        distance = float(np.linalg.norm(direction))
        if not np.isfinite(distance) or distance <= 0:
            raise SensorUnavailable("coincident finger pads")
        normal = direction / distance
        # Support radius of each oriented box along the line between centers.
        radii = [sum(abs(np.asarray(data.geom_xmat[i]).reshape(3, 3).T @ normal) * model.geom_size[i])
                 for i in (a, b)]
        if not np.all(np.isfinite(radii)):
            raise SensorUnavailable("invalid finger pad geometry")
        return max(0., distance - sum(radii))

    def observe(self, obs):
        self._ready = False
        try:
            data = self.sim.data
            contacts = [False, False]
            for i in range(data.ncon):
                contact = data.contact[i]
                if not math.isfinite(float(contact.dist)):
                    raise SensorUnavailable("invalid physical contact distance")
                # geom pairs are symmetric; ignore contacts with unrelated bodies.
                pair = {int(contact.geom1), int(contact.geom2)}
                if contact.dist <= 0 and pair & self._object_geoms:
                    for side, geoms in enumerate(self._finger_geoms):
                        contacts[side] |= bool(pair & geoms)
            frame = MotorFrame(float(data.time), obs["robot0_eef_pos"],
                               obs[f"{self.object_name}_pos"], self.target,
                               data.get_site_xvelp(self._site), self._aperture(), *contacts)
        except (KeyError, AttributeError, IndexError, TypeError) as exc:
            raise SensorUnavailable("required LIBERO sensor unavailable") from exc
        self._ready = True
        return frame

    def step(self, command):
        if self._done or not self._ready or self.step_calls >= self.max_steps:
            raise RuntimeError("observe a valid frame first; reset/rebind after terminal state")
        native = self.command.encode(command)
        self._ready = False
        import numpy as np
        self.step_calls += 1
        obs, reward, done, info = self.env.step(np.asarray(native))
        self._done = bool(done)
        frame = self.observe(obs)
        # Native done is not a skill-completion proposition; callers must log it.
        return frame, {"native_reward": float(reward), "native_done": self._done,
                       "native_info": info, "native_command": list(native),
                       "step_calls": self.step_calls, "budget_exhausted": self.step_calls >= self.max_steps}
