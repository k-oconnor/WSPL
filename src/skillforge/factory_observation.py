"""Factory observation boundary: explicit sensor allowlist and requested captures.

The backend may own privileged simulator state. This class never serializes it.
Camera captures are prospective, timestamped, and do not advance physics.
"""
import hashlib
import math
from pathlib import Path
import numpy as np
from PIL import Image


CAMERAS = ('frontview', 'robot0_eye_in_hand')
EVENTS = ('command_start', 'command_end', 'gripper_command_change')


def vector(value, n, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError('Invalid ' + name)
    return a


class ObservationSession:
    def __init__(self, backend, output, seed, max_steps=120, max_frames=32):
        self.backend = backend
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.rng = np.random.default_rng(seed)
        self.max_steps, self.max_frames = max_steps, max_frames
        self.steps = 0
        self.frames = []
        self.log = []
        self.last_command = [0., 0., 0., -1.]
        self._cached_tick = None
        self._cached_telemetry = None

    def observe(self):
        # One sensor sample per physical tick: repeated reads cannot average away noise.
        if self._cached_tick == self.steps:
            return dict(self._cached_telemetry)
        raw = self.backend.sensors()
        q = vector(raw['joint_position_rad'], 7, 'joint position')
        dq = vector(raw['joint_velocity_rad_s'], 7, 'joint velocity')
        grip = vector(raw['gripper_joint_position_m'], 2, 'gripper')
        # Pose is nominal FK from the SAME noisy/quantized encoders, not true object state.
        q = np.round((q + self.rng.normal(0, 1e-4, 7)) / 1e-4) * 1e-4
        pose = self.backend.forward_kinematics(q)
        position = vector(pose['position_m'], 3, 'FK position')
        rotation = vector(pose['quaternion_xyzw'], 4, 'FK orientation')
        result = dict(timestamp_s=float(raw['timestamp_s']), sequence=self.steps,
            joint_position_rad=q.tolist(),
            joint_velocity_rad_s=(np.round(dq / 1e-3) * 1e-3).tolist(),
            gripper_joint_position_m=(np.round(grip / 1e-4) * 1e-4).tolist(),
            nominal_tool_position_m=position.tolist(), nominal_tool_quaternion_xyzw=rotation.tolist(),
            last_command=list(self.last_command),
            controller_status='simulation_running',
            remaining_steps=self.max_steps-self.steps,
            remaining_frames=self.max_frames-len(self.frames))
        if not math.isfinite(result['timestamp_s']):
            raise ValueError('Invalid clock')
        self._cached_tick, self._cached_telemetry = self.steps, result
        return dict(result)

    def capture(self, camera, event='requested_now'):
        if camera not in CAMERAS:
            raise ValueError('Unknown camera')
        if len(self.frames) >= self.max_frames:
            raise ValueError('Frame budget exhausted')
        time_before = self.backend.sensors()['timestamp_s']
        rgb = self.backend.render(camera)
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (256, 256, 3):
            raise ValueError('Camera must produce 256x256 RGB uint8')
        if self.backend.sensors()['timestamp_s'] != time_before:
            raise ValueError('Rendering advanced physics')
        name = f'frame_{len(self.frames):04d}.png'
        Image.fromarray(rgb).save(self.output/name)
        item = dict(frame_id=len(self.frames), path=name, camera=camera, event=event,
                    timestamp_s=float(time_before), sequence=self.steps,
                    sha256=hashlib.sha256((self.output/name).read_bytes()).hexdigest())
        self.frames.append(item)
        self.log.append(dict(kind='frame', **item))
        self._cached_tick = None  # Update budget without redrawing sensors below.
        # Preserve the sensor draw if one was already observed this tick.
        if self._cached_telemetry is not None and self._cached_telemetry['sequence'] == self.steps:
            self._cached_telemetry['remaining_frames'] = self.max_frames-len(self.frames)
            self._cached_tick = self.steps
        return dict(item)

    def act(self, command, ticks, cameras=(), events=(), clip_stride=None):
        command = vector(command, 4, 'OSC command')
        if np.any(np.abs(command[:3]) > .2) or abs(command[3]) > 1:
            raise ValueError('Command exceeds engineering action envelope')
        if type(ticks) is not int or not 1 <= ticks <= 20 or self.steps+ticks > self.max_steps:
            raise ValueError('Step budget exceeded')
        if len(set(cameras)) != len(cameras) or any(c not in CAMERAS for c in cameras):
            raise ValueError('Invalid cameras')
        if len(set(events)) != len(events) or any(e not in EVENTS for e in events):
            raise ValueError('Events must be observable command transitions')
        if clip_stride is not None and (type(clip_stride) is not int or not 1 <= clip_stride <= ticks):
            raise ValueError('Invalid clip stride')
        changed = command[3] != self.last_command[3]
        event_count = int('command_start' in events)+int('command_end' in events)+int(changed and 'gripper_command_change' in events)
        needed = len(cameras)*(event_count+(ticks//clip_stride if clip_stride else 0))
        if needed > 12:
            raise ValueError('At most 12 requested frames per operation')
        if len(self.frames)+needed > self.max_frames:
            raise ValueError('Requested capture exceeds frame budget')
        start_frames = len(self.frames)
        for event in ('command_start', 'gripper_command_change'):
            if event in events and (event != 'gripper_command_change' or changed):
                for camera in cameras: self.capture(camera, event)
        self.last_command = command.tolist()
        samples = []
        for i in range(ticks):
            self.backend.step(command)
            self.steps += 1
            self._cached_tick = None
            sample = self.observe()
            samples.append(sample)
            self.log.append(dict(kind='telemetry', **sample))
            if clip_stride and (i+1) % clip_stride == 0:
                for camera in cameras: self.capture(camera, 'clip_sample')
        if 'command_end' in events:
            for camera in cameras: self.capture(camera, 'command_end')
        return dict(telemetry=samples, frames=self.frames[start_frames:])


class PandaBackend:
    """Robosuite physics with camera and encoder interfaces; no object observations."""
    def __init__(self, seed=1, mapping=None):
        import robosuite as suite
        from robosuite.controllers import load_controller_config
        np.random.seed(seed)
        self.env = suite.make('Lift', robots='Panda',
            controller_configs=load_controller_config(default_controller='OSC_POSE'),
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
            use_object_obs=False, control_freq=20, horizon=1000, ignore_done=False,
            hard_reset=False, camera_names=list(CAMERAS))
        self.env.reset()
        for _ in range(20): self.env.step(np.array([0.,0.,0.,0.,0.,0.,-1.]))
        # A separate MjData performs nominal FK without altering the physical world.
        import mujoco
        self._mj = mujoco
        self._fk = mujoco.MjData(self.env.sim.model._model)
        self.robot = self.env.robots[0]
        self._joint_indices = self.robot._ref_joint_pos_indexes
        self._velocity_indices = self.robot._ref_joint_vel_indexes
        self._gripper_indices = self.robot._ref_gripper_joint_pos_indexes
        self._eef_site = self.robot.eef_site_id
        self.mapping = np.eye(3) if mapping is None else np.asarray(mapping,dtype=float)
        if self.mapping.shape != (3,3) or not np.isfinite(self.mapping).all():
            raise ValueError('Invalid private actuator configuration')

    def sensors(self):
        data = self.env.sim.data
        return dict(timestamp_s=float(data.time),
            joint_position_rad=np.array(data.qpos[self._joint_indices]),
            joint_velocity_rad_s=np.array(data.qvel[self._velocity_indices]),
            gripper_joint_position_m=np.array(data.qpos[self._gripper_indices]))

    def forward_kinematics(self, q):
        self._fk.qpos[:] = self.env.sim.model._model.qpos0
        self._fk.qpos[self._joint_indices] = q
        self._mj.mj_forward(self.env.sim.model._model, self._fk)
        quaternion = np.empty(4)
        self._mj.mju_mat2Quat(quaternion, self._fk.site_xmat[self._eef_site])
        return dict(position_m=self._fk.site_xpos[self._eef_site].copy(),
                    quaternion_xyzw=quaternion[[1,2,3,0]])

    def step(self, command):
        translated = np.clip(self.mapping @ np.asarray(command[:3]),-1,1)
        _, _, done, _ = self.env.step(np.array([*translated,0.,0.,0.,command[3]]))
        if done: raise RuntimeError('Native episode terminated')

    def render(self, camera):
        return np.ascontiguousarray(self.env.sim.render(width=256,height=256,camera_name=camera)[::-1])

    def close(self): self.env.close()
