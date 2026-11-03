"""Versioned motor interface and independent, invocation-scoped observations.

No simulator imports, controller policy, scripted skill or task success shortcut.
"""
from dataclasses import dataclass
import math

INTERFACE = "skillforge.tabletop12.osc4.v1"
CHANNELS = ("object_dx", "object_dy", "object_dz", "target_dx", "target_dy",
            "target_dz", "eef_vx", "eef_vy", "eef_vz", "aperture",
            "left_object_contact", "right_object_contact")


class SensorUnavailable(ValueError):
    """A required measurement is unavailable; do not substitute false or zero."""


def vector(value, size, name):
    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise SensorUnavailable(f"invalid {name}") from exc
    if len(result) != size or not all(math.isfinite(x) for x in result):
        raise SensorUnavailable(f"invalid {name}")
    return result


@dataclass(frozen=True)
class MotorFrame:
    time: float
    eef: tuple
    object: tuple
    target: tuple
    velocity: tuple
    aperture: float
    left_contact: bool
    right_contact: bool

    def __post_init__(self):
        for name in ("eef", "object", "target", "velocity"):
            object.__setattr__(self, name, vector(getattr(self, name), 3, name))
        for name in ("time", "aperture"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise SensorUnavailable(f"invalid {name}")
        if type(self.left_contact) is not bool or type(self.right_contact) is not bool:
            raise SensorUnavailable("contact indicators must be known Booleans")

    @property
    def channels(self):
        return (tuple(a-b for a, b in zip(self.object, self.eef)) +
                tuple(a-b for a, b in zip(self.target, self.eef)) + self.velocity +
                (self.aperture, float(self.left_contact), float(self.right_contact)))


@dataclass(frozen=True)
class OSCCommand:
    """Normalized neural command -> native OSC_POSE delta command.

    2 cm requested translation at magnitude 1; native controller scales 1 to
    5 cm. Zero rotational increments; gripper -1 opens, +1 closes, 0 holds
    the gripper's internal command. These are setpoint increments, not achieved
    physical displacements. No workspace/collision guard is implied.
    """
    translation_step_m: float = .02
    native_translation_m: float = .05

    def __post_init__(self):
        for x in (self.translation_step_m, self.native_translation_m):
            if isinstance(x, bool) or not math.isfinite(x) or x <= 0:
                raise ValueError("positive finite translation scales required")
        if self.translation_step_m > self.native_translation_m:
            raise ValueError("requested scale exceeds native command range")

    def encode(self, command):
        u = vector(command, 4, "motor command")
        if any(abs(x) > 1 for x in u):
            raise ValueError("motor commands must already be bounded to [-1,1]")
        scale = self.translation_step_m / self.native_translation_m
        return tuple(x*scale for x in u[:3]) + (0., 0., 0., u[3])


@dataclass(frozen=True)
class ObserverConfig:
    position_tolerance_m: float = .01
    speed_tolerance_m_s: float = .03
    lift_height_m: float = .05
    stable_ticks: int = 3
    tick_seconds: float = .05

    def __post_init__(self):
        for x in (self.position_tolerance_m, self.speed_tolerance_m_s, self.lift_height_m, self.tick_seconds):
            if isinstance(x, bool) or not math.isfinite(x) or x <= 0:
                raise ValueError("observer thresholds must be positive and finite")
        if type(self.stable_ticks) is not int or self.stable_ticks < 1:
            raise ValueError("stable_ticks must be a positive integer")


class SkillObserver:
    """Measurements, not neural claims, produce one completion pulse per call.

    Approach binds frame.target to a supplied grasp pose. Close only establishes
    bilateral object contact; it does not certify holding/pickup. Lift requires
    bilateral contact at invocation start and every sampled tick, plus elevation
    relative to that start. Missing/invalid ticks abort the invocation. This does
    not establish contact between samples. These are engineering definitions, not
    calibrated reliability/admission criteria. No collision predicate exists yet.
    """
    def __init__(self, config=ObserverConfig()):
        self.config = config
        self._ids = set()
        self._active = False

    def begin(self, invocation_id, skill, frame):
        if not isinstance(invocation_id, str) or not invocation_id or invocation_id in self._ids:
            raise ValueError("fresh nonempty invocation ID required")
        if skill not in ("approach", "close", "lift"):
            raise ValueError("unsupported observed skill")
        if not isinstance(frame, MotorFrame):
            raise SensorUnavailable("MotorFrame required")
        self._ids.add(invocation_id)
        self._active = True
        self.skill, self.invocation_id = skill, invocation_id
        self._start_z, self._time = frame.object[2], frame.time
        self._target = frame.target
        self._contact_intact = frame.left_contact and frame.right_contact
        self._had_contact = self._contact_intact
        self._count = 0
        self._completed = False

    def update(self, frame):
        if not self._active:
            raise ValueError("begin an invocation before observing")
        if (not isinstance(frame, MotorFrame) or
            not math.isclose(frame.time-self._time, self.config.tick_seconds, rel_tol=0., abs_tol=1e-6)):
            self._active = False
            raise SensorUnavailable("missing or invalid motor tick; invocation aborted")
        if frame.target != self._target:
            self._active = False
            raise SensorUnavailable("bound target changed during invocation")
        self._time = frame.time
        bilateral = frame.left_contact and frame.right_contact
        lost = self._had_contact and not bilateral
        self._had_contact = bilateral
        self._contact_intact = self._contact_intact and bilateral
        aligned = (math.dist(frame.eef, frame.target) <= self.config.position_tolerance_m and
                   math.sqrt(sum(x*x for x in frame.velocity)) <= self.config.speed_tolerance_m_s)
        elevated = frame.object[2] - self._start_z >= self.config.lift_height_m
        condition = {"approach": aligned, "close": bilateral,
                     "lift": elevated and self._contact_intact}[self.skill]
        self._count = self._count + 1 if condition else 0
        pulse = self._count >= self.config.stable_ticks and not self._completed
        self._completed = self._completed or pulse
        return {"aligned": aligned, "bilateral_contact": bilateral, "elevated": elevated,
                "contact_lost": lost, "approach_done": pulse and self.skill == "approach",
                "close_done": pulse and self.skill == "close", "lift_done": pulse and self.skill == "lift"}
