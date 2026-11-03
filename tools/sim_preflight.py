"""Opt-in headless integration check, not a skill/benchmark/training run.

Run in its own interpreter; LIBERO uses process-global import configuration.
Writes only the requested work directory. Never imports legacy motor routines.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def windows_binding_patch():
    """Two narrowly checked upstream 1.4.1 Windows fixes, inside this venv only.

    Legacy experiments identified the same DLL lookup / EGL issues. This is a
    fresh implementation, not an import of their environment-mutating helper.
    Before/after hashes are retained. No legacy environment is modified.
    """
    if sys.platform != "win32":
        return None
    if sys.prefix == sys.base_prefix or metadata.version("robosuite") != "1.4.1":
        raise RuntimeError("Windows fix requires robosuite 1.4.1 in a virtualenv")
    path = Path(importlib.util.find_spec("robosuite").origin).parent / "utils/binding_utils.py"
    if not path.resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("refusing to modify a package outside the active virtualenv")
    before = sha(path)
    text = path.read_text(encoding="utf-8")
    replacements = [
        ('os.path.dirname(__file__), "mujoco.dll"', 'os.path.dirname(mujoco.__file__), "mujoco.dll"'),
        ('if macros.MUJOCO_GPU_RENDERING and os.environ.get',
         'if _SYSTEM != "Windows" and macros.MUJOCO_GPU_RENDERING and os.environ.get'),
    ]
    for old, new in replacements:
        if text.count(old) == 1:
            text = text.replace(old, new)
        elif text.count(new) != 1:
            raise RuntimeError("unexpected robosuite source: manual compatibility audit required")
    if hashlib.sha256(text.encode()).hexdigest() != before:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    return {"package_file": "robosuite/utils/binding_utils.py", "before_sha256": before,
            "after_sha256": sha(path), "legacy_knowledge": "experiments/libero_control/patch_runtime.py"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bindings", type=Path, default=ROOT / "artifacts/local_paths.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fix-windows-bindings", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "engineering integration unit check", "scientific_run": False,
              "training": False, "skill_policy": None, "simulator_steps": 0, "status": "failed"}
    env = None
    started = time.monotonic()
    try:
        bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
        checkout = Path(bindings["libero_checkout"]).resolve(strict=True)
        base = checkout / "libero/libero"
        task = base / "bddl_files/libero_goal/put_the_cream_cheese_in_the_bowl.bddl"
        if not task.is_file():
            raise ValueError("expected LIBERO engineering task asset missing")
        report["versions"] = {n: metadata.version(n) for n in
                              ("mujoco", "robosuite", "numpy", "gym", "bddl")}
        if report["versions"]["robosuite"] != "1.4.1":
            raise RuntimeError("adapter preflight audited only against robosuite 1.4.1")
        report["libero_commit"] = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        report["libero_dirty"] = bool(subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain"], text=True).strip())
        report["task_sha256"] = sha(task)
        report["source_sha256"] = {str(f.relative_to(ROOT)).replace("\\", "/"): sha(f)
                                    for f in [Path(__file__).resolve(), ROOT / "src/skillforge/motor.py",
                                              ROOT / "src/skillforge/libero_adapter.py"]}
        if args.fix_windows_bindings:
            report["windows_fix"] = windows_binding_patch()
        config = args.output / "libero_config"
        config.mkdir()
        data = args.output / "empty_datasets"
        data.mkdir()
        paths = {"benchmark_root": base, "bddl_files": base / "bddl_files",
                 "init_states": base / "init_files", "assets": base / "assets", "datasets": data.resolve()}
        (config / "config.yaml").write_text(json.dumps({k: v.as_posix() for k, v in paths.items()}), encoding="utf-8")
        if "libero.libero" in sys.modules:
            raise RuntimeError("use a fresh process; LIBERO already configured")
        os.environ["LIBERO_CONFIG_PATH"] = str(config.resolve())
        sys.path.insert(0, str(checkout))
        from libero.libero.envs.env_wrapper import ControlEnv
        from skillforge.libero_adapter import LiberoMotorAdapter
        from skillforge.motor import CHANNELS, INTERFACE
        env = ControlEnv(bddl_file_name=str(task), use_camera_obs=False, has_renderer=False,
                         has_offscreen_renderer=False, control_freq=20, horizon=50,
                         ignore_done=False, hard_reset=False)
        env.seed(17)
        # Wrapper reset retries forever on RandomizationError. Preflight fails
        # explicitly on a bad layout instead of concealing retry cost.
        obs = env.env.reset()
        target = tuple(float(x) for x in obs["cream_cheese_1_pos"])
        adapter = LiberoMotorAdapter(env, "cream_cheese_1", target)
        first = adapter.observe(obs)
        frames = [{"time": first.time, "channels": first.channels}]
        logs = []
        for _ in range(5):
            frame, log = adapter.step((0., 0., 0., 0.))
            report["simulator_steps"] += 1
            if frame.time <= frames[-1]["time"]:
                raise AssertionError("simulator time failed to advance")
            frames.append({"time": frame.time, "channels": frame.channels})
            logs.append(log)
            if log["native_done"]:
                break
        report.update(status="passed", interface=INTERFACE, channels=CHANNELS,
                      frames=frames, logs=logs, control_hz=20,
                      policy="five zero setpoint increments, gripper internal-command hold",
                      limitations=["No physical skill reliability or no-op difficulty result",
                                   "No collision observer or physical planner/executor integration",
                                   "Bound target is object origin for sensor wiring only, not a grasp pose"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if env is not None:
            env.close()
        report["wall_seconds"] = time.monotonic() - started
        (args.output / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"status": report["status"], "simulator_steps": report["simulator_steps"],
                          "report": str(args.output / "preflight.json")}))


if __name__ == "__main__":
    main()
