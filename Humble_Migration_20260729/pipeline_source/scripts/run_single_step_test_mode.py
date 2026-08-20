#!/usr/bin/env python3
"""Run one SocialNavDiffusion inference/projection/control step without ROS topics.

This is intended for repository smoke testing inside a prepared environment.
It does not start Gazebo and does not connect to a robot.
"""

import configparser
import os
import sys
from pathlib import Path


def bootstrap_runtime(snd_root: Path):
    """Use the model venv and acados library path before imports/dlopen."""
    env = os.environ.copy()
    changed = False

    venv_dir = Path(
        env.get(
            "SOCIAL_NAV_DIFFUSION_VENV",
            "/home/ubuntu/social_nav_diffusion_humble_venv",
        )
    )
    venv_python = venv_dir / "bin/python"
    in_model_venv = Path(sys.prefix).resolve() == venv_dir.resolve()
    if venv_python.exists() and not in_model_venv:
        env["SND_SINGLE_STEP_BOOTSTRAPPED"] = "1"
        os.execve(str(venv_python), [str(venv_python), *sys.argv], env)

    acados_root = Path(env.get("ACADOS_SOURCE_DIR", "/home/ubuntu/acados"))
    acados_lib = str(acados_root / "lib")
    env["ACADOS_SOURCE_DIR"] = str(acados_root)
    ld_paths = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
    if acados_lib not in ld_paths:
        env["LD_LIBRARY_PATH"] = acados_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        changed = True

    if changed and env.get("SND_SINGLE_STEP_LD_BOOTSTRAPPED") != "1":
        env["SND_SINGLE_STEP_LD_BOOTSTRAPPED"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], env)

    os.environ.update(env)


def add_inference_paths(snd_root: Path):
    for rel in ["", "crowd_nav/policy", "diffusers_unet_1d_condition"]:
        path = str(snd_root / rel) if rel else str(snd_root)
        if path not in sys.path:
            sys.path.insert(0, path)


def main():
    snd_root = Path(os.environ.get("SND_ROOT", "/workspace/SocialNavDiffusion_Inference"))
    ckpt = snd_root / "ckpt_step990000_sogudiff_singleaxis_1p5M.pt"
    norm = snd_root / "norm_stats_sogudiff_allarms_1p5M.npy"
    policy_config = snd_root / "crowd_nav/configs/policy_test_mode.config"
    env_config = snd_root / "crowd_nav/configs/env.config"

    print("[single_step] checking files")
    for path in [snd_root, ckpt, norm, policy_config, env_config]:
        print(f"  {path}: exists={path.exists()}")
        if not path.exists():
            raise FileNotFoundError(path)

    bootstrap_runtime(snd_root)
    add_inference_paths(snd_root)

    print("[single_step] importing policy/state classes")
    from crowd_nav.policy.diffusion_CondUNetCFG_test_mode import DiffusionConditionalUNet1DCFG
    from crowd_sim.envs.utils.state import FullState, JointState, ObservableState

    print("[single_step] loading config and model")
    config = configparser.RawConfigParser()
    config.read(policy_config)
    section = "diffusion_conditional_unet1dcfg"
    config.set(section, "ckpt_path", str(ckpt))
    config.set(section, "norm_file", str(norm))

    policy = DiffusionConditionalUNet1DCFG()
    policy.configure(config)

    env = configparser.RawConfigParser()
    env.read(env_config)
    policy.time_step = env.getfloat("env", "time_step", fallback=0.25)
    print(f"[single_step] policy.time_step={policy.time_step}")

    print("[single_step] creating sample conditioning state")
    robot = FullState(
        px=0.0, py=0.0, vx=0.0, vy=0.0,
        radius=env.getfloat("robot", "radius", fallback=0.25),
        gx=2.0, gy=0.0,
        v_pref=env.getfloat("robot", "v_pref", fallback=1.0),
        theta=0.0, omega=0.0,
    )
    humans = [
        ObservableState(px=1.0, py=0.8, vx=0.0, vy=0.0, radius=0.25),
    ]
    state = JointState(robot, humans)

    print("[single_step] setting sample static map")
    try:
        import numpy as np
        occ = np.array([[1.5, 1.5], [1.5, -1.5], [2.5, 1.5], [2.5, -1.5]], dtype=float)
    except Exception:
        occ = None
    policy.set_static_map(occ, 1.0 if occ is not None else 0.0, policy.map_extent)

    print("[single_step] running conditioning -> inference -> acados/projection -> action")
    action = policy.predict(state)
    fields = action._asdict() if hasattr(action, "_asdict") else vars(action)
    print(f"[single_step] raw action: type={type(action).__name__}, fields={fields}")
    if "v" in fields and "r" in fields:
        angular_z = float(fields["r"]) / float(policy.time_step)
        print(f"[single_step] final control: linear.x={float(fields['v']):.6f}, angular.z={angular_z:.6f}")
    elif "vx" in fields and "vy" in fields:
        print(f"[single_step] final control ActionXY: vx={float(fields['vx']):.6f}, vy={float(fields['vy']):.6f}")
    else:
        print("[single_step] final control: unknown action format")
    print("[single_step] done")


if __name__ == "__main__":
    main()
