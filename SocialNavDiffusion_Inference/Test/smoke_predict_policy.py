import configparser
import copy
import os
import sys
import time
import traceback


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_DIR = os.path.join(REPO_ROOT, "crowd_nav", "policy")
CONFIG_PATH = os.path.join(REPO_ROOT, "crowd_nav", "configs", "policy.config")
ENV_CONFIG_PATH = os.path.join(REPO_ROOT, "crowd_nav", "configs", "env.config")
CKPT_PATH = os.path.join(REPO_ROOT, "SocialGuidedNavPlanner.pt")

for path in (REPO_ROOT, POLICY_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def resolve_existing_path(configured_path, *extra_candidates):
    candidates = [
        configured_path,
        os.path.join(REPO_ROOT, configured_path),
        os.path.join(REPO_ROOT, "crowd_nav", configured_path),
        os.path.join(REPO_ROOT, "crowd_nav", "configs", configured_path),
    ]
    candidates.extend(extra_candidates)
    return next((path for path in candidates if os.path.exists(path)), None), candidates


def shape_or_none(value):
    if value is None:
        return None
    return getattr(value, "shape", None)


def action_fields(action):
    if hasattr(action, "_asdict"):
        return dict(action._asdict())
    return vars(action) if hasattr(action, "__dict__") else str(action)


try:
    from crowd_nav.policy.diffusion_CondUNetCFG import DiffusionConditionalUNet1DCFG
    from crowd_sim.envs.utils.state import FullState, JointState, ObservableState
except Exception as exc:
    print("IMPORT_FAILED")
    print(f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    raise SystemExit(1)

if not os.path.exists(CKPT_PATH):
    print(f"CHECKPOINT_MISSING: {CKPT_PATH}")
    raise SystemExit(1)

config = configparser.RawConfigParser()
config.read(CONFIG_PATH)

section = "diffusion_conditional_unet1dcfg"
configured_norm = config.get(section, "norm_file")
norm_path, norm_candidates = resolve_existing_path(configured_norm)
if norm_path is None:
    print("NORM_FILE_MISSING")
    print(f"configured norm_file: {configured_norm}")
    print("searched:")
    for path in norm_candidates:
        print(f"  - {path}")
    raise SystemExit(1)

config = copy.deepcopy(config)
config.set(section, "ckpt_path", CKPT_PATH)
config.set(section, "norm_file", norm_path)

policy = DiffusionConditionalUNet1DCFG()
try:
    policy.configure(config)
except Exception as exc:
    print("POLICY_CONFIGURE_FAILED")
    print(f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    raise SystemExit(1)

env_config = configparser.RawConfigParser()
env_config.read(ENV_CONFIG_PATH)
policy.time_step = (
    env_config.getfloat("env", "time_step")
    if env_config.has_option("env", "time_step")
    else 0.25
)

# Keep map conditioning enabled exactly as configured, but provide an explicit
# "no map available" state so use_map=True follows the policy's null-map path.
if getattr(policy, "use_map", False):
    policy.set_static_map(None, has_map=0.0, map_extent=policy.map_extent)


def make_state(with_human=False):
    robot = FullState(
        px=0.0,
        py=0.0,
        vx=0.0,
        vy=0.0,
        radius=0.3,
        gx=3.0,
        gy=0.0,
        v_pref=1.0,
        theta=0.0,
    )
    humans = []
    if with_human:
        humans.append(ObservableState(
            px=10.0,
            py=10.0,
            vx=0.0,
            vy=0.0,
            radius=0.3,
        ))
    return JointState(robot, humans)


state = make_state(with_human=False)
try:
    t0 = time.perf_counter()
    action = policy.predict(state)
    runtime_ms = (time.perf_counter() - t0) * 1000.0
except Exception as first_exc:
    print("PREDICT_NO_HUMANS_FAILED")
    print(f"{type(first_exc).__name__}: {first_exc}")
    traceback.print_exc()
    print("Retrying with one far-away human at (10, 10).")
    state = make_state(with_human=True)
    try:
        t0 = time.perf_counter()
        action = policy.predict(state)
        runtime_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as second_exc:
        print("PREDICT_WITH_FAR_HUMAN_FAILED")
        print(f"{type(second_exc).__name__}: {second_exc}")
        traceback.print_exc()
        raise SystemExit(1)

predicted = policy.predicted_traj or {}
selected_sample = predicted.get("selected_sample")
projection = predicted.get("projection")

print("SMOKE_PREDICT_OK")
print(f"state_human_count: {len(state.human_states)}")
print(f"time_step: {policy.time_step}")
print(f"action type: {type(action).__name__}")
print(f"action fields: {action_fields(action)}")
print(f"predicted_traj keys: {list(predicted.keys())}")
print(f"selected_sample shape: {shape_or_none(selected_sample)}")
print(f"projection shape: {shape_or_none(projection)}")
print(f"used_projection: {predicted.get('used_projection')}")
print(f"runtime milliseconds: {runtime_ms:.3f}")
