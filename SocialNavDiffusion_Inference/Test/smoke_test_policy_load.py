import configparser
import copy
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_DIR = os.path.join(REPO_ROOT, "crowd_nav", "policy")
CONFIG_PATH = os.path.join(REPO_ROOT, "crowd_nav", "configs", "policy.config")
CKPT_PATH = os.path.join(REPO_ROOT, "SocialGuidedNavPlanner.pt")

for path in (REPO_ROOT, POLICY_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

required_config = [
    "start_dim",
    "goal_dim",
    "token_dim",
    "horizon",
    "k_max",
    "num_diffusion_timesteps",
    "num_inference_steps",
    "num_samples",
    "safety_radius",
    "style_vector",
]

try:
    from crowd_nav.policy.diffusion_CondUNetCFG import DiffusionConditionalUNet1DCFG
except Exception as exc:
    print("POLICY_IMPORT_FAILED")
    print(f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    raise SystemExit(1)

if not os.path.exists(CKPT_PATH):
    print(f"CHECKPOINT_MISSING: {CKPT_PATH}")
    raise SystemExit(1)

config = configparser.RawConfigParser()
config.read(CONFIG_PATH)

section = "diffusion_conditional_unet1dcfg"
missing = [
    key for key in required_config
    if not config.has_option(section, key)
]

if missing:
    print("CONFIG_MISSING_FIELDS")
    for key in missing:
        print(f"  - [{section}] {key}")
    raise SystemExit(1)

configured_ckpt = config.get(section, "ckpt_path")
configured_norm = config.get(section, "norm_file")

candidate_norm_paths = [
    configured_norm,
    os.path.join(REPO_ROOT, configured_norm),
    os.path.join(REPO_ROOT, "crowd_nav", configured_norm),
    os.path.join(REPO_ROOT, "crowd_nav", "configs", configured_norm),
]
norm_path = next((path for path in candidate_norm_paths if os.path.exists(path)), None)

if norm_path is None:
    print("NORM_FILE_MISSING")
    print(f"configured norm_file: {configured_norm}")
    print("searched:")
    for path in candidate_norm_paths:
        print(f"  - {path}")
    raise SystemExit(1)

if configured_ckpt != os.path.basename(CKPT_PATH) and not os.path.exists(configured_ckpt):
    print("CONFIG_CKPT_PATH_MISMATCH")
    print(f"configured ckpt_path: {configured_ckpt}")
    print(f"available checkpoint: {CKPT_PATH}")
    print("Using available checkpoint for this smoke test.")

config = copy.deepcopy(config)
config.set(section, "ckpt_path", CKPT_PATH)
config.set(section, "norm_file", norm_path)

try:
    policy = DiffusionConditionalUNet1DCFG()
    policy.configure(config)
except Exception as exc:
    print("POLICY_CONFIGURE_FAILED")
    print(f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    raise SystemExit(1)

print("POLICY_CONFIGURE_OK")
print("Note: this smoke test does not call predict() or ROS.")
