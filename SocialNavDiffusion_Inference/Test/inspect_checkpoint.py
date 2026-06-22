import os
import sys
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.path.join(REPO_ROOT, "SocialGuidedNavPlanner.pt")

if not os.path.exists(CKPT_PATH):
    print(f"CHECKPOINT_MISSING: {CKPT_PATH}")
    raise SystemExit(1)

ckpt = torch.load(CKPT_PATH, map_location="cpu")

print(f"checkpoint_path: {CKPT_PATH}")
print(f"checkpoint_type: {type(ckpt).__name__}")

if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print("top_level_keys:")
    for key in keys:
        print(f"  - {key}")

    expected = [
        "model_state_dict",
        "ema_model_state_dict",
        "scene_embedder_state_dict",
        "ema_scene_embedder_state_dict",
        "style_embedder_state_dict",
        "ema_style_embedder_state_dict",
    ]
    print("expected_key_status:")
    for key in expected:
        value = ckpt.get(key)
        if isinstance(value, dict):
            print(f"  - {key}: present, {len(value)} entries")
        else:
            print(f"  - {key}: {'present' if key in ckpt else 'missing'}")

    for step_key in ("step", "global_step", "epoch", "iteration"):
        if step_key in ckpt:
            print(f"{step_key}: {ckpt[step_key]}")

    for style_key in ("style_axes", "STYLE_AXES", "style_names"):
        if style_key in ckpt:
            print(f"{style_key}: {ckpt[style_key]}")
else:
    print("top_level_keys: not a dict checkpoint")
