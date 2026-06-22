import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_DIR = os.path.join(REPO_ROOT, "crowd_nav", "policy")

for path in (REPO_ROOT, POLICY_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from crowd_nav.policy.diffusion_CondUNetCFG import DiffusionConditionalUNet1DCFG
except Exception as exc:
    print("IMPORT_FAILED")
    print(f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    raise SystemExit(1)

print("IMPORT_OK")
print(DiffusionConditionalUNet1DCFG)
