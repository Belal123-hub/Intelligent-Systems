"""Smoke-test the config layer. Delete after Step 4 completes."""
import argparse
from pathlib import Path

from train import load_config, validate_config, apply_overrides

root = Path(__file__).resolve().parents[1]
cfg_path = root / "configs" / "base.yaml"

print("Loading:", cfg_path)
cfg = load_config(cfg_path)
print("Loaded OK. Keys:", sorted(cfg))

validate_config(cfg)
print("Validation OK")

# Test override
args = argparse.Namespace(seed=42, epochs=None)
cfg2 = apply_overrides(cfg, args)
assert cfg2["seed"] == 42, "seed override failed"
assert cfg["seed"] == 0, "original config was mutated!"

# Test bad lr rejection
bad = dict(cfg)
bad["train"] = dict(cfg["train"])
bad["train"]["lr"] = -1
try:
    validate_config(bad)
    raise SystemExit("ERROR: lr=-1 should have been rejected")
except ValueError as e:
    print("Correctly rejected lr=-1:", e)

print("All config-layer checks passed.")