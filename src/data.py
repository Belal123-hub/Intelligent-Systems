"""Data loading and splitting for the Intelligent Systems course.

Source: scikit-learn load_digits (8x8 handwritten digits, 1797 samples).
Rule: stratified 60/20/20 -> actually 20% test, then 25% of the rest as val,
which gives 1077 / 360 / 360.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


SPLIT_SEED = 0
TEST_SIZE = 0.20
VAL_SIZE_OF_REST = 0.25


def load_and_split():
    """Return (X, y, train_idx, val_idx, test_idx) using the course split rule."""
    X, y = load_digits(return_X_y=True)
    idx = np.arange(len(y))

    rest, test = train_test_split(
        idx,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=SPLIT_SEED,
    )
    train, val = train_test_split(
        rest,
        test_size=VAL_SIZE_OF_REST,
        stratify=y[rest],
        random_state=SPLIT_SEED,
    )
    return X, y, train, val, test


def sanity_checks(X, y, train, val, test):
    """Fail loudly if the split is wrong. These checks come from Lab 0 §3.1.4."""
    assert X.shape == (1797, 64), f"unexpected X shape: {X.shape}"
    assert X.min() == 0 and X.max() == 16, "feature range should be 0..16"

    n_total = len(train) + len(val) + len(test)
    assert n_total == 1797, f"split sum {n_total} != 1797"

    # Sizes we expect from the course rule
    assert len(train) == 1077, f"train size {len(train)} != 1077"
    assert len(val) == 360, f"val size {len(val)} != 360"
    assert len(test) == 360, f"test size {len(test)} != 360"

    # No overlap between parts
    sets = [set(train.tolist()), set(val.tolist()), set(test.tolist())]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])

    # Each part contains all 10 classes
    for name, part in [("train", train), ("val", val), ("test", test)]:
        classes = sorted(set(y[part].tolist()))
        assert classes == list(range(10)), f"{name} classes: {classes}"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_split_and_manifest(root: Path):
    """Create data/split.json and data/manifest.json (or extend manifest)."""
    X, y, train, val, test = load_and_split()
    sanity_checks(X, y, train, val, test)

    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    split_path = data_dir / "split.json"
    split_payload = {
        "split_seed": SPLIT_SEED,
        "test_size": TEST_SIZE,
        "val_size_of_rest": VAL_SIZE_OF_REST,
        "train": train.tolist(),
        "val": val.tolist(),
        "test": test.tolist(),
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)

    split_sha = sha256_of_file(split_path)

    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {}

    manifest["load_digits"] = {
        "source": "scikit-learn",
        "sklearn_version": sklearn.__version__,
        "loader": "sklearn.datasets.load_digits(return_X_y=True)",
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_range": [int(X.min()), int(X.max())],
        "split_rule": (
            "stratified by class; test_size=0.20 of all; then val_size=0.25 "
            "of the remainder; both with random_state=SPLIT_SEED"
        ),
        "split_seed": SPLIT_SEED,
        "sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "split_file": "data/split.json",
        "split_sha256": split_sha,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Wrote:", split_path)
    print("Wrote:", manifest_path)
    print("SHA-256 of split.json:", split_sha)
    print("Sizes:",
          {"train": len(train), "val": len(val), "test": len(test)})


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    write_split_and_manifest(project_root)