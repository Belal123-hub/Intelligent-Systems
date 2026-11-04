"""Training loop for the Lab 0 MLP.

This file will grow in Step 4 to become the full run harness.
Right now it only exposes the pure training function that takes a
config dict and returns (history, val_pred, test_pred).

No file I/O, no MLflow, no git checks yet — those come in Step 4.
"""

import math
import time
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import MLP
import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import yaml


def _set_seed(seed: int) -> None:
    """Fix every source of randomness we control, for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _make_loader(X, y, batch_size, seed, shuffle):
    """Deterministic DataLoader: its own seeded generator, num_workers=0."""
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=g,
    )


def _evaluate(model, X, y, batch_size=256):
    """Return (accuracy, mean_loss, y_pred) on the given set."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()

    correct = 0
    total_loss = 0.0
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size]
            yb = y_t[i:i + batch_size]
            logits = model(xb)
            total_loss += loss_fn(logits, yb).item()
            p = logits.argmax(dim=1)
            preds.append(p.cpu().numpy())
            correct += (p == yb).sum().item()

    y_pred = np.concatenate(preds, axis=0)
    return correct / len(y_t), total_loss / len(y_t), y_pred


def train_epochs(
    params: dict,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_threads: int = 1,
    progress_cb=None,
):
    """Train one MLP for `params['epochs']` epochs.

    `params` keys (all required):
      n_layers, hidden, dropout, lr, weight_decay, batch_size, epochs

    `progress_cb(epoch, val_acc)` is called after each epoch.
    It is used later by Optuna's pruner. If it returns True, we stop early.

    Returns a dict:
      history: list of dicts with epoch, train_loss, val_loss, val_acc, epoch_time_s
      model:   the trained MLP (in eval mode)
    """
    required = {"n_layers", "hidden", "dropout", "lr", "weight_decay", "batch_size", "epochs"}
    missing = required - set(params)
    if missing:
        raise ValueError(f"missing params: {sorted(missing)}")

    lr = float(params["lr"])
    weight_decay = float(params["weight_decay"])
    batch_size = int(params["batch_size"])
    epochs = int(params["epochs"])
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    torch.set_num_threads(num_threads)
    _set_seed(seed)

    model = MLP(
        n_layers=int(params["n_layers"]),
        hidden=int(params["hidden"]),
        dropout=float(params["dropout"]),
    )
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    loader = _make_loader(X_train, y_train, batch_size, seed, shuffle=True)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        n_seen = 0
        for xb, yb in loader:
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            total_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = total_loss / n_seen

        val_acc, val_loss, _ = _evaluate(model, X_val, y_val)
        epoch_time = time.perf_counter() - t0

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "epoch_time_s": float(epoch_time),
        })

        if progress_cb is not None:
            stop = progress_cb(epoch, float(val_acc), history[-1])
            if stop:
                break

    model.eval()
    return {"history": history, "model": model}


def evaluate_model(model, X, y):
    """Public helper: returns (acc, loss, y_pred). Used by the harness."""
    return _evaluate(model, X, y)

# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------

REQUIRED_TOP_KEYS = {"experiment", "series", "split_seed", "seed", "model", "train", "runtime"}
REQUIRED_MODEL_KEYS = {"n_layers", "hidden", "dropout"}
REQUIRED_TRAIN_KEYS = {"lr", "weight_decay", "batch_size", "epochs"}
REQUIRED_RUNTIME_KEYS = {"device", "num_threads"}


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping, got {type(cfg).__name__}")
    return cfg


def validate_config(cfg: dict) -> None:
    """Fail fast if the config is missing keys or contains nonsense."""
    missing = REQUIRED_TOP_KEYS - set(cfg)
    if missing:
        raise ValueError(f"config missing keys: {sorted(missing)}")
    for section, keys in [
        ("model", REQUIRED_MODEL_KEYS),
        ("train", REQUIRED_TRAIN_KEYS),
        ("runtime", REQUIRED_RUNTIME_KEYS),
    ]:
        sub = cfg.get(section)
        if not isinstance(sub, dict):
            raise ValueError(f"config['{section}'] must be a mapping")
        miss = keys - set(sub)
        if miss:
            raise ValueError(f"config['{section}'] missing keys: {sorted(miss)}")

    if float(cfg["train"]["lr"]) <= 0:
        raise ValueError(f"train.lr must be > 0, got {cfg['train']['lr']}")
    if float(cfg["train"]["weight_decay"]) < 0:
        raise ValueError(f"train.weight_decay must be >= 0")
    if int(cfg["train"]["batch_size"]) < 1:
        raise ValueError(f"train.batch_size must be >= 1")
    if int(cfg["train"]["epochs"]) < 1:
        raise ValueError(f"train.epochs must be >= 1")
    if int(cfg["model"]["n_layers"]) < 1:
        raise ValueError(f"model.n_layers must be >= 1")
    if int(cfg["model"]["hidden"]) < 1:
        raise ValueError(f"model.hidden must be >= 1")
    if not (0.0 <= float(cfg["model"]["dropout"]) <= 1.0):
        raise ValueError(f"model.dropout must be in [0,1]")
    if cfg["runtime"]["device"] not in ("cpu", "cuda"):
        raise ValueError(f"runtime.device must be 'cpu' or 'cuda'")
    if int(cfg["runtime"]["num_threads"]) < 1:
        raise ValueError(f"runtime.num_threads must be >= 1")


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """CLI overrides only what it sets; everything else comes from the file."""
    import copy
    cfg = copy.deepcopy(cfg)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    return cfg

# ---------------------------------------------------------------------------
# Git + run directory
# ---------------------------------------------------------------------------

def git_commit_hash() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def git_is_clean() -> bool:
    out = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout
    return out.strip() == ""


def new_run_dir(root: Path) -> tuple[str, Path]:
    """Create runs/<run_id>/ with a sortable, unique name.

    Format: <YYYYMMDD-HHMMSS>-<6-char-uid>. Sortable by time, collision-safe.
    """
    import datetime as _dt
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    uid = uuid.uuid4().hex[:6]
    run_id = f"{stamp}-{uid}"
    path = root / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return run_id, path


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------------------
# Run directory contents: metadata, history, status
# ---------------------------------------------------------------------------

import platform

def write_metadata(
    run_dir: Path,
    run_id: str,
    series: str,
    seed: int,
    split_sha256: str,
    cfg: dict,
    status: str = "running",
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Write (or update) runs/<run_id>/metadata.json.

    Required by Lab 0 §3.3 and Rules §4.2:
      run_id, work and series, status, start and end times, commit,
      Python and library versions, device, seed, SHA-256 of input data,
      model information.
    """
    import torch, sklearn, mlflow, optuna, numpy

    meta = {
        "run_id": run_id,
        "experiment": cfg["experiment"],
        "series": series,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "error": error,
        "commit": git_commit_hash(),
        "python_version": platform.python_version(),
        "versions": {
            "torch": torch.__version__,
            "scikit-learn": sklearn.__version__,
            "mlflow": mlflow.__version__,
            "optuna": optuna.__version__,
            "numpy": numpy.__version__,
        },
        "device": cfg["runtime"]["device"],
        "num_threads": int(cfg["runtime"]["num_threads"]),
        "seed": int(seed),
        "split_sha256": split_sha256,
        "model": {
            "n_layers": int(cfg["model"]["n_layers"]),
            "hidden": int(cfg["model"]["hidden"]),
            "dropout": float(cfg["model"]["dropout"]),
        },
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def append_history_csv(run_dir: Path, row: dict) -> None:
    """Append one epoch row to history.csv.

    Lab 0 §3.3: 'history.csv rows are appended after each epoch.'
    We do this immediately, not at the end, so a crash keeps partial data.
    """
    path = run_dir / "history.csv"
    header = ["epoch", "train_loss", "val_loss", "val_acc", "epoch_time_s"]
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new_file:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(row[k]) for k in header) + "\n")


# ---------------------------------------------------------------------------
# Predictions + metrics (recomputed from disk, not from memory)
# ---------------------------------------------------------------------------

def write_predictions_csv(
    run_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    split: str,
    idx: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Append one row per example to predictions.csv.

    Columns: split, index, y_true, y_pred.
    Appended, not overwritten, so val and test can coexist in one file.
    """
    path = run_dir / "predictions.csv"
    new_file = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new_file:
            f.write("split,index,y_true,y_pred\n")
        for i, p in zip(idx.tolist(), y_pred.tolist()):
            f.write(f"{split},{int(i)},{int(y[i])},{int(p)}\n")


def recompute_metrics(run_dir: Path) -> dict:
    """Recompute val_acc, val_loss, test_acc from files on disk.

    This is the core reproducibility guarantee of Lab 0:
    numbers in metrics.json must be reproducible by anyone reading
    predictions.csv + history.csv, without rerunning training.
    """
    import csv

    metrics: dict = {}

    # From history.csv: val_loss at the final recorded epoch.
    with open(run_dir / "history.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows:
        metrics["val_loss"] = float(rows[-1]["val_loss"])
        metrics["epochs_run"] = int(rows[-1]["epoch"])

    # From predictions.csv: accuracy per split.
    splits: dict = {}
    with open(run_dir / "predictions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            splits.setdefault(r["split"], []).append(int(r["y_true"]) == int(r["y_pred"]))

    for name, results in splits.items():
        acc = sum(results) / len(results)
        if name == "val":
            metrics["val_acc"] = acc
        elif name == "test":
            metrics["test_acc"] = acc
        metrics[f"n_{name}"] = len(results)

    return metrics

# ---------------------------------------------------------------------------
# Main harness: python src/train.py --config configs/base.yaml [--seed N] [--eval-test]
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Lab 0 run harness")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--seed", type=int, default=None,
                   help="override the seed from the config")
    p.add_argument("--epochs", type=int, default=None,
                   help="override train.epochs from the config")
    p.add_argument("--eval-test", action="store_true",
                   help="after training, evaluate on the test split once")
    p.add_argument("--allow-dirty", action="store_true",
                   help="bypass the git-clean check (debug runs only)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    import datetime as _dt
    import traceback

    args = _parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]

    # 1. Config
    cfg = apply_overrides(load_config(args.config), args)
    validate_config(cfg)

    # 2. Dirty git check (Rules §4.3)
    if not args.allow_dirty and not git_is_clean():
        print("ERROR: working tree is dirty. Commit or stash first, "
              "or pass --allow-dirty for a debug run.", file=sys.stderr)
        return 2

    # 3. Run directory
    run_id, run_dir = new_run_dir(project_root)
    started_at = _dt.datetime.now().isoformat(timespec="seconds")
    print(f"[harness] run_id = {run_id}")
    print(f"[harness] run_dir = {run_dir}")

    # 4. Persist the final config (after overrides).
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # 5. Data
    from data import load_and_split, sha256_of_file
    X, y, train_idx, val_idx, test_idx = load_and_split()
    split_sha = sha256_of_file(project_root / "data" / "split.json")

    # 6. Initial metadata (status=running)
    write_metadata(
        run_dir=run_dir, run_id=run_id, series=cfg["series"],
        seed=int(cfg["seed"]), split_sha256=split_sha, cfg=cfg,
        status="running", started_at=started_at,
    )

    # 7. MLflow
    from mlflow_setup import configure_mlflow
    configure_mlflow(project_root)
    import mlflow

    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({
            "experiment": cfg["experiment"],
            "series": cfg["series"],
            "seed": int(cfg["seed"]),
            "split_seed": int(cfg["split_seed"]),
            "n_layers": int(cfg["model"]["n_layers"]),
            "hidden": int(cfg["model"]["hidden"]),
            "dropout": float(cfg["model"]["dropout"]),
            "lr": float(cfg["train"]["lr"]),
            "weight_decay": float(cfg["train"]["weight_decay"]),
            "batch_size": int(cfg["train"]["batch_size"]),
            "epochs": int(cfg["train"]["epochs"]),
        })
        mlflow.set_tags({
            "work": "lr0",
            "series": cfg["series"],
            "commit": git_commit_hash(),
            "device": cfg["runtime"]["device"],
        })

        # 8. Train
        params = {
            "n_layers": int(cfg["model"]["n_layers"]),
            "hidden": int(cfg["model"]["hidden"]),
            "dropout": float(cfg["model"]["dropout"]),
            "lr": float(cfg["train"]["lr"]),
            "weight_decay": float(cfg["train"]["weight_decay"]),
            "batch_size": int(cfg["train"]["batch_size"]),
            "epochs": int(cfg["train"]["epochs"]),
        }

        def _progress(epoch, val_acc, row):
            append_history_csv(run_dir, row)
            mlflow.log_metrics(
                {"val_acc": row["val_acc"],
                 "train_loss": row["train_loss"],
                 "val_loss": row["val_loss"]},
                step=epoch,
            )
            return False

        try:
            result = train_epochs(
                params=params, seed=int(cfg["seed"]),
                X_train=X[train_idx], y_train=y[train_idx],
                X_val=X[val_idx], y_val=y[val_idx],
                num_threads=int(cfg["runtime"]["num_threads"]),
                progress_cb=_progress,
            )
            model = result["model"]

            # 9. Predictions
            _, _, val_pred = evaluate_model(model, X[val_idx], y[val_idx])
            write_predictions_csv(run_dir, X, y, "val", val_idx, val_pred)

            if args.eval_test:
                _, _, test_pred = evaluate_model(model, X[test_idx], y[test_idx])
                write_predictions_csv(run_dir, X, y, "test", test_idx, test_pred)

            # 10. Recompute metrics from disk
            metrics = recompute_metrics(run_dir)
            with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            mlflow.log_metrics({k: v for k, v in metrics.items()
                                if isinstance(v, (int, float))})

            # 11. Artifacts
            for name in ("config.yaml", "metadata.json", "history.csv",
                         "predictions.csv", "metrics.json"):
                p = run_dir / name
                if p.exists():
                    mlflow.log_artifact(str(p))

            # 12. Finalize
            ended_at = _dt.datetime.now().isoformat(timespec="seconds")
            write_metadata(
                run_dir=run_dir, run_id=run_id, series=cfg["series"],
                seed=int(cfg["seed"]), split_sha256=split_sha, cfg=cfg,
                status="completed", started_at=started_at, ended_at=ended_at,
            )
            mlflow.set_tag("status", "completed")
            print(f"[harness] done. metrics = {metrics}")
            return 0

        except Exception as e:
            tb = traceback.format_exc()
            ended_at = _dt.datetime.now().isoformat(timespec="seconds")
            write_metadata(
                run_dir=run_dir, run_id=run_id, series=cfg["series"],
                seed=int(cfg["seed"]), split_sha256=split_sha, cfg=cfg,
                status="failed", started_at=started_at, ended_at=ended_at,
                error=tb,
            )
            mlflow.set_tag("status", "failed")
            print("ERROR:", e, file=sys.stderr)
            print(tb, file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())