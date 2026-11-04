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
            stop = progress_cb(epoch, float(val_acc))
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