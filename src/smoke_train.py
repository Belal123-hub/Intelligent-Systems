"""Throwaway smoke test: train the base config once and print val accuracy.
We will delete this file in Step 4 when the real harness exists.
"""

import json
from pathlib import Path

import numpy as np

from data import load_and_split
from train import train_epochs, evaluate_model


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]

    X, y, train_idx, val_idx, test_idx = load_and_split()

    params = {
        "n_layers": 1,
        "hidden": 64,
        "dropout": 0.0,
        "lr": 0.001,
        "weight_decay": 0.0,
        "batch_size": 64,
        "epochs": 30,
    }

    result = train_epochs(
        params=params,
        seed=0,
        X_train=X[train_idx], y_train=y[train_idx],
        X_val=X[val_idx],     y_val=y[val_idx],
        num_threads=1,
        progress_cb=None,
    )

    model = result["model"]
    val_acc, val_loss, val_pred = evaluate_model(model, X[val_idx], y[val_idx])

    print("Epochs recorded:", len(result["history"]))
    print("First epoch:", result["history"][0])
    print("Last epoch :", result["history"][-1])
    print(f"Final val_acc = {val_acc:.4f}, val_loss = {val_loss:.4f}")

    # Expected range from Lab 0 §3.4 self-check:
    # base config val_acc on CPU is roughly 0.95-0.96.
    assert 0.90 <= val_acc <= 0.99, f"val_acc {val_acc} is suspicious"
    print("OK")