"""Optuna search for Lab 0, §4.2.

Runs 40 trials with TPESampler(seed=42) and MedianPruner.
Each trial is a nested MLflow run under a parent run named "lr0_search".
Stores the study in sqlite:///optuna.db so it can be resumed.

Usage:
    python src/search.py              # run (or resume) the search
    python src/search.py --trials 5   # override the total number of trials
"""

import argparse
import sys
from pathlib import Path

import mlflow
import optuna
import yaml
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_and_split, sha256_of_file
from mlflow_setup import configure_mlflow
from train import train_epochs, validate_config


def load_space(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def suggest_from_space(trial: optuna.Trial, space: dict) -> dict:
    lr = trial.suggest_float("lr", space["lr"]["low"], space["lr"]["high"],
                             log=space["lr"]["log"])
    n_layers = trial.suggest_categorical("n_layers", space["n_layers"]["choices"])
    hidden = trial.suggest_int("hidden", space["hidden"]["low"],
                               space["hidden"]["high"],
                               log=space["hidden"]["log"])
    dropout = trial.suggest_float("dropout", space["dropout"]["low"],
                                  space["dropout"]["high"])
    wd = trial.suggest_float("weight_decay", space["weight_decay"]["low"],
                             space["weight_decay"]["high"],
                             log=space["weight_decay"]["log"])
    bs = trial.suggest_categorical("batch_size", space["batch_size"]["choices"])
    return {
        "lr": lr,
        "n_layers": n_layers,
        "hidden": hidden,
        "dropout": dropout,
        "weight_decay": wd,
        "batch_size": bs,
        "epochs": 30,
    }


def make_objective(X, y, train_idx, val_idx, project_root: Path):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_from_space(trial, load_space(project_root / "configs" / "search_space.yaml"))

        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.set_tag("trial_state", "running")

            def _progress(epoch, val_acc, row):
                mlflow.log_metrics(
                    {"val_acc": row["val_acc"],
                     "train_loss": row["train_loss"],
                     "val_loss": row["val_loss"]},
                    step=epoch,
                )
                trial.report(float(val_acc), step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            result = train_epochs(
                params=params, seed=0,
                X_train=X[train_idx], y_train=y[train_idx],
                X_val=X[val_idx], y_val=y[val_idx],
                num_threads=1,
                progress_cb=_progress,
            )
            last = result["history"][-1]
            mlflow.set_tag("trial_state", "complete")
            mlflow.log_metric("final_val_acc", float(last["val_acc"]))
            return float(last["val_acc"])

    return objective


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]

    configure_mlflow(project_root)
    mlflow.set_experiment("lr0")

    X, y, train_idx, val_idx, _ = load_and_split()

    study = optuna.create_study(
        study_name="lr0_search",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=5,
        ),
    )

    objective = make_objective(X, y, train_idx, val_idx, project_root)

    with mlflow.start_run(run_name="lr0_search"):
        mlflow.set_tag("work", "lr0")
        mlflow.set_tag("series", "search")
        study.optimize(
            objective,
            n_trials=args.trials,
            callbacks=[MaxTrialsCallback(args.trials, states=None)],
        )

    # --- summary
    n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == TrialState.PRUNED])
    n_fail = len([t for t in study.trials if t.state == TrialState.FAIL])
    print(f"[search] trials: complete={n_complete} pruned={n_pruned} fail={n_fail}")
    print(f"[search] best value: {study.best_value:.6f}")
    print(f"[search] best trial : {study.best_trial.number}")
    print("[search] best params:")
    for k, v in study.best_params.items():
        print(f"    {k} = {v}")

    # --- write configs/best.yaml
    best = dict(study.best_params)
    best["epochs"] = 30
    cfg_out = {
        "experiment": "lr0",
        "series": "best",
        "split_seed": 0,
        "seed": 0,
        "model": {
            "n_layers": int(best["n_layers"]),
            "hidden": int(best["hidden"]),
            "dropout": float(best["dropout"]),
        },
        "train": {
            "lr": float(best["lr"]),
            "weight_decay": float(best["weight_decay"]),
            "batch_size": int(best["batch_size"]),
            "epochs": int(best["epochs"]),
        },
        "runtime": {"device": "cpu", "num_threads": 1},
    }

    best_path = project_root / "configs" / "best.yaml"
    with open(best_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_out, f, sort_keys=False)
    print(f"[search] wrote {best_path}")

    # Sanity: the produced config must validate
    validate_config(cfg_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())