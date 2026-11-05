"""Analysis script for Lab 0.

Reads runs/ and optuna.db, prints summary tables, saves a plot.

Usage:
    python src/analyze_lr0.py

Outputs go to docs/analysis/ (gitignored). Run after Experiments 1–3.
"""

import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
OUT_DIR = PROJECT_ROOT / "docs" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_runs():
    """Return a list of dicts: one dict per run directory."""
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        metrics_path = d / "metrics.json"
        config_path = d / "config.yaml"
        if not (meta_path.exists() and metrics_path.exists() and config_path.exists()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        runs.append({
            "run_id": d.name,
            "dir": d,
            "series": meta.get("series"),
            "seed": meta.get("seed"),
            "status": meta.get("status"),
            "model": meta.get("model"),
            "metrics": metrics,
        })
    return runs


def print_table(title, rows, headers):
    print(f"\n=== {title} ===")
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------

def experiment1(runs):
    base = [r for r in runs if r["series"] == "base" and r["status"] == "completed"]
    base.sort(key=lambda r: r["seed"])

    rows = []
    vals = []
    for r in base:
        acc = r["metrics"].get("val_acc")
        rows.append([r["seed"], r["run_id"], f"{acc:.6f}" if acc is not None else "—"])
        if acc is not None:
            vals.append(acc)
    print_table("Experiment 1 — base config, val_acc by seed", rows,
                ["seed", "run_id", "val_acc"])

    if vals:
        print(f"\nval_acc min = {min(vals):.6f}")
        print(f"val_acc median = {float(np.median(vals)):.6f}")
        print(f"val_acc max = {max(vals):.6f}")
        print(f"spread (max-min) = {(max(vals) - min(vals)) * 100:.2f} p.p.")


def experiment1_pairwise(runs, seed_a, seed_b):
    a = next((r for r in runs if r["series"] == "base" and r["seed"] == seed_a), None)
    b = next((r for r in runs if r["series"] == "base" and r["seed"] == seed_b), None)
    if not a or not b:
        print(f"\n[Experiment 1 pairwise] one of seeds {seed_a}/{seed_b} not found")
        return

    preds = {}
    for r, tag in ((a, "A"), (b, "B")):
        path = r["dir"] / "predictions.csv"
        correct = {}
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] != "val":
                    continue
                correct[int(row["index"])] = int(row["y_true"]) == int(row["y_pred"])
        preds[tag] = correct

    common = set(preds["A"]) & set(preds["B"])
    a_only = sum(1 for i in common if preds["A"][i] and not preds["B"][i])
    b_only = sum(1 for i in common if preds["B"][i] and not preds["A"][i])
    both = sum(1 for i in common if preds["A"][i] and preds["B"][i])
    neither = sum(1 for i in common if not preds["A"][i] and not preds["B"][i])

    print(f"\n=== Experiment 1 pairwise (seed {seed_a} vs seed {seed_b}, val) ===")
    print(f"both correct:           {both}")
    print(f"only seed {seed_a} correct: {a_only}")
    print(f"only seed {seed_b} correct: {b_only}")
    print(f"both wrong:             {neither}")
    print(f"total val examples:     {len(common)}")


# ---------------------------------------------------------------------------
# Experiment 2
# ---------------------------------------------------------------------------

def experiment2():
    import optuna
    from optuna.trial import TrialState

    db = f"sqlite:///{(PROJECT_ROOT / 'optuna.db').as_posix()}"
    try:
        study = optuna.load_study(study_name="lr0_search", storage=db)
    except Exception as e:
        print(f"\n[Experiment 2] could not load study: {e}")
        return

    states = {s: 0 for s in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)}
    for t in study.trials:
        states[t.state] = states.get(t.state, 0) + 1

    print("\n=== Experiment 2 — trials by state ===")
    for k, v in states.items():
        print(f"{k.name:>10}: {v}")
    print(f"total:     {sum(states.values())}")

    print(f"\nbest trial: {study.best_trial.number}")
    print(f"best value: {study.best_value:.6f}")
    print("best params:")
    for k, v in study.best_params.items():
        print(f"  {k} = {v}")

    # Plot best-so-far
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        values = []
        best_so_far = []
        current_best = -1.0
        for t in study.trials:
            if t.state == TrialState.COMPLETE and t.value is not None:
                values.append(t.number)
                current_best = max(current_best, t.value)
                best_so_far.append(current_best)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(values, best_so_far, marker="o", markersize=3, linewidth=1)
        ax.set_xlabel("trial number")
        ax.set_ylabel("best val_acc so far")
        ax.set_title("Experiment 2 — best-so-far val_acc")
        ax.grid(alpha=0.3)
        out = OUT_DIR / "exp2_best_so_far.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"\nplot saved to {out}")
    except Exception as e:
        print(f"\n[Experiment 2] plot skipped: {e}")


# ---------------------------------------------------------------------------
# Experiment 3
# ---------------------------------------------------------------------------

def experiment3(runs):
    base = {r["seed"]: r for r in runs
            if r["series"] == "base" and r["status"] == "completed"
            and "test_acc" in r["metrics"]}
    best = {r["seed"]: r for r in runs
            if r["series"] == "best" and r["status"] == "completed"
            and "test_acc" in r["metrics"]}

    seeds = sorted(set(base) & set(best))
    if not seeds:
        print("\n[Experiment 3] no matching seeds with test_acc")
        return

    rows = []
    dv, dt = [], []
    base_correct = 0
    best_correct = 0
    for s in seeds:
        bv = base[s]["metrics"]["val_acc"]
        bt = base[s]["metrics"]["test_acc"]
        kv = best[s]["metrics"]["val_acc"]
        kt = best[s]["metrics"]["test_acc"]
        dv_pp = (kv - bv) * 100
        dt_pp = (kt - bt) * 100
        dv.append(dv_pp)
        dt.append(dt_pp)
        rows.append([s, f"{bv:.5f}", f"{bt:.5f}", f"{kv:.5f}", f"{kt:.5f}",
                     f"{dv_pp:+.2f}", f"{dt_pp:+.2f}"])
        base_correct += round(bt * 360)
        best_correct += round(kt * 360)

    print_table("Experiment 3 — base vs best on seeds " + str(seeds),
                rows,
                ["seed", "base_val", "base_test", "best_val", "best_test",
                 "Δval (p.p.)", "Δtest (p.p.)"])

    print(f"\nmedian Δval = {float(np.median(dv)):+.2f} p.p.")
    print(f"median Δtest = {float(np.median(dt)):+.2f} p.p.")
    print(f"range Δval = {min(dv):+.2f} … {max(dv):+.2f} p.p.")
    print(f"range Δtest = {min(dt):+.2f} … {max(dt):+.2f} p.p.")
    print(f"\nbase correct on test (5 seeds × 360): {base_correct}")
    print(f"best correct on test (5 seeds × 360): {best_correct}")
    print(f"difference: {best_correct - base_correct:+d} images")


# ---------------------------------------------------------------------------

def main():
    runs = load_all_runs()
    print(f"Loaded {len(runs)} runs from {RUNS_DIR}")

    experiment1(runs)
    experiment1_pairwise(runs, seed_a=1, seed_b=5)
    experiment2()
    experiment3(runs)


if __name__ == "__main__":
    main()