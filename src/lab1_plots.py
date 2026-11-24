import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    runs = {}
    for path in sorted(root.glob("lr1-*"), key=lambda item: item.stat().st_mtime):
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata["status"] == "completed":
            runs[metadata["series"]] = (path.name, json.loads((path / "metrics.json").read_text()))
    output = root / "report"
    output.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("lr1")
    by_name = {run.info.run_name: run.info.run_id for run in client.search_runs([experiment.experiment_id])}

    def save(fig, filename, series):
        path = output / filename
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        client.log_artifact(by_name[runs[series][0]], str(path), "report")

    prompt = sorted(runs["prompt"][1].values(), key=lambda row: row["prompt_tokens"])
    x = [row["prompt_tokens"] for row in prompt]
    y = np.array([row["ttft_median_s"] for row in prompt])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(x, y, yerr=[y - [row["ttft_min_s"] for row in prompt],
                            [row["ttft_max_s"] for row in prompt] - y], marker="o", capsize=4)
    ax.set(xlabel="Prompt length (tokens)", ylabel="Time to first token (s)", title="Prompt length and time to first token")
    ax.grid(alpha=0.25)
    save(fig, "prompt_length.png", "prompt")

    answer = sorted(runs["answer"][1].values(), key=lambda row: row["output_tokens"])
    x = np.array([row["output_tokens"] for row in answer])
    y = np.array([row["e2e_median_s"] for row in answer])
    slope, intercept = np.polyfit(x, y, 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, label="Measured medians")
    ax.plot(x, intercept + x * slope, label=f"Fit: {intercept:.3f} + {slope:.4f} × tokens")
    ax.set(xlabel="Output length (tokens)", ylabel="Total response time (s)", title="Output length and response time")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "answer_length.png", "answer")

    load = sorted(runs["load"][1].values(), key=lambda row: row["concurrency"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["tokens_per_s"] for row in load], [row["ttft_p90_s"] for row in load], marker="o")
    for row in load:
        ax.annotate(str(row["concurrency"]), (row["tokens_per_s"], row["ttft_p90_s"]), xytext=(5, 4), textcoords="offset points")
    ax.axhline(2, color="tab:red", linestyle="--", label="2 s TTFT target")
    ax.set(xlabel="Output throughput (tokens/s)", ylabel="90th-percentile TTFT (s)", title="Throughput and latency by concurrency")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "server_load.png", "load")

    fields = ("concurrency", "count", "errors", "tokens_per_s", "requests_per_s", "ttft_median_s", "ttft_p90_s",
              "tpot_median_s", "fraction_ttft_within_limit", "goodput_requests_per_s", "goodput_fraction")
    with (output / "load_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(load)
    client.log_artifact(by_name[runs["load"][0]], str(output / "load_table.csv"), "report")
    qualifying = [row["concurrency"] for row in load if row["fraction_ttft_within_limit"] >= 0.9 and row["tpot_median_s"] <= 0.1]
    summary = {"answer_fit_intercept_s": intercept, "answer_fit_slope_s_per_token": slope,
               "highest_qualifying_concurrency": max(qualifying, default=None),
               "runs": {series: run_id for series, (run_id, _) in runs.items()}}
    (output / "report_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
