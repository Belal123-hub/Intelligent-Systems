import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from urllib.request import urlopen
from uuid import uuid4

import mlflow
import yaml
from openai import AsyncOpenAI
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "data/pride_and_prejudice.txt"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SERVER = "http://127.0.0.1:8000"
TTFT_LIMIT = 2.0
TPOT_LIMIT = 0.100


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)


def server_metrics():
    with urlopen(f"{SERVER}/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    names = ("vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
    return [line for line in lines if any(line.startswith(name + " ") or line.startswith(name + "{") for name in names)]


class Prompts:
    def __init__(self, text, revision):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision)
        self.book = self.tokenizer.encode(text, add_special_tokens=False)

    def messages(self, requested_tokens, label, offset=0):
        prefix = f"{hashlib.sha256(label.encode()).hexdigest()[:20]}\nSummarize the following text in one sentence.\n\n"
        size = requested_tokens
        tried = set()
        while size > 0 and size not in tried:
            tried.add(size)
            excerpt = self.tokenizer.decode(self.book[offset:offset + size], clean_up_tokenization_spaces=False)
            messages = [{"role": "user", "content": prefix + excerpt}]
            actual = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False))
            if actual == requested_tokens:
                return messages
            size += requested_tokens - actual
        raise ValueError(f"Could not tokenize {label} to exactly {requested_tokens} tokens")


async def measure(client, messages, prompt_tokens, output_tokens, label):
    row = {"id": label, "requested_prompt_tokens": prompt_tokens, "requested_output_tokens": output_tokens}
    t0 = perf_counter()
    times, parts = [], []
    usage = finish = None
    try:
        stream = await client.chat.completions.create(
            model=MODEL, messages=messages, max_tokens=output_tokens, temperature=0,
            stream=True, stream_options={"include_usage": True}, extra_body={"ignore_eos": True},
        )
        async with stream:
            async for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.finish_reason is not None:
                        finish = choice.finish_reason
                    if choice.delta.content:
                        times.append(perf_counter() - t0)
                        parts.append(choice.delta.content)
        if usage is None or not times:
            raise ValueError("Missing usage or response text")
        count = usage.completion_tokens
        row.update(prompt_tokens=usage.prompt_tokens, output_tokens=count, cached_prompt_tokens=getattr(usage.prompt_tokens_details, "cached_tokens", None), ttft_s=times[0],
                   e2e_s=times[-1], tpot_s=(times[-1] - times[0]) / (count - 1) if count > 1 else None)
        if (usage.prompt_tokens, count) != (prompt_tokens, output_tokens):
            raise ValueError(f"Server returned {usage.prompt_tokens} input and {count} output tokens")
        row.update(status="ok", error=None)
    except Exception as error:
        row.update(status="error", error=f"{type(error).__name__}: {error}")
    row.update(text="".join(parts), finish_reason=finish, chunk_times_s=times,
               started_s=t0, completed_s=perf_counter())
    return row


def conditions(experiment):
    if experiment == "prompt":
        return [(f"prompt_{n}", n, 16, 1, 5) for n in (256, 1024, 4096, 8192, 16384)]
    if experiment == "cache":
        return [("prefix_cache", 8192, 16, 1, 3)]
    if experiment == "answer":
        return [(f"answer_{n}", 1024, n, 1, 5) for n in (16, 64, 256, 1024)]
    if experiment == "pilot":
        return [("pilot_4", 1024, 128, 4, 4)]
    return [(f"load_{n}", 1024, 128, n, max(16, 4 * n)) for n in (1, 2, 4, 8, 16, 32)]


async def run_conditions(experiment, prompts, path, run_id):
    client = AsyncOpenAI(base_url=f"{SERVER}/v1", api_key="unused", timeout=180, max_retries=0)
    try:
        with (path / "results.jsonl").open("w", encoding="utf-8") as results:
            for name, input_size, output_size, concurrency, count in conditions(experiment):
                warmup_id = f"{run_id}/{name}/warmup"
                warmup = await measure(client, prompts.messages(input_size, warmup_id), input_size, output_size, warmup_id)
                with (path / "warmups.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(warmup, ensure_ascii=False) + "\n")
                if warmup["status"] != "ok":
                    raise RuntimeError(warmup["error"])
                before = server_metrics() if experiment == "load" else []
                queue = asyncio.Queue()
                for index in range(count):
                    label = f"{run_id}/{name}/{index:04d}"
                    prompt_label = f"{run_id}/{name}/repeat" if experiment == "cache" else label
                    offset = index * 97 if experiment == "load" else 0
                    queue.put_nowait((label, prompts.messages(input_size, prompt_label, offset)))
                print(f"Running {name}: {count} requests, concurrency {concurrency}", flush=True)

                async def worker():
                    while not queue.empty():
                        label, messages = queue.get_nowait()
                        row = await measure(client, messages, input_size, output_size, label)
                        row.update(condition=name, concurrency=concurrency)
                        results.write(json.dumps(row, ensure_ascii=False) + "\n")
                        results.flush()

                await asyncio.gather(*(worker() for _ in range(concurrency)))
                if experiment == "load":
                    write_json(path / f"{name}_server_metrics.json", {"before": before, "after": server_metrics()})
    finally:
        await client.close()


def summarize(path):
    result = {}
    file = path / "results.jsonl"
    if not file.exists():
        return result
    for line in file.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        result.setdefault(row["condition"], []).append(row)
    summaries = {}
    for condition, rows in result.items():
        ok = [row for row in rows if row["status"] == "ok"]
        elapsed = max(row["completed_s"] for row in rows) - min(row["started_s"] for row in rows)
        ttfts = [row["ttft_s"] for row in ok]
        tpots = [row["tpot_s"] for row in ok if row["tpot_s"] is not None]
        good = sum(row["ttft_s"] <= TTFT_LIMIT and row["tpot_s"] is not None and row["tpot_s"] <= TPOT_LIMIT for row in ok)
        summary = {"count": len(rows), "errors": len(rows) - len(ok), "concurrency": rows[0]["concurrency"],
                   "prompt_tokens": rows[0]["requested_prompt_tokens"], "output_tokens": rows[0]["requested_output_tokens"],
                   "elapsed_s": elapsed, "tokens_per_s": sum(row["output_tokens"] for row in ok) / elapsed,
                   "requests_per_s": len(ok) / elapsed, "goodput_requests_per_s": good / elapsed, "goodput_fraction": good / len(rows)}
        if ttfts:
            summary.update(ttft_min_s=min(ttfts), ttft_median_s=median(ttfts), ttft_max_s=max(ttfts),
                           ttft_p90_s=percentile(ttfts, 0.9), e2e_median_s=median(row["e2e_s"] for row in ok),
                           fraction_ttft_within_limit=sum(value <= TTFT_LIMIT for value in ttfts) / len(rows),
                           prefill_estimate_tokens_per_s=rows[0]["requested_prompt_tokens"] / median(ttfts))
        if tpots:
            summary["tpot_median_s"] = median(tpots)
        if condition == "prefix_cache":
            summary["ttft_by_request_s"] = ttfts
            if len(ttfts) >= 2:
                summary["second_request_speedup"] = ttfts[0] / ttfts[1]
        summaries[condition] = summary
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("pilot", "prompt", "cache", "answer", "load"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if not args.debug and git("status", "--porcelain"):
        parser.error("Commit the code and configuration before a formal run.")
    content = BOOK.read_bytes()
    fingerprint = hashlib.sha256(content).hexdigest()
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    if fingerprint != manifest["pride_and_prejudice"]["sha256"]:
        raise ValueError("Book SHA-256 differs from data/manifest.json")
    text = re.split(r"(?im)^\*\*\* START OF .*?\*\*\*\s*$", content.decode("utf-8-sig"), maxsplit=1)[-1]
    run_id = f"lr1-{args.experiment}-{uuid4().hex[:10]}"
    path = args.output_root.resolve() / run_id
    path.mkdir(parents=True)
    print(f"Run directory: {path}", flush=True)
    config = vars(args) | {"output_root": str(args.output_root), "model": MODEL, "server": SERVER,
                            "temperature": 0, "ttft_limit_s": TTFT_LIMIT, "tpot_limit_s": TPOT_LIMIT}
    (path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    from importlib.metadata import version
    metadata = {"run_id": run_id, "lab": "lr1", "series": args.experiment, "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(), "commit": git("rev-parse", "HEAD"),
                "python": sys.version.split()[0], "packages": {name: version(name) for name in ("openai", "transformers", "mlflow")},
                "device": subprocess.getoutput("nvidia-smi --query-gpu=name --format=csv,noheader"),
                "input_sha256": fingerprint, "model": MODEL, "revision": args.revision, "weights_format": "FP16",
                "server": SERVER, "seed": 0}
    write_json(path / "metadata.json", metadata)
    mlflow.set_tracking_uri(f"sqlite:///{args.output_root.resolve() / 'mlflow.db'}")
    experiment_name = "lr1-debug" if args.debug else "lr1"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=(args.output_root.resolve() / "mlartifacts").as_uri())
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_id):
        mlflow.set_tags({key: metadata[key] for key in ("lab", "series", "commit", "model")})
        mlflow.log_params(config)
        try:
            prompts = Prompts(text, args.revision)
            asyncio.run(run_conditions(args.experiment, prompts, path, run_id))
            metrics = summarize(path)
            if any(value["errors"] for value in metrics.values()):
                raise RuntimeError("Some requests failed. See results.jsonl.")
            metadata["status"] = "completed"
        except Exception as error:
            metadata.update(status="failed", error=str(error))
            raise
        finally:
            metadata["ended_at"] = datetime.now(timezone.utc).isoformat()
            write_json(path / "metadata.json", metadata)
            metrics = summarize(path)
            write_json(path / "metrics.json", metrics)
            for name, summary in metrics.items():
                mlflow.log_metrics({f"{name}.{key}": value for key, value in summary.items() if isinstance(value, (int, float))})
            mlflow.log_artifacts(str(path))
            print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
