# Lab 1 reproduction

Use a Colab T4 runtime and clone the lab1 branch. Install requirements-lab1.txt in an isolated Python environment using the PyTorch CUDA 12.9 wheel index. From the repository root, download the text from the URL in data/manifest.json into the path listed there. The benchmark verifies its SHA-256.

Add the environment bin directory to PATH, then launch the server from it:

    vllm serve Qwen/Qwen3-4B-Instruct-2507 --revision <revision from endpoint.yaml> --dtype half --max-model-len 32768 --gpu-memory-utilization 0.94 --max-num-seqs 32 --enable-auto-tool-choice --tool-call-parser hermes --host 127.0.0.1 --port 8000

Wait until GET /v1/models lists Qwen. Set REVISION to the hash from endpoint.yaml. Run from a clean Git checkout:

    python src/lab1_benchmark.py prompt --revision "$REVISION"
    python src/lab1_benchmark.py cache --revision "$REVISION"
    python src/lab1_benchmark.py answer --revision "$REVISION"
    python src/lab1_benchmark.py load --revision "$REVISION"

The first prompt-length series uses distinct prefixes to avoid prefix reuse. The cache series sends the same prompt three times. Each condition has a separate warmup excluded from its summary. Results are stored as JSONL and summarized into metrics.json. MLflow stores parameters, metrics and run files. Optional --output-root points to an alternative artifact directory.
