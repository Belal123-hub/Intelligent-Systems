# Lab 1 predictions

1. TTFT should increase with prompt length because Qwen processes more input before emitting its first token. I expect faster growth at the longest lengths.
2. At a fixed prompt length, response time should increase roughly linearly with output length. The pilot measured about 36 ms per output token after the first.
3. Throughput should rise initially with concurrency, while individual TTFT rises. I predict 16 concurrent users is the highest load meeting both limits: at least 90% of requests start within 2 seconds and median TPOT is at most 100 ms.
