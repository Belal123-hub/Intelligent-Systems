from time import perf_counter

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="unused",
    timeout=30,
    max_retries=0,
)

first_text = None
last_text = None
usage = None

started = perf_counter()
with client.chat.completions.create(
    model="Qwen/Qwen3-4B-Instruct-2507",
    messages=[
        {"role": "user", "content": "Why is the sky blue? Answer in one sentence."}
    ],
    temperature=0,
    max_tokens=64,
    stream=True,
    stream_options={"include_usage": True},
) as response:
    for chunk in response:
        if chunk.usage is not None:
            usage = chunk.usage

        piece = chunk.choices[0].delta.content if chunk.choices else None
        if piece:
            last_text = perf_counter()
            if first_text is None:
                first_text = last_text
            print(piece, end="", flush=True)

if first_text is None or last_text is None:
    raise RuntimeError("The server returned no answer text.")

if usage is None:
    raise RuntimeError("The server did not report token counts.")

tokens = usage.completion_tokens
print(f"\nOutput tokens: {tokens}")
print(f"Time to first token: {(first_text - started) * 1000:.1f} ms")
print(f"Time to last text: {(last_text - started) * 1000:.1f} ms")

if tokens > 1:
    tpot_ms = (last_text - first_text) * 1000 / (tokens - 1)
    print(f"Time per output token after the first: {tpot_ms:.1f} ms")
