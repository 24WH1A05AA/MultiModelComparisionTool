import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load OPENROUTER_API_KEY from .env into os.environ
load_dotenv()

# All free-tier models on OpenRouter (name → OpenRouter ID)
MODELS = [
    ("Qwen3 Next 80B",             "qwen/qwen3-next-80b-a3b-instruct:free"),
    ("Gemma 4 31B",                "google/gemma-4-31b-it:free"),
    ("GPT-OSS 120B",               "openai/gpt-oss-120b:free"),
    ("GPT-OSS 20B",                "openai/gpt-oss-20b:free"),
    ("NVIDIA Nemotron 3 Nano 30B", "nvidia/nemotron-3-nano-30b-a3b:free"),
    ("NVIDIA Nemotron 3 Nano Omni","nvidia/nemotron-3-nano-omni:free"),
    ("NVIDIA Nemotron Nano 9B V2", "nvidia/nemotron-nano-9b-v2:free"),
    ("Mistral Small 3.1 24B",      "mistralai/mistral-small-3.1-24b-instruct:free"),
    ("GLM 4.5 Air",                "z-ai/glm-4.5-air:free"),
]

# USD per million tokens for each model ID; all are free-tier → $0.00
PRICES = {
    "qwen/qwen3-next-80b-a3b-instruct:free":         {"in": 0.0, "out": 0.0},
    "google/gemma-4-31b-it:free":                    {"in": 0.0, "out": 0.0},
    "openai/gpt-oss-120b:free":                      {"in": 0.0, "out": 0.0},
    "openai/gpt-oss-20b:free":                       {"in": 0.0, "out": 0.0},
    "nvidia/nemotron-3-nano-30b-a3b:free":           {"in": 0.0, "out": 0.0},
    "nvidia/nemotron-3-nano-omni:free":              {"in": 0.0, "out": 0.0},
    "nvidia/nemotron-nano-9b-v2:free":               {"in": 0.0, "out": 0.0},
    "mistralai/mistral-small-3.1-24b-instruct:free": {"in": 0.0, "out": 0.0},
    "z-ai/glm-4.5-air:free":                        {"in": 0.0, "out": 0.0},
}

TIMEOUT_SECONDS = 30   # per-request wall-clock limit
PREVIEW_CHARS   = 120  # characters shown in the summary table

# The question we want answered
QUESTION = "What is the capital of France, and what is it famous for?"

# Point the OpenAI client at OpenRouter's API base
client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    timeout=TIMEOUT_SECONDS,
)


def ask(question, model):
    """Send question to model; return (answer, latency_s, in_tokens, out_tokens, cost_usd)."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
    )
    latency = time.perf_counter() - start

    answer  = response.choices[0].message.content
    in_tok  = response.usage.prompt_tokens
    out_tok = response.usage.completion_tokens
    price   = PRICES.get(model, {"in": 0.0, "out": 0.0})
    cost    = (in_tok * price["in"] + out_tok * price["out"]) / 1_000_000

    return answer, latency, in_tok, out_tok, cost


def print_table(rows):
    """
    rows: list of (name, preview, latency_str, cost_str)
    Prints a fixed-width table with column headers.
    """
    COL = {"name": 26, "preview": 52, "latency": 10, "cost": 12}
    SEP = "  "

    header = (
        f"{'Model':<{COL['name']}}{SEP}"
        f"{'Answer preview':<{COL['preview']}}{SEP}"
        f"{'Latency':>{COL['latency']}}{SEP}"
        f"{'Cost (USD)':>{COL['cost']}}"
    )
    rule = "-" * len(header)

    print(rule)
    print(header)
    print(rule)
    for name, preview, latency_str, cost_str in rows:
        # Truncate preview and replace newlines so it fits on one line
        flat = " ".join(preview.split())
        flat = flat[:COL["preview"] - 1] + "…" if len(flat) > COL["preview"] else flat
        print(
            f"{name:<{COL['name']}}{SEP}"
            f"{flat:<{COL['preview']}}{SEP}"
            f"{latency_str:>{COL['latency']}}{SEP}"
            f"{cost_str:>{COL['cost']}}"
        )
    print(rule)


if __name__ == "__main__":
    # ── CLI entry point ───────────────────────────────────────────────────────

    print(f"Question: {QUESTION}\n")
    print(f"Querying {len(MODELS)} models (timeout={TIMEOUT_SECONDS}s each)…\n")

    table_rows = []

    for name, model_id in MODELS:
        print(f"  • {name}…", end=" ", flush=True)
        try:
            answer, latency, in_tok, out_tok, cost = ask(QUESTION, model_id)
            latency_str = f"{latency:.2f}s"
            cost_str    = f"${cost:.6f}"
            preview     = answer
            print(f"done ({latency_str})")
        except Exception as e:
            latency_str = "—"
            cost_str    = "—"
            preview     = f"ERROR: {e}"
            print(f"FAILED — {e}")

        table_rows.append((name, preview, latency_str, cost_str))

    print()
    print_table(table_rows)
