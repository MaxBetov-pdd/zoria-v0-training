#!/usr/bin/env python3
"""Colab entry point for the Zoria Qwen3.5-9B pilot."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3.5-9B"
DATA_DIR = ROOT / "data" / "processed-colab"
ADAPTER_DIR = ROOT / "outputs" / "zoria-qwen35-9b"
TASKS = "belebele_ukr_Cyrl,xlsum_uk,ifeval_uk"
BENCHMARK_DIR = ROOT / "benchmarks" / "ukrainian-llm-leaderboard"


def run(*args: str, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(list(args), cwd=ROOT, env=env, check=True)


def prepare() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    run(
        sys.executable,
        "scripts/prepare_mvp_data.py",
        "--output-dir",
        str(DATA_DIR),
        "--max-per-source",
        "3000",
        "--seed",
        "42",
    )


def ensure_benchmark_repo() -> None:
    if BENCHMARK_DIR.exists():
        return
    run(
        "git",
        "clone",
        "https://github.com/lang-uk/ukrainian-llm-leaderboard.git",
        str(BENCHMARK_DIR),
    )
    run(
        "git",
        "-C",
        str(BENCHMARK_DIR),
        "checkout",
        "74d8069",
    )


def benchmark(adapter: str | None = None) -> None:
    ensure_benchmark_repo()
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HOME"] = env.get("HF_HOME", "/content/huggingface")
    model_args = (
        f"pretrained={MODEL},load_in_4bit=True,torch_dtype=float16,device_map=auto"
    )
    if adapter:
        model_args += f",peft={adapter}"
    run(
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        TASKS,
        "--include_path",
        "benchmarks/ukrainian-llm-leaderboard/tasks",
        "--batch_size",
        "1",
        "--limit",
        "50",
        "--apply_chat_template",
        "--output_path",
        "eval-results/qwen35-9b-colab"
        + ("-adapter" if adapter else "-base"),
        "--log_samples",
        env=env,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("prepare", "baseline", "smoke", "train", "benchmark-adapter"),
        required=True,
    )
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare()
    elif args.mode == "baseline":
        prepare()
        benchmark()
    elif args.mode == "smoke":
        prepare()
        run("axolotl", "train", "configs/qwen3.5-9b-colab-qlora-smoke.yaml")
    elif args.mode == "train":
        prepare()
        run("axolotl", "train", "configs/qwen3.5-9b-colab-qlora.yaml")
    elif args.mode == "benchmark-adapter":
        benchmark(str(ADAPTER_DIR))


if __name__ == "__main__":
    main()
