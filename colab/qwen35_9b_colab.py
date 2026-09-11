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
FULL_TASKS = "ukrainian_bench"
BENCHMARK_DIR = ROOT / "benchmarks" / "ukrainian-llm-leaderboard"


def run(*args: str, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(list(args), cwd=ROOT, env=env, check=True)


def prepare() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if (DATA_DIR / "zoria_v0_manifest.json").exists() and (
        DATA_DIR / "zoria_v0_train.jsonl"
    ).exists():
        print(f"Using existing prepared data in {DATA_DIR}")
        return
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


def prepare_a100() -> None:
    output_dir = ROOT / "data" / "processed-colab-a100"
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "zoria_v0_manifest.json").exists() and (
        output_dir / "zoria_v0_train.jsonl"
    ).exists():
        print(f"Using existing prepared data in {output_dir}")
        return
    run(
        sys.executable,
        "scripts/prepare_mvp_data.py",
        "--output-dir",
        str(output_dir),
        "--max-per-source",
        "5000",
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


def benchmark(adapter: str | None = None, full: bool = False) -> None:
    ensure_benchmark_repo()
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HOME"] = env.get("HF_HOME", "/content/huggingface")
    # A100 has enough VRAM for FP16 evaluation of the 9B model. Training
    # remains 4-bit QLoRA. The current lm-eval/Transformers path for
    # Qwen3.5 does not accept load_in_4bit as a direct model constructor kwarg.
    model_args = f"pretrained={MODEL},dtype=float16,device_map=auto"
    if adapter:
        model_args += f",peft={adapter}"
    args = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        FULL_TASKS if full else TASKS,
        "--include_path",
        "benchmarks/ukrainian-llm-leaderboard/tasks",
        "--batch_size",
        "1",
        "--apply_chat_template",
        "--confirm_run_unsafe_code",
        "--output_path",
        "eval-results/qwen35-9b-colab"
        + ("-adapter" if adapter else "-base")
        + ("-full" if full else "-smoke"),
        "--log_samples",
    ]
    if not full:
        args.extend(["--limit", "50"])
    run(*args, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "prepare",
            "baseline",
            "smoke",
            "train",
            "a100-smoke",
            "a100-train",
            "baseline-full",
            "benchmark-adapter",
            "benchmark-adapter-full",
        ),
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
    elif args.mode == "a100-smoke":
        prepare_a100()
        run(
            "axolotl",
            "train",
            "configs/qwen3.5-9b-colab-a100-smoke.yaml",
        )
    elif args.mode == "a100-train":
        prepare_a100()
        run(
            "axolotl",
            "train",
            "configs/qwen3.5-9b-colab-a100.yaml",
        )
    elif args.mode == "baseline-full":
        prepare()
        benchmark(full=True)
    elif args.mode == "benchmark-adapter":
        benchmark(str(ADAPTER_DIR))
    elif args.mode == "benchmark-adapter-full":
        benchmark(str(ADAPTER_DIR), full=True)


if __name__ == "__main__":
    main()
