#!/usr/bin/env python3
"""Gemma 4 E4B Colab pilot runner.

Modes:
  sanity  - load Gemma 4 and generate Ukrainian answers
  prepare - prepare the existing Zoria data mixture
  smoke   - 20-step 8-bit LoRA smoke test
  train   - 300-step 8-bit LoRA pilot
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = "google/gemma-4-E4B-it"
DATA_DIR = ROOT / "data" / "processed-colab-a100"


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
        "5000",
        "--seed",
        "42",
    )


def sanity() -> None:
    script = r"""
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

model_id = "google/gemma-4-E4B-it"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

prompts = [
    "Відповідай українською: коротко поясни, що таке Київська Русь.",
    "Відповідай українською: виправ граматичну помилку «Я рахую що це правильно».",
    "Відповідай українською одним реченням: як працює сонячна панель?",
]

for prompt in prompts:
    messages = [{"role": "user", "content": prompt}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    answer = processor.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=False,
    )
    print("\nUSER:", prompt, "\nMODEL:", answer)
"""
    run(sys.executable, "-c", script)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sanity", "prepare", "smoke", "train"), required=True)
    args = parser.parse_args()

    if args.mode == "sanity":
        sanity()
    elif args.mode == "prepare":
        prepare()
    elif args.mode == "smoke":
        prepare()
        run("axolotl", "train", "configs/gemma4-e4b-colab-lora-8bit-smoke.yaml")
    elif args.mode == "train":
        prepare()
        run("axolotl", "train", "configs/gemma4-e4b-colab-lora-8bit.yaml")


if __name__ == "__main__":
    main()
