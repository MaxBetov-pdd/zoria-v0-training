#!/usr/bin/env python3
"""Upload a completed Zoria adapter to Hugging Face.

Requires a logged-in HF session or HF_TOKEN. The script never stores tokens.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", default="outputs/zoria-qwen35-9b")
    parser.add_argument("--repo-id", required=True, help="for example: MaxBetov-pdd/zoria-qwen35-9b")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise SystemExit(f"Adapter directory does not exist: {adapter_dir}")

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(adapter_dir),
        path_in_repo=".",
        commit_message="Upload Zoria Qwen3.5-9B Colab pilot adapter",
    )
    print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
