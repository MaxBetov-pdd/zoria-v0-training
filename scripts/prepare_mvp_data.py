#!/usr/bin/env python3
"""Build the fixed Zoria v0 text-only SFT mixture.

The script streams public Hugging Face datasets, converts them to the
OpenAI Messages shape expected by Axolotl, removes exact duplicates, and
writes a small deterministic train file plus an audit holdout.

It deliberately does not download Kobza, FinePDFs, leaderboard data, or
the Lapa reasoning dataset. Those belong to later experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from tqdm import tqdm


SOURCE_SPECS = [
    # Ukrainian task data.
    {"id": "lapa-llm/hermes3-uk", "split": "train", "limit": 15000},
    {"id": "lapa-llm/fiftyfive-best", "split": "train", "limit": 15000},
    {"id": "lapa-llm/wiki-facts-conversations", "split": "train", "limit": 8000},
    {"id": "lapa-llm/impossible-questions", "split": "train", "limit": 4000},
    {"id": "lapa-llm/antipropaganda-safe", "split": "train", "limit": 3000},
    {"id": "grammarly/spivavtor", "split": "train", "limit": 8000},
    {"id": "FIdo-AI/ua-squad", "split": "train", "limit": 5000},
    # Small English replay. Keep this separate in the manifest/report.
    {"id": "lapa-llm/hermes3-en-fixed", "split": "train", "limit": 7000},
]


def canonical_role(value: Any) -> str:
    role = str(value or "").lower()
    if role in {"human", "user", "prompt"}:
        return "user"
    if role in {"assistant", "gpt", "bot", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    return role or "user"


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(x for x in (text(item) for item in value) if x)
    if isinstance(value, dict):
        if "text" in value:
            return text(value["text"])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def messages_from_conversations(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        content = text(item.get("content", item.get("value", item.get("text"))))
        if not content:
            continue
        role = canonical_role(item.get("role", item.get("from")))
        if role not in {"system", "user", "assistant"}:
            continue
        result.append({"role": role, "content": content})
    return result


def make_messages(row: dict[str, Any], source: str) -> Iterable[list[dict[str, str]]]:
    conversations = row.get("conversations", row.get("conversation", row.get("messages")))
    parsed = messages_from_conversations(conversations)
    if len(parsed) >= 2:
        yield parsed

    if source == "lapa-llm/wiki-facts-conversations":
        article = text(row.get("text"))
        summary = text(row.get("summary"))
        if article and summary:
            yield [
                {
                    "role": "user",
                    "content": "Стисло підсумуй наведений український текст.\n\n" + article,
                },
                {"role": "assistant", "content": summary},
            ]
        return

    if source == "grammarly/spivavtor":
        src = text(row.get("src"))
        tgt = text(row.get("tgt"))
        if src and tgt:
            yield [
                {"role": "user", "content": src},
                {"role": "assistant", "content": tgt},
            ]
        return

    if source == "FIdo-AI/ua-squad":
        nested = row.get("data")
        if isinstance(nested, dict):
            context = text(nested.get("context"))
            question = text(nested.get("question"))
            answer = text(nested.get("answer"))
            if context and question and answer:
                yield [
                    {
                        "role": "user",
                        "content": (
                            "Прочитай контекст і дай коротку відповідь українською. "
                            "Відповідь має бути присутня в контексті.\n\n"
                            f"Контекст:\n{context}\n\nЗапитання:\n{question}"
                        ),
                    },
                    {"role": "assistant", "content": answer},
                ]


def normalize(source: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for messages in make_messages(row, source):
        if not messages or messages[-1]["role"] != "assistant":
            continue
        if not any(item["role"] == "user" for item in messages):
            continue
        canonical = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        result.append(
            {
                "messages": messages,
                "source": source,
                "example_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Override every source cap; useful for a tiny smoke test.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "zoria_v0_train.jsonl"
    holdout_path = output_dir / "zoria_v0_audit_holdout.jsonl"
    manifest_path = output_dir / "zoria_v0_manifest.json"

    seen: set[str] = set()
    train_items: list[dict[str, Any]] = []
    holdout_items: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for spec in SOURCE_SPECS:
        source = spec["id"]
        limit = args.max_per_source or spec["limit"]
        print(f"streaming {source}:{spec['split']} (cap={limit})")
        dataset = load_dataset(source, split=spec["split"], streaming=True)
        dataset = dataset.shuffle(seed=args.seed, buffer_size=10000)
        emitted = 0
        for row in tqdm(dataset, desc=source):
            for item in normalize(source, dict(row)):
                if emitted >= limit:
                    break
                if item["example_hash"] in seen:
                    continue
                seen.add(item["example_hash"])
                # Keep a deterministic 2% audit holdout out of training.
                bucket = int(item["example_hash"][:8], 16) % 100
                if bucket < 2:
                    holdout_items.append(item)
                else:
                    train_items.append(item)
                counts[source] += 1
                emitted += 1
            if emitted >= limit:
                break

    # One global deterministic shuffle makes source order irrelevant.
    train_items.sort(
        key=lambda item: hashlib.sha256(
            f"{args.seed}:{item['example_hash']}".encode("utf-8")
        ).hexdigest()
    )
    holdout_items.sort(key=lambda item: item["example_hash"])

    with train_path.open("w", encoding="utf-8") as handle:
        for item in train_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with holdout_path.open("w", encoding="utf-8") as handle:
        for item in holdout_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "seed": args.seed,
        "train_examples": len(train_items),
        "audit_holdout_examples": len(holdout_items),
        "source_examples_before_holdout": dict(counts),
        "sources": SOURCE_SPECS,
        "official_leaderboard_policy": "evaluation_only",
        "excluded_from_v0": [
            "Goader/kobza",
            "HuggingFaceFW/finepdfs",
            "lapa-llm/openthoughts_no_think",
            "official Ukrainian leaderboard evaluation data",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
