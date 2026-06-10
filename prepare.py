"""Prepare and evaluate MedQA for autoresearch.

Usage:
    uv run prepare.py

This file owns all dataset loading and evaluation. train.py should import the
public helpers here and must not reimplement the metric.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-8B"
DATASET_NAME = "bigbio/med_qa"
DATASET_CONFIG = "med_qa_en_4options_source"

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-medqa")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_PATH = os.path.join(DATA_DIR, "validation.jsonl")
TEST_PATH = os.path.join(DATA_DIR, "test.jsonl")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

SEED = 42
MAX_NEW_TOKENS = 96
# Use validation during autoresearch so the held-out test split stays clean.
DEFAULT_EVAL_SPLIT = "validation"
DEFAULT_EVAL_LIMIT = 512


@dataclass(frozen=True)
class MedQAExample:
    id: str
    question: str
    options: list[dict[str, str]]
    answer: str
    answer_idx: str
    meta_info: str = ""


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with `uv add datasets` "
            "or run with `uv run --with datasets prepare.py`."
        ) from exc
    return load_dataset


def _option_sort_key(opt: dict[str, str]):
    key = str(opt.get("key", ""))
    return (len(key), key)


def _coerce_example(raw: dict, split: str, idx: int) -> MedQAExample:
    options = list(raw.get("options", []))
    options = sorted(options, key=_option_sort_key)
    return MedQAExample(
        id=f"{split}-{idx}",
        question=str(raw.get("question", "")).strip(),
        options=[{"key": str(o["key"]).strip(), "value": str(o["value"]).strip()} for o in options],
        answer=str(raw.get("answer", "")).strip(),
        answer_idx=str(raw.get("answer_idx", "")).strip(),
        meta_info=str(raw.get("meta_info", "")).strip(),
    )


def _write_jsonl(path: str, rows: Iterable[MedQAExample]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            n += 1
    return n


def prepare_data() -> None:
    """Download MedQA from Hugging Face and normalize it to local JSONL files."""
    os.makedirs(DATA_DIR, exist_ok=True)
    load_dataset = _require_datasets()
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, trust_remote_code=True)

    counts: dict[str, int] = {}
    split_to_path = {"train": TRAIN_PATH, "validation": VAL_PATH, "test": TEST_PATH}
    for split, path in split_to_path.items():
        rows = (_coerce_example(ex, split, i) for i, ex in enumerate(ds[split]))
        counts[split] = _write_jsonl(path, rows)

    metadata = {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "seed": SEED,
        "counts": counts,
        "paths": split_to_path,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


def ensure_prepared() -> None:
    missing = [p for p in (TRAIN_PATH, VAL_PATH, TEST_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "MedQA data is not prepared. Run `uv run prepare.py` first. "
            f"Missing: {missing}"
        )


def load_examples(split: str = DEFAULT_EVAL_SPLIT, limit: int | None = None, seed: int = SEED) -> list[MedQAExample]:
    ensure_prepared()
    path = {"train": TRAIN_PATH, "validation": VAL_PATH, "test": TEST_PATH}[split]
    rows: list[MedQAExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(MedQAExample(**json.loads(line)))
    if limit is not None and limit > 0 and limit < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, limit)
    return rows


def format_prompt(example: MedQAExample) -> str:
    choices = "\n".join(f"{o['key']}. {o['value']}" for o in example.options)
    return (
        "You are answering a medical board-style multiple-choice question.\n"
        "Return only the final answer as a short phrase. Do not include reasoning, "
        "letters, explanations, or caveats.\n\n"
        f"Question:\n{example.question}\n\n"
        f"Choices:\n{choices}\n\n"
        "Final short answer:"
    )


# ---------------------------------------------------------------------------
# Evaluation (kept out of train.py on purpose)
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _normalize_answer(text: str) -> str:
    text = _strip_thinking(text).lower().strip()
    text = text.splitlines()[0] if text else ""
    text = re.sub(r"^[\s\-–—]*(answer|final answer)\s*[:\-]\s*", "", text)
    text = re.sub(r"^[\(\[]?[a-j][\)\].:\-]\s*", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _short_answer_match(prediction: str, gold: str) -> bool:
    pred = _normalize_answer(prediction)
    tgt = _normalize_answer(gold)
    return bool(pred) and (pred == tgt or pred in tgt or tgt in pred)


def evaluate_short_answer_accuracy(
    predict: Callable[[str], str],
    split: str = DEFAULT_EVAL_SPLIT,
    limit: int = DEFAULT_EVAL_LIMIT,
) -> dict[str, float | int | str]:
    examples = load_examples(split=split, limit=limit)
    correct = 0
    for i, ex in enumerate(examples, 1):
        pred = predict(format_prompt(ex))
        correct += int(_short_answer_match(pred, ex.answer))
        if i % 25 == 0:
            print(f"eval {i}/{len(examples)} accuracy={correct / i:.4f}", flush=True)
    total = len(examples)
    accuracy = correct / total if total else 0.0
    return {"split": split, "n": total, "correct": correct, "accuracy": accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MedQA for autoresearch")
    parser.add_argument("--force", action="store_true", help="Re-download/rewrite even if files exist")
    args = parser.parse_args()
    if args.force or not all(os.path.exists(p) for p in (TRAIN_PATH, VAL_PATH, TEST_PATH)):
        prepare_data()
    else:
        print(f"MedQA already prepared in {DATA_DIR}")
