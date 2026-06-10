"""Prompt Qwen3-8B on MedQA and report short-answer-match accuracy.

Usage:
    uv run train.py

The evaluation implementation lives in prepare.py. This file only defines how
we prompt/generate with the model under test.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from prepare import (
    DEFAULT_EVAL_LIMIT,
    DEFAULT_EVAL_SPLIT,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    evaluate_short_answer_accuracy,
)

# Redirect default cache to local scratch
os.environ.setdefault("HF_HOME", "/local/scratch/kdhole/.cache/huggingface")
os.environ.setdefault("XDG_CACHE_HOME", "/local/scratch/kdhole/.cache")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install it with `uv add transformers accelerate safetensors` "
            "or run with `uv run --with transformers --with accelerate --with safetensors train.py`."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-8B on MedQA")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--split", default=DEFAULT_EVAL_SPLIT, choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=DEFAULT_EVAL_LIMIT)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    AutoModelForCausalLM, AutoTokenizer = _require_transformers()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    def predict(prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a precise medical exam assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        input_ids = input_ids.to(model.device)
        kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            kwargs["temperature"] = args.temperature
        with torch.inference_mode():
            if isinstance(input_ids, torch.Tensor):
                output_ids = model.generate(input_ids, **kwargs)
                input_len = input_ids.shape[-1]
            else:
                output_ids = model.generate(**input_ids, **kwargs)
                input_len = input_ids["input_ids"].shape[-1]
        generated = output_ids[0, input_len:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    result = evaluate_short_answer_accuracy(predict, split=args.split, limit=args.limit)
    total_seconds = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

    print("---")
    print(f"short_answer_accuracy: {result['accuracy']:.6f}")
    print(f"correct: {result['correct']}")
    print(f"n_eval: {result['n']}")
    print(f"split: {result['split']}")
    print(f"total_seconds: {total_seconds:.1f}")
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")
    print("json:", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
