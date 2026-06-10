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


FEW_SHOT_EXAMPLES = [
    {
        "question": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?",
        "options": "A. Ampicillin\nB. Ceftriaxone\nC. Doxycycline\nD. Nitrofurantoin",
        "answer": "Nitrofurantoin"
    },
    {
        "question": "A 20-year-old woman presents with menorrhagia for the past several years. She says that her menses “have always been heavy”, and she has experienced easy bruising for as long as she can remember. Family history is significant for her mother, who had similar problems with bruising easily. The patient's vital signs include: heart rate 98/min, respiratory rate 14/min, temperature 36.1°C (96.9°F), and blood pressure 110/87 mm Hg. Physical examination is unremarkable. Laboratory tests show the following: platelet count 200,000/mm3, PT 12 seconds, and PTT 43 seconds. Which of the following is the most likely cause of this patient’s symptoms?",
        "options": "A. Hemophilia A\nB. Lupus anticoagulant\nC. Protein C deficiency\nD. Von Willebrand disease",
        "answer": "Von Willebrand disease"
    },
    {
        "question": "A 1-year-old boy presents to the emergency department with weakness and a change in his behavior. His parents state that they first noticed the change in his behavior this morning and it has been getting worse. They noticed the patient was initially weak in his upper body and arms, but now he won’t move his legs with as much strength or vigor as he used to. Physical exam is notable for bilateral ptosis with a sluggish pupillary response, a very weak sucking and gag reflex, and shallow respirations. The patient is currently drooling and his diaper is dry. The parents state he has not had a bowel movement in over 1 day. Which of the following is the pathophysiology of this patient’s condition?",
        "options": "A. Autoantibodies against the presynaptic voltage-gated calcium channels\nB. Autoimmune demyelination of peripheral nerves\nC. Blockade of presynaptic acetylcholine release at the neuromuscular junction\nD. Lower motor neuron destruction in the anterior horn",
        "answer": "Blockade of presynaptic acetylcholine release at the neuromuscular junction"
    }
]


def main() -> None:
    args = parse_args()
    if args.limit == DEFAULT_EVAL_LIMIT:
        args.limit = 128  # Use smaller limit for reasoning/few-shot iteration
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
        few_shot_text = ""
        for ex in FEW_SHOT_EXAMPLES:
            few_shot_text += f"Question:\n{ex['question']}\n\nChoices:\n{ex['options']}\n\nFinal short answer: {ex['answer']}\n\n---\n\n"
        
        full_prompt = (
            "You are a precise medical assistant. Below are examples of medical questions and their short-phrase answers.\n\n"
            f"{few_shot_text}"
            f"{prompt}"
        )

        messages = [
            {"role": "system", "content": "You are a precise medical exam assistant. Answer with only the short phrase."},
            {"role": "user", "content": full_prompt},
        ]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
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
            "max_new_tokens": max(args.max_new_tokens, 2048),
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
