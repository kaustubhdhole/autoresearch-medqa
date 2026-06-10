# autoresearch-medqa

This repository is an autoresearch experiment for medical question answering on MedQA. The system under test is a prompting-based Qwen3-8B solver, and the optimization target is short-answer-match accuracy.

## Files in scope

You may change these files during this redesign:

- `prepare.py` — loads `bigbio/med_qa` from Hugging Face, writes normalized local JSONL splits, formats prompts, and owns the short-answer-match evaluation. The metric logic must stay here and must not be copied into `train.py`.
- `train.py` — loads/prompts `Qwen/Qwen3-8B`, defines generation behavior, calls the evaluator from `prepare.py`, and prints the final metrics.

Do not hide or duplicate evaluation logic in `train.py`. 
`train.py` may tune prompting, generation settings, batching, decoding, ensembling, add medical rules, or model-loading choices, but it should treat evaluation as an imported black-box helper from `prepare.py`.

## Setup

Prepare the MedQA data once:

```bash
uv run prepare.py
```

The preparation script loads:

```python
load_dataset("bigbio/med_qa", "med_qa_en_4options_source", trust_remote_code=True)
```

It writes normalized splits under `/local/scratch/kdhole/.cache/autoresearch-medqa/data/`.

If your environment does not already have the Hugging Face libraries, install or inject them before running:

```bash
uv add datasets transformers accelerate safetensors
```

or run with `uv run --with ...` equivalents.

## Experimentation

Run the current model/prompt experiment with:

```bash
uv run train.py > run.log 2>&1
```

The default evaluation split is MedQA validation, sampled to a fixed limit for fast iteration. The held-out test split should be used only for final reporting:

```bash
uv run train.py --split test --limit 0 > test.log 2>&1
```

`--limit 0` means evaluate all examples in the selected split.

## Objective

Maximize:

```text
short_answer_accuracy
```

The metric is a normalized short-answer match against the gold answer text. The model is instructed to return only the final answer phrase, not the option letter or explanation.

## What to tune

Good autoresearch experiments include:

- prompt wording and formatting
- whether to include option letters, answer-only constraints, or few-shot examples
- Qwen3 thinking vs non-thinking mode
- decoding parameters such as `max_new_tokens` and temperature
- batching/caching improvements that do not change the metric
- validation sample size for faster iteration vs lower noise

Keep experiments simple. Prefer changes that improve accuracy without making the prompt or harness brittle.

## Output format

At the end of a successful run, `train.py` prints:

```text
---
short_answer_accuracy: 0.000000
correct: 0
n_eval: 512
split: validation
total_seconds: 0.0
peak_vram_mb: 0.0
json: {...}
```

Extract the key metric with:

```bash
grep "^short_answer_accuracy:\|^peak_vram_mb:" run.log
```

## Logging results

Record experiments in `results.tsv` using tab-separated columns:

```text
commit	short_answer_accuracy	memory_gb	status	description
```

Use `keep`, `discard`, or `crash` for status. For crashes, set accuracy to `0.000000` and memory to `0.0`.

## Experiment loop

1. Check the current branch and git state.
2. Change only the in-scope behavior needed for the next experiment.
3. Commit the change.
4. Run `uv run train.py > run.log 2>&1`.
5. Read `short_answer_accuracy` and `peak_vram_mb` from the log.
6. Update `results.tsv`.
7. Keep changes that improve validation accuracy or simplify the system without hurting accuracy.
8. Reserve final test evaluation for the best validation candidate.
