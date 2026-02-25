from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm


DEFAULT_MODEL_NAME = "meta-llama/Llama-2-7b-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small TransformerLens analysis on GSM8K with Llama2-7B."
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def load_gsm8k_examples(split: str, max_samples: int) -> list[dict[str, str]]:
    ds = load_dataset("gsm8k", "main", split=split)
    examples: list[dict[str, str]] = []
    for row in ds.select(range(min(max_samples, len(ds)))):
        examples.append({"question": row["question"], "answer": row["answer"]})
    return examples


def build_prompt(question: str) -> str:
    return (
        "Solve the following grade-school math problem.\n"
        "Show short reasoning and provide the final numeric answer.\n\n"
        f"Question: {question}\nAnswer:"
    )


def analyze_examples(
    model: HookedTransformer,
    examples: list[dict[str, str]],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    model.eval()
    results: list[dict[str, Any]] = []

    for idx, ex in enumerate(tqdm(examples, desc="Analyzing")):
        prompt = build_prompt(ex["question"])
        tokens = model.to_tokens(prompt)
        with torch.inference_mode():
            logits, cache = model.run_with_cache(tokens)
            generated_tokens = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                stop_at_eos=False,
                verbose=False,
            )

        layer_key = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
        resid = cache[layer_key][0, -1]
        last_token_logits = logits[0, -1]
        pred_token_id = int(torch.argmax(last_token_logits).item())
        pred_token = model.tokenizer.decode([pred_token_id])
        generation = model.to_string(generated_tokens[0])

        results.append(
            {
                "index": idx,
                "question": ex["question"],
                "reference_answer": ex["answer"],
                "predicted_next_token": pred_token,
                "generated_text": generation,
                "last_layer_resid_l2_norm": float(torch.norm(resid, p=2).item()),
                "last_layer_resid_mean": float(torch.mean(resid).item()),
                "last_layer_resid_std": float(torch.std(resid).item()),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_gsm8k_examples(args.split, args.max_samples)
    model = HookedTransformer.from_pretrained(args.model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    results = analyze_examples(model, examples, args.max_new_tokens)

    json_path = output_dir / "analysis_results.json"
    csv_path = output_dir / "analysis_results.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(results).to_csv(csv_path, index=False)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()

