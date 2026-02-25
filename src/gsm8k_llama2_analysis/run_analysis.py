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
    parser.add_argument(
        "--mode",
        default="direct",
        choices=["direct", "cot", "icl", "icl_cot"],
        help="Prompting mode: direct answer, chain-of-thought prompt, ICL examples, or ICL with CoT examples.",
    )
    parser.add_argument(
        "--icl-k",
        type=int,
        default=8,
        help="Number of demonstrations used in ICL modes.",
    )
    return parser.parse_args()


def load_gsm8k_examples(split: str, max_samples: int, start_index: int = 0) -> list[dict[str, str]]:
    ds = load_dataset("gsm8k", "main", split=split)
    end_index = min(start_index + max_samples, len(ds))
    if start_index >= end_index:
        return []

    examples: list[dict[str, str]] = []
    for row in ds.select(range(start_index, end_index)):
        examples.append({"question": row["question"], "answer": row["answer"]})
    return examples


def extract_final_answer(answer: str) -> str:
    if "####" in answer:
        return answer.split("####", maxsplit=1)[1].strip()
    return answer.strip().splitlines()[-1].strip()


def extract_reasoning(answer: str) -> str:
    if "####" in answer:
        return answer.split("####", maxsplit=1)[0].strip()
    lines = answer.strip().splitlines()
    return "\n".join(lines[:-1]).strip()


def load_icl_demonstrations(mode: str, icl_k: int) -> list[dict[str, str]] | None:
    if mode not in {"icl", "icl_cot"}:
        return None

    ds = load_dataset("gsm8k", "main", split="train")
    k = min(icl_k, len(ds))
    demonstrations: list[dict[str, str]] = []
    for row in ds.select(range(k)):
        final_answer = extract_final_answer(row["answer"])
        if mode == "icl":
            demo_answer = f"Final answer: {final_answer}"
        else:
            reasoning = extract_reasoning(row["answer"])
            if reasoning:
                demo_answer = f"{reasoning}\nFinal answer: {final_answer}"
            else:
                demo_answer = f"Step 1: Compute the required arithmetic.\nFinal answer: {final_answer}"

        demonstrations.append({"question": row["question"], "answer": demo_answer})
    return demonstrations


def build_prompt(
    question: str,
    mode: str,
    demonstrations: list[dict[str, str]] | None,
) -> str:
    prompt_lines: list[str] = [
        "Solve the following grade-school math problem.",
        "Output format (all modes): end your response with exactly one line `Final answer: <number>`.",
    ]

    if mode == "direct":
        prompt_lines.append("Do not include reasoning. Return only the final answer line.")
    elif mode == "cot":
        prompt_lines.append(
            "Use explicit staged reasoning with numbered steps (Step 1, Step 2, ...), then conclude with `Final answer: <number>`."
        )
    elif mode in {"icl", "icl_cot"}:
        if not demonstrations:
            raise ValueError(f"Demonstrations are required for mode={mode}")
        if mode == "icl":
            prompt_lines.append("Follow the examples. Do not include reasoning for the target question.")
        else:
            prompt_lines.append(
                "Follow the examples. Use explicit staged reasoning with numbered steps for the target question."
            )
        prompt_lines.append("")
        for demo in demonstrations:
            prompt_lines.append(f"Question: {demo['question']}")
            prompt_lines.append(f"Answer:\n{demo['answer']}")
            prompt_lines.append("")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    prompt_lines.append(f"Question: {question}")
    prompt_lines.append("Answer:")
    return "\n".join(prompt_lines)


def analyze_examples(
    model: HookedTransformer,
    examples: list[dict[str, str]],
    max_new_tokens: int,
    mode: str,
    demonstrations: list[dict[str, str]] | None,
) -> list[dict[str, Any]]:
    model.eval()
    results: list[dict[str, Any]] = []

    for idx, ex in enumerate(tqdm(examples, desc="Analyzing")):
        prompt = build_prompt(ex["question"], mode=mode, demonstrations=demonstrations)
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

    demonstrations = load_icl_demonstrations(args.mode, args.icl_k)
    eval_start_index = args.icl_k if args.split == "train" and demonstrations else 0
    examples = load_gsm8k_examples(args.split, args.max_samples, start_index=eval_start_index)

    model = HookedTransformer.from_pretrained(args.model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    results = analyze_examples(
        model,
        examples,
        args.max_new_tokens,
        mode=args.mode,
        demonstrations=demonstrations,
    )

    json_path = output_dir / "analysis_results.json"
    csv_path = output_dir / "analysis_results.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(results).to_csv(csv_path, index=False)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
