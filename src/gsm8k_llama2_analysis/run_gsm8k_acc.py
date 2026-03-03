from __future__ import annotations

import argparse
import json
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm


# Llama-3 Instruct を推奨（CLIで上書き可）
DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GSM8K prompting experiments for accuracy ONLY (no run_with_cache)."
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate per example.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--evaluation-mode",
        default="single",
        choices=["single", "all"],
        help="Whether to evaluate only --mode or all prompting modes.",
    )
    parser.add_argument(
        "--mode",
        default="direct",
        choices=["direct", "cot", "icl", "icl_cot"],
        help="Prompting mode: direct answer, chain-of-thought prompt, ICL examples, or ICL with CoT examples.",
    )
    parser.add_argument("--num-shots", type=int, default=4, help="Number of demonstrations used in ICL modes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible demo selection.")
    parser.add_argument(
        "--save-messages",
        action="store_true",
        help="If set, save rendered chat messages into results (bigger JSON).",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=0,
        help="If >0, truncate tokenized prompt to last N tokens (OOM safety). 0 means no truncation.",
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


def load_demo_candidate_pool(split: str, excluded_indices: set[int] | None = None) -> list[dict[str, str]]:
    ds = load_dataset("gsm8k", "main", split=split)
    excluded = excluded_indices or set()
    candidates: list[dict[str, str]] = []
    for idx, row in enumerate(ds):
        if idx in excluded:
            continue
        candidates.append({"question": row["question"], "answer": row["answer"]})
    return candidates


def extract_final_answer(text: str) -> str | None:
    if not text:
        return None

    candidate_text = text
    if "####" in text:
        candidate_text = text.split("####", maxsplit=1)[1].strip()

    final_answer_line_pattern = re.compile(r"final\s*answer\s*:\s*(.+)", flags=re.IGNORECASE)
    for line in reversed(candidate_text.splitlines()):
        match = final_answer_line_pattern.search(line.strip())
        if match:
            candidate_text = match.group(1).strip()
            break

    number_pattern = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
    matches = number_pattern.findall(candidate_text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def answers_match(predicted: str | None, reference: str | None) -> bool:
    if predicted is None or reference is None:
        return False

    try:
        pred_decimal = Decimal(predicted)
        ref_decimal = Decimal(reference)
        return pred_decimal == ref_decimal
    except InvalidOperation:
        return predicted.strip() == reference.strip()


def extract_reasoning(answer: str) -> str:
    if "####" in answer:
        return answer.split("####", maxsplit=1)[0].strip()
    lines = answer.strip().splitlines()
    return "\n".join(lines[:-1]).strip()


def load_icl_demonstrations(
    mode: str,
    num_shots: int,
    seed: int,
    eval_split: str,
    eval_start_index: int,
    max_samples: int,
) -> list[dict[str, str]] | None:
    if mode not in {"icl", "icl_cot"}:
        return None

    # Avoid leakage if evaluating on train and selecting a contiguous slice
    excluded_indices: set[int] = set()
    if eval_split == "train":
        eval_end_index = eval_start_index + max_samples
        excluded_indices = set(range(eval_start_index, eval_end_index))

    pool = load_demo_candidate_pool(split="train", excluded_indices=excluded_indices)
    rng = random.Random(seed)
    rng.shuffle(pool)

    demonstrations: list[dict[str, str]] = []
    for row in pool:
        if len(demonstrations) >= num_shots:
            break

        final_answer = extract_final_answer(row["answer"])
        if final_answer is None:
            continue

        if mode == "icl":
            demo_answer = f"Final answer: {final_answer}"
        else:
            reasoning = extract_reasoning(row["answer"])
            if reasoning:
                demo_answer = f"{reasoning}\nFinal answer: {final_answer}"
            else:
                demo_answer = f"Step 1: Compute the required arithmetic.\nFinal answer: {final_answer}"

        demonstrations.append({"question": row["question"], "answer": demo_answer})

    if len(demonstrations) < num_shots:
        raise ValueError(
            f"Unable to build enough demonstrations for mode={mode}. requested={num_shots}, got={len(demonstrations)}"
        )
    return demonstrations


def build_messages(
    question: str,
    mode: str,
    demonstrations: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """
    Chat-template-friendly messages for Llama-3 Instruct.
    We keep instructions concise to reduce prompt length.
    """
    system_msg = (
        "You are a careful grade-school math solver. "
        "Follow the user's instructions exactly. "
        "Always end with exactly one line: `Final answer: <number>`."
    )

    user_lines: list[str] = [
        "Solve the following grade-school math problem.",
        "Output format: end your response with exactly one line `Final answer: <number>`.",
    ]

    if mode == "direct":
        user_lines.append("Do not include reasoning. Return only the final answer line.")
    elif mode == "cot":
        user_lines.append("Use numbered steps (Step 1, Step 2, ...) then the final answer line.")
    elif mode in {"icl", "icl_cot"}:
        if not demonstrations:
            raise ValueError(f"Demonstrations are required for mode={mode}")
        if mode == "icl":
            user_lines.append("Follow the examples. Do not include reasoning for the target question.")
        else:
            user_lines.append("Follow the examples. Use numbered steps for the target question.")
        user_lines.append("")
        for demo in demonstrations:
            user_lines.append(f"Question: {demo['question']}")
            user_lines.append("Answer:")
            user_lines.append(demo["answer"])
            user_lines.append("")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    user_lines.append(f"Question: {question}")
    user_lines.append("Answer:")

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def messages_to_tokens(model: HookedTransformer, messages: list[dict[str, str]]) -> torch.Tensor:
    """
    Convert chat messages to tokens using HF tokenizer's chat template.
    Falls back to a simple concatenation if apply_chat_template is unavailable.
    """
    tok = model.tokenizer
    if hasattr(tok, "apply_chat_template"):
        token_ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        tokens = token_ids
    else:
        # fallback (less optimal for instruct models)
        text = "\n\n".join([f"{m['role'].upper()}:\n{m['content']}" for m in messages])
        tokens = model.to_tokens(text)

    device = next(model.parameters()).device
    return tokens.to(device)


def analyze_examples(
    model: HookedTransformer,
    examples: list[dict[str, str]],
    max_new_tokens: int,
    mode: str,
    demonstrations: list[dict[str, str]] | None,
    num_shots: int,
    seed: int,
    save_messages: bool,
    max_prompt_tokens: int,
) -> list[dict[str, Any]]:
    model.eval()
    results: list[dict[str, Any]] = []

    for idx, ex in enumerate(tqdm(examples, desc=f"Generating ({mode})")):
        messages = build_messages(ex["question"], mode=mode, demonstrations=demonstrations)
        tokens = messages_to_tokens(model, messages)

        if max_prompt_tokens and tokens.shape[-1] > max_prompt_tokens:
            # Keep the end of the prompt (where the answer should be generated)
            tokens = tokens[:, -max_prompt_tokens:]

        with torch.inference_mode():
            generated_tokens = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                stop_at_eos=False,
                verbose=False,
            )

        prompt_token_count = tokens.shape[-1]
        generated_completion = model.to_string(generated_tokens[0, prompt_token_count:])
        generated_full_text = model.to_string(generated_tokens[0])

        predicted_answer = extract_final_answer(generated_completion)
        if predicted_answer is None:
            predicted_answer = extract_final_answer(generated_full_text)

        reference_final_answer = extract_final_answer(ex["answer"])
        is_correct = answers_match(predicted_answer, reference_final_answer)

        row: dict[str, Any] = {
            "index": idx,
            "question": ex["question"],
            "reference_answer": ex["answer"],
            "generated_text": generated_completion,
            "generated_text_full": generated_full_text,
            "predicted_answer": predicted_answer,
            "reference_final_answer": reference_final_answer,
            "is_correct": is_correct,
            "mode": mode,
            "num_shots": num_shots,
            "seed": seed,
            "prompt_tokens": int(prompt_token_count),
            "completion_chars": int(len(generated_completion)),
            "full_text_chars": int(len(generated_full_text)),
        }

        # Save demos so we can reproduce exact ICL prompts later (for logitlens script).
        if mode in {"icl", "icl_cot"} and demonstrations is not None:
            row["demonstrations"] = demonstrations

        if save_messages:
            row["messages"] = messages

        results.append(row)

        # Help reduce fragmentation in long runs
        del generated_tokens, tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = [args.mode] if args.evaluation_mode == "single" else ["direct", "cot", "icl", "icl_cot"]
    eval_start_index = args.num_shots if args.split == "train" and any(m in {"icl", "icl_cot"} for m in modes) else 0
    examples = load_gsm8k_examples(args.split, args.max_samples, start_index=eval_start_index)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(args.model_name, device=device)

    results: list[dict[str, Any]] = []
    summary_metrics: list[dict[str, Any]] = []

    for mode in modes:
        demonstrations = load_icl_demonstrations(
            mode=mode,
            num_shots=args.num_shots,
            seed=args.seed,
            eval_split=args.split,
            eval_start_index=eval_start_index,
            max_samples=args.max_samples,
        )

        mode_results = analyze_examples(
            model=model,
            examples=examples,
            max_new_tokens=args.max_new_tokens,
            mode=mode,
            demonstrations=demonstrations,
            num_shots=args.num_shots,
            seed=args.seed,
            save_messages=args.save_messages,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        results.extend(mode_results)

        total_samples = len(mode_results)
        num_correct = sum(1 for row in mode_results if row.get("is_correct") is True)
        accuracy = (num_correct / total_samples) if total_samples > 0 else 0.0

        summary_metrics.append(
            {
                "mode": mode,
                "num_shots": args.num_shots,
                "seed": args.seed,
                "split": args.split,
                "total_samples": total_samples,
                "num_correct": num_correct,
                "accuracy": accuracy,
            }
        )

    json_path = output_dir / "analysis_results.json"
    csv_path = output_dir / "analysis_results.csv"
    summary_path = output_dir / "summary_metrics.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(results).to_csv(csv_path, index=False)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved summary metrics: {summary_path}")


if __name__ == "__main__":
    main()