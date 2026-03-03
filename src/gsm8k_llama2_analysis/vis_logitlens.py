from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Logit-lens visualization from analysis_results.json (TransformerLens).")
    p.add_argument("--input-json", required=True, help="Path to analysis_results.json produced by accuracy script.")
    p.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B-Instruct", help="HF model name, e.g., meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--outdir", default="viz_logitlens", help="Output directory for figures and CSVs.")
    p.add_argument("--mode", default="", choices=["", "direct", "cot", "icl", "icl_cot"], help="Filter by mode.")
    p.add_argument("--only-correct", action="store_true", help="Visualize only correct samples.")
    p.add_argument("--only-incorrect", action="store_true", help="Visualize only incorrect samples.")
    p.add_argument("--max-examples", type=int, default=10, help="Max number of examples to visualize.")
    p.add_argument("--topk", type=int, default=10, help="Top-k tokens to print for a few layers.")
    p.add_argument(
        "--probe",
        default="after_final",
        choices=["after_final", "start"],
        help=(
            "Probe position. "
            "'after_final' probes the next token right after the last 'Final answer:' in generated completion. "
            "'start' probes at the very start of assistant generation."
        ),
    )
    p.add_argument("--max-prompt-tokens", type=int, default=0, help="If >0, truncate tokenized prompt to last N tokens.")
    p.add_argument(
        "--index",
        type=int,
        default=-1,
        help="If >=0, visualize only the row with this per-run index (0..max_samples-1).",
    )
    p.add_argument(
        "--indexes",
        default="",
        help="Comma-separated per-run indexes to visualize, e.g. '3,7,12'. Overrides --index if set.",
    )
    return p.parse_args()


# --------- Utilities copied / compatible with accuracy script ---------

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


def build_messages_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    """
    Prefer saved messages. If absent, reconstruct from question/mode/demonstrations in row.
    This assumes the same build_messages() logic used in the accuracy script.
    """
    if "messages" in row and isinstance(row["messages"], list):
        return row["messages"]

    mode = row["mode"]
    question = row["question"]
    demonstrations = row.get("demonstrations")

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
            raise ValueError("Row has no demonstrations and no saved messages; cannot reconstruct ICL prompt.")
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


def messages_to_tokens(model: HookedTransformer, messages: list[dict[str, str]], add_generation_prompt: bool) -> torch.Tensor:
    tok = model.tokenizer
    if hasattr(tok, "apply_chat_template"):
        token_ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
        tokens = token_ids
    else:
        # Fallback: simple concatenation (less ideal for instruct)
        text = "\n\n".join([f"{m['role'].upper()}:\n{m['content']}" for m in messages])
        tokens = model.to_tokens(text)

    device = next(model.parameters()).device
    return tokens.to(device)


def assistant_prefix_for_after_final(generated_completion: str) -> str:
    """
    Take the generated completion and keep everything up to the LAST occurrence of 'Final answer:' (inclusive).
    This lets us probe the next token distribution right after 'Final answer:'.
    """
    if not generated_completion:
        return "Final answer:"

    # Find last occurrence of "Final answer:"
    pattern = re.compile(r"final\s*answer\s*:", flags=re.IGNORECASE)
    matches = list(pattern.finditer(generated_completion))
    if not matches:
        # If model didn't print it, we still anchor with the intended string
        return generated_completion.rstrip() + "\nFinal answer:"

    last = matches[-1]
    return generated_completion[: last.end()].rstrip()


def token_rank(logits_1d: torch.Tensor, token_id: int) -> int:
    sorted_idx = torch.argsort(logits_1d, descending=True)
    return int((sorted_idx == token_id).nonzero(as_tuple=False).item()) + 1


def decode_token(model: HookedTransformer, token_id: int) -> str:
    return model.tokenizer.decode([token_id])


@torch.no_grad()
def compute_logitlens_ranks(
    model: HookedTransformer,
    tokens: torch.Tensor,
    pos: int,
    target_token_id: int,
) -> tuple[list[int], list[float]]:
    """
    Logit-lens using resid_post at each layer:
      resid_post -> ln_final -> unembed -> logits
    Returns per-layer:
      - rank of target token
      - log prob of target token
    """
    model.eval()

    # Cache only resid_post hooks to reduce memory
    def nf(name: str) -> bool:
        return name.endswith("hook_resid_post")

    _, cache = model.run_with_cache(tokens, names_filter=nf)

    ranks: list[int] = []
    logps: list[float] = []

    for layer in range(model.cfg.n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        resid = cache[key][:, pos, :]  # [1, d_model]

        resid_ln = model.ln_final(resid)      # [1, d_model]
        logits = model.unembed(resid_ln)[0]   # [vocab]
        probs = F.softmax(logits, dim=-1)

        ranks.append(token_rank(logits.detach().cpu(), target_token_id))
        logps.append(float(torch.log(probs[target_token_id] + 1e-30).item()))

    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ranks, logps


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.input_json, "r", encoding="utf-8") as f:
        rows: list[dict[str, Any]] = json.load(f)

    # Build target index list (if specified)
    target_indexes: list[int] = []
    if args.indexes.strip():
        target_indexes = [int(x.strip()) for x in args.indexes.split(",") if x.strip() != ""]
    elif args.index >= 0:
        target_indexes = [int(args.index)]

    # Filter rows
    filtered: list[dict[str, Any]] = []

    if target_indexes:
        # Fast path: pick exactly those indexes (1 run loads model once, then loops)
        row_by_index: dict[int, dict[str, Any]] = {}
        for r in rows:
            if "index" not in r:
                continue
            try:
                ix = int(r.get("index"))
            except Exception:
                continue
            row_by_index[ix] = r

        for ix in target_indexes:
            r = row_by_index.get(ix)
            if r is None:
                print(f"[skip] index {ix} not found in JSON")
                continue
            if args.mode and r.get("mode") != args.mode:
                print(f"[skip] index {ix} mode mismatch (row mode={r.get('mode')})")
                continue
            if args.only_correct and not r.get("is_correct"):
                print(f"[skip] index {ix} not correct")
                continue
            if args.only_incorrect and r.get("is_correct"):
                print(f"[skip] index {ix} not incorrect")
                continue
            filtered.append(r)
    else:
        # Legacy behavior: filter all rows then take first max-examples
        for r in rows:
            if args.mode and r.get("mode") != args.mode:
                continue
            if args.only_correct and not r.get("is_correct"):
                continue
            if args.only_incorrect and r.get("is_correct"):
                continue
            filtered.append(r)
        filtered = filtered[: args.max_examples]

    if not filtered:
        raise SystemExit(
            "No rows matched filters (or all specified indexes were skipped). "
            "Try removing filters or check input JSON / indexes."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(args.model_name, device=device)

    for i, row in enumerate(filtered):
        mode = row["mode"]
        ref = row.get("reference_final_answer") or extract_final_answer(row.get("reference_answer", ""))
        if ref is None:
            print(f"[skip] row {i} has no parsable reference_final_answer")
            continue

        # Target token: FIRST token of the reference number (simple, robust)
        target_ids = model.to_tokens(ref, prepend_bos=False)[0].tolist()
        if not target_ids:
            print(f"[skip] row {i} reference tokenization empty")
            continue
        target_token_id = target_ids[0]
        target_tok = decode_token(model, target_token_id)

        base_messages = build_messages_from_row(row)

        # Build probe tokens
        if args.probe == "start":
            # Probe next token at start of assistant generation
            probe_messages = base_messages
            tokens = messages_to_tokens(model, probe_messages, add_generation_prompt=True)
            # We want next-token distribution after the last prompt token
            pos = tokens.shape[-1] - 1

        else:
            # Probe right after the last "Final answer:" in the *generated* completion
            completion = row.get("generated_text", "")
            assistant_prefix = assistant_prefix_for_after_final(completion)

            # Add assistant message with prefix; do NOT add generation prompt
            probe_messages = base_messages + [{"role": "assistant", "content": assistant_prefix}]
            tokens = messages_to_tokens(model, probe_messages, add_generation_prompt=False)
            pos = tokens.shape[-1] - 1

        if args.max_prompt_tokens and tokens.shape[-1] > args.max_prompt_tokens:
            tokens = tokens[:, -args.max_prompt_tokens:]
            pos = tokens.shape[-1] - 1

        ranks, logps = compute_logitlens_ranks(model, tokens, pos, target_token_id)

        # Plot rank curve
        plt.figure()
        plt.plot(list(range(len(ranks))), ranks)
        plt.gca().invert_yaxis()
        plt.xlabel("Layer")
        plt.ylabel(f"Rank of correct first-token {target_tok!r} (lower is better)")
        row_index = row.get("index", i)
        plt.title(f"Logit-lens rank | mode={mode} | idx={row_index} | correct={row.get('is_correct')} | probe={args.probe}")
        fig_rank = outdir / f"{i:03d}_idx{row_index}_rank_{mode}_correct{int(bool(row.get('is_correct')))}_{args.probe}.png"
        plt.savefig(fig_rank, dpi=150, bbox_inches="tight")
        plt.close()

        # Plot logprob curve
        plt.figure()
        plt.plot(list(range(len(logps))), logps)
        plt.xlabel("Layer")
        plt.ylabel(f"log P(correct first-token {target_tok!r})")
        plt.title(f"Logit-lens logprob | mode={mode} | idx={row_index} | correct={row.get('is_correct')} | probe={args.probe}")
        fig_logp = outdir / f"{i:03d}_idx{row_index}_logp_{mode}_correct{int(bool(row.get('is_correct')))}_{args.probe}.png"
        plt.savefig(fig_logp, dpi=150, bbox_inches="tight")
        plt.close()

        # Save per-layer numbers
        csv_path = outdir / f"{i:03d}_idx{row_index}_layers_{mode}_correct{int(bool(row.get('is_correct')))}_{args.probe}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("layer,rank,logp\n")
            for l, (rk, lp) in enumerate(zip(ranks, logps)):
                f.write(f"{l},{rk},{lp}\n")

        print(f"[saved] {fig_rank}")
        print(f"[saved] {fig_logp}")
        print(f"[saved] {csv_path}")

    print(f"Done. Output dir: {outdir}")


if __name__ == "__main__":
    main()