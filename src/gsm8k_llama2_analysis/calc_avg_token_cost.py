import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def main():
    # ====== 設定 ======
    input_json = "/workspace/outputs/llama3-8b-instruct/icl_cot/analysis_results.json"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # ==================

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    rows = json.loads(Path(input_json).read_text(encoding="utf-8"))

    # completion（生成部分）のトークン数を再計算
    for r in rows:
        completion = r.get("generated_text") or ""
        # add_special_tokens=False で純粋なテキスト分のみ数える
        r["completion_tokens"] = len(tokenizer(completion, add_special_tokens=False).input_ids)
        r["total_tokens"] = int(r.get("prompt_tokens", 0)) + int(r["completion_tokens"])

    df = pd.DataFrame(rows)

    # modeごとの平均
    summary = (
        df.groupby("mode", as_index=False)
        .agg(
            samples=("index", "count"),
            accuracy=("is_correct", "mean"),
            avg_prompt_tokens=("prompt_tokens", "mean"),
            avg_completion_tokens=("completion_tokens", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
        )
        .sort_values("mode")
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()