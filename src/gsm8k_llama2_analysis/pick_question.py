import argparse
import json
import random


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def idx_set(rows, correct: bool):
    return set(int(r["index"]) for r in rows if bool(r.get("is_correct")) == correct)


def idx_to_row(rows):
    return {int(r["index"]): r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="analysis_results.json for baseline (usually direct)")
    ap.add_argument("--icl-cot", required=True, help="analysis_results.json for icl_cot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pick", choices=["first", "random"], default="random")
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--verify-question", action="store_true",
                    help="If set, also verify that the question text matches for the same index.")
    args = ap.parse_args()

    base_rows = load_json(args.baseline)
    iclcot_rows = load_json(args.icl_cot)

    base_bad = idx_set(base_rows, correct=False)
    iclcot_good = idx_set(iclcot_rows, correct=True)

    candidates = sorted(base_bad.intersection(iclcot_good))
    if not candidates:
        raise SystemExit(
            "No candidates found.\n"
            "Check that both runs used the same split/max-samples so that 'index' aligns."
        )

    base_map = idx_to_row(base_rows)
    iclcot_map = idx_to_row(iclcot_rows)

    rng = random.Random(args.seed)
    chosen = candidates[: args.show] if args.pick == "first" else rng.sample(candidates, k=min(args.show, len(candidates)))

    print("CANDIDATE_COUNT", len(candidates))
    print("CHOSEN_INDEXES", chosen)

    for idx in chosen:
        b = base_map[idx]
        c = iclcot_map[idx]

        if args.verify_question:
            qb = b.get("question", "")
            qc = c.get("question", "")
            print("QUESTION_MATCH", qb == qc)

        print("----")
        print("INDEX", idx)
        print("BASELINE predicted:", b.get("predicted_answer"), "ref:", b.get("reference_final_answer"))
        print("ICL_COT predicted:", c.get("predicted_answer"), "ref:", c.get("reference_final_answer"))
        # 先頭数行だけ表示（長すぎるので）
        bt = (b.get("generated_text", "") or "").strip().splitlines()
        ct = (c.get("generated_text", "") or "").strip().splitlines()
        print("BASELINE gen head:", " | ".join(bt[:3])[:300])
        print("ICL_COT gen head:", " | ".join(ct[:3])[:300])

    # 参考：可視化コマンドのひな形
    print("\n# Example visualization commands:")
    if chosen:
        ex = chosen[0]
        print(f"# IDX={ex}")
        print("# python viz_logitlens_from_results.py --input-json <baseline_json> --mode direct --index $IDX ...")
        print("# python viz_logitlens_from_results.py --input-json <icl_cot_json>  --mode icl_cot --index $IDX ...")


if __name__ == "__main__":
    main()