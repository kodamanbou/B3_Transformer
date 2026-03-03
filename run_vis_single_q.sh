MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

IDX=4

uv run src/gsm8k_llama2_analysis/vis_logitlens.py \
  --input-json outputs/llama3-8b-instruct/direct/analysis_results.json \
  --model-name $MODEL --mode direct --probe after_final \
  --index $IDX --max-examples 1 --outdir outputs/llama3-8b-instruct/direct/${IDX}

uv run src/gsm8k_llama2_analysis/vis_logitlens.py \
  --input-json outputs/llama3-8b-instruct/icl_cot/analysis_results.json \
  --model-name $MODEL --mode icl_cot --probe after_final \
  --index $IDX --max-examples 1 --outdir outputs/llama3-8b-instruct/icl_cot/${IDX}

uv run src/gsm8k_llama2_analysis/vis_logitlens.py \
  --input-json outputs/llama3-8b-instruct/cot/analysis_results.json \
  --model-name $MODEL --mode cot --probe after_final \
  --index $IDX --max-examples 1 --outdir outputs/llama3-8b-instruct/cot/${IDX}

uv run src/gsm8k_llama2_analysis/vis_logitlens.py \
  --input-json outputs/llama3-8b-instruct/icl/analysis_results.json \
  --model-name $MODEL --mode icl --probe after_final \
  --index $IDX --max-examples 1 --outdir outputs/llama3-8b-instruct/icl/${IDX}