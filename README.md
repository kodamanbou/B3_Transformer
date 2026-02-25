# llama2-transformer-lens-gsm8k

TransformerLens を使って `Llama-2-7b` を GSM8K で解析するための実験プロジェクトです。  
パッケージ管理は `uv`、実行環境は Docker を前提にしています。

## 前提

- Docker / Docker Compose が使えること
- Hugging Face の `meta-llama/Llama-2-7b-hf` にアクセス権があること
- `HF_TOKEN` を環境変数で設定していること

```bash
export HF_TOKEN=hf_xxx
```

## 実行方法

### 1. Docker イメージをビルド

```bash
docker compose build
```

### 2. 解析を実行

```bash
docker compose run --rm app
```

## ローカル (uv) 実行

```bash
uv sync
uv run run-analysis --max-samples 16 --split test
```

## 出力

- `outputs/analysis_results.json`
- `outputs/analysis_results.csv`

各サンプルについて以下を保存します:

- 問題文
- 参照解答
- 次トークン予測
- 生成テキスト
- 最終層残差 (`resid_post`) の統計量

## ディレクトリ構成

```text
.
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── src
    └── gsm8k_llama2_analysis
        └── run_analysis.py
```

## メモ

- 初回実行時はモデル/データセットのダウンロードに時間がかかります。
- GPU が利用可能な場合は自動で `cuda`、それ以外は `cpu` で動作します。

