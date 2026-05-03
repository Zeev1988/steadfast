# Steadfast

Support ticket triage pipeline: BM25 retrieval over the KB, LLM classification, validation, and optional evaluation.

## Requirements

- Python **3.11+**
- An API key for whatever model you set in `.env` (LiteLLM-supported providers)

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For tests and linting:

```bash
pip install -r requirements-dev.txt
```

## Configure the LLM

Copy the example env file and edit it:

```bash
cp .env.example .env
```

Set **`LLM_MODEL`** to a LiteLLM model id (see comments in `.env.example`) and the matching **`ANTHROPIC_API_KEY`**, **`OPENAI_API_KEY`**, or other provider key. Optional: **`LLM_MAX_TOKENS`**

## Run the pipeline

Run from the repo root so imports resolve (`src/pipeline.py` adds `src` to the path):

```bash
python src/pipeline.py
```

Defaults: KB `data/knowledge_base.csv`, tickets `data/eval_set.json`, results `output/eval_results.json`.

Useful flags:

- `--eval` — write `eval_metrics.json` next to the results and run error analysis
- `--input FILE` / `--kb FILE` — alternate ticket JSON or KB CSV
- `--limit N` — first N tickets only
- `--output FILE` — results path

Example:

```bash
python src/pipeline.py --eval --limit 10
```

## Optional tooling

With **`PYTHONPATH=src`** from the repo root:

| Command | Purpose |
|--------|---------|
| `PYTHONPATH=src python -m loader.eda_kb_eval` | KB + eval EDA log → `output/eda_kb_eval.log` |
| `PYTHONPATH=src python -m visualization.summary` | KB correlation heatmap → `output/kb_correlation.png` |
| `PYTHONPATH=src python -m analyze.analyze` | Charts from `output/eval_metrics.json` → `output/figures/` |

`analyze.analyze` expects `output/eval_metrics.json` (produced by `pipeline.py --eval`). Use `--metrics` / `--output-dir` to point elsewhere.

## Tests and lint

```bash
pytest
ruff check src tests
```

`pytest.ini` sets `pythonpath = src`, so tests do not need a manual `PYTHONPATH`.

## Verifying this document

After `pip install -r requirements-dev.txt`, run **`bash scripts/verify_readme.sh`**: it checks `python src/pipeline.py --help`, the optional-module commands (with outputs under a temp directory), **`ruff check`**, and **`pytest`**. That covers everything except the LLM call — run **`python src/pipeline.py --limit 1`** locally once keys in **`.env`** are set. CI runs the same script on every push/PR (`readme-smoke` job).
