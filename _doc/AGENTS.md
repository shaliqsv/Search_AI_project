# Search and Ranking on OTTO

Learning project: build search and ranking on the OTTO dataset in notebooks, then move one frozen model set through MLOps Levels 0, 1 and 2.

## Setup
Python 3.11 with uv.
- Install: `uv sync`
- Add a dependency: `uv add <pkg>` (dev-only: `uv add --dev <pkg>`). Never use pip directly.
- Test: `uv run pytest`
- Lint: `uv run ruff check .`
- Notebooks: `uv run jupyter lab` (kernel `search-ranking`). One-time setup commands are in the README.

## Files
- Tasks are GitHub issues, one at a time. Process is in `_doc/process.md`.
- `_doc/outdate/` holds the old plan, task list and architecture note. Do not treat them as current.

## Rules
- One task at a time. If it needs an output that doesn't exist yet, say so and stop.
- Phases A to F are notebooks only: no AWS, MLflow, Docker or pipelines.
- Split by time only. Build categories, co-visitation and features from the training window only.
- OTTO has no text, queries or positions, so categories and queries are synthetic. Label them so.
- 8GB RAM: use DuckDB or Polars, never full-dataset pandas.
- Report NDCG@10 with bootstrap intervals. Compare models with paired bootstrap.
- Cache every LLM call locally and use recorded fixtures in tests. Keep model IDs in config.
- Don't commit OTTO data until the license check (task 3) allows it.
- Minimise cost.
