# Search and Ranking on OTTO

Learning project: build search and ranking on the OTTO dataset in notebooks, then move one frozen model set through MLOps Levels 0, 1 and 2.

## Setup
Python 3.11 with uv. Task 1 creates `pyproject.toml`; until it is merged these commands do not work yet.
- Install: `uv sync`
- Add a dependency: `uv add <pkg>` (dev-only: `uv add --dev <pkg>`). Never use pip directly.
- Test: `uv run pytest`
- Lint: `uv run ruff check .`
- Notebooks: `uv run jupyter lab`
Update this section if task 1 changes any command.

## Files
- `_doc/process.md`: how work is organised
-

## Rules
- One task at a time. If it needs an output that doesn't exist yet, say so and stop.
- Phases A to F are notebooks only: no AWS, MLflow, Docker or pipelines.
- Split by time only. Build categories, co-visitation and features from the training window only.
- OTTO has no text, queries or positions, so categories and queries are synthetic. Label them so.
- 8GB RAM: use DuckDB or Polars, never full-dataset pandas.
- Report NDCG@10 with bootstrap intervals. Compare models with paired bootstrap.
- Cache every LLM call locally and use recorded fixtures in tests. Keep model IDs in config.
- Don't commit OTTO data until the license check (task 3) allows it.
- Minimise cost. Serving is Lambda (`_doc/decisions/001-serving-architecture.md`).
- Changing a decision means a new entry in `_doc/decisions/`.
