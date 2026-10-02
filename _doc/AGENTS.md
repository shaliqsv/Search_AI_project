# Search and Ranking on OTTO

Learning project: build search and ranking on the OTTO dataset, moving one model set through MLOps Levels 0, 1 and 2. Each Level is the outer loop; inside it, work runs through phases from `_doc/data_scientist.md`. Level 0 runs Phases 0-7 (Problem framing through Evaluation) manually in notebooks, ending at the go/no-go decision. Level 1 revisits the phases that need it to add pipeline automation, a shared feature store, experiment tracking, a model registry, continuous training and human approval before production — still short of deployment. Level 2 covers Phase 8 (Deployment readiness) and Phase 9 (Monitoring): CI/CD, automated validation, canary deployment, monitoring with alerting and retraining triggers, and live comparison.

## Setup
Python 3.11 with uv. `pyproject.toml` does not exist yet — recreating it is the first setup step; until it is merged these commands do not work.
- Install: `uv sync`
- Add a dependency: `uv add <pkg>` (dev-only: `uv add --dev <pkg>`). Never use pip directly.
- Test: `uv run pytest`
- Lint: `uv run ruff check .`
- Notebooks: `uv run jupyter lab`
Update this section once `pyproject.toml` exists if any command changes.

## Files
- `_doc/process.md`: how work is organised (one GitHub issue per phase, comments logged as steps complete)
- `_doc/data_scientist.md`: the phase methodology — phase order, entry/exit criteria, issue content — used inside every MLOps Level
- `_doc/eda_guide.md`: step guide for Phase 3 (EDA), used in place of the phase's default steps
- `_doc/modeling_guide.md`: step guide for Phase 6 (Modeling), used in place of the phase's default steps
- `_doc/outdate/`: superseded plan, task list and serving-architecture note from before the restart; kept for reference only, not authoritative

## Rules
- One task at a time. If it needs an output that doesn't exist yet, say so and stop.
- Level 0 (Phases 0-7) is notebooks only: no AWS, MLflow, Docker or pipelines. Level 1 revisits phases as needed to add pipeline automation, a feature store, experiment tracking, a model registry, continuous training and human approval before production. Level 2 is Phase 8 (Deployment readiness) and Phase 9 (Monitoring): CI/CD, automated validation, canary deployment, monitoring with alerting and retraining triggers, and live comparison.
- Do not start a phase (or a Level) until the previous one's exit criteria are met and recorded, per `_doc/data_scientist.md`.
- Split by time only. Build categories, co-visitation and features from the training window only.
- OTTO has no text, queries or positions, so categories and queries are synthetic. Label them so.
- 8GB RAM: use DuckDB or Polars, never full-dataset pandas.
- Report NDCG@10 with bootstrap intervals. Compare models with paired bootstrap.
- Cache every LLM call locally and use recorded fixtures in tests. Keep model IDs in config.
- Don't commit OTTO data until the license/compliance check is done and recorded (part of Phase 1: Dataset creation).
- Minimise cost. Serving is Lambda — see the archived rationale in `_doc/outdate/001-serving-architecture.md`; re-file it as a numbered entry in `_doc/decisions/` once a Level reaches deployment.
- Changing a decision means a new entry in `_doc/decisions/` (create the folder on first use).
