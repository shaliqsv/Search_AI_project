# Search_AI_project
Performing search and ranking 

## Setup
- Install: `uv sync`
- Test: `uv run pytest`
- Lint: `uv run ruff check .`

## Notebooks
- One-time per machine: `uv run python -m ipykernel install --user --name search-ranking`
- One-time per clone (strips notebook outputs on commit): `uv run nbstripout --install --attributes .gitattributes`
- Start: `uv run jupyter lab`, and pick the `search-ranking` kernel
- Name notebooks `notebooks/NN_short_name.ipynb`, where NN is the issue number. Start from `notebooks/00_template.ipynb`
- Large data lives in `data/` (gitignored)

## Data license
OTTO data is CC BY 4.0 (credit OTTO, link the license, say what changed). Kaggle rules are not verified, so no OTTO data or derived samples are committed. Details: `docs/otto-license.md`.
