"""Logging helpers for the modelling notebook (`notebooks/MODELING.ipynb`), following `_doc/modeling_guide.md`.

- `log_step` writes one entry per step to `modeling/model_log.md` in the guide's format (replaced when the step is rerun).
- `log_experiments` appends every experiment, including failures, to `modeling/experiments.csv` (rule 8).
- `save_table` keeps every comparison table in `modeling/tables/`; `cached` reloads an expensive result
  (set MODEL_FORCE=1 to recompute everything).
"""

import os
import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
MODELING = ROOT / "modeling"
TABLES, FIGURES, ARTIFACTS = MODELING / "tables", MODELING / "figures", MODELING / "artifacts"
LOG, EXPERIMENTS = MODELING / "model_log.md", MODELING / "experiments.csv"
SEED = 42
FORCE = os.environ.get("MODEL_FORCE") == "1"
for d in (TABLES, FIGURES, ARTIFACTS):
    d.mkdir(parents=True, exist_ok=True)

TEMPLATE = """### Step {n}. {title}
Kind: {kind}
Candidates run: {options}
Observed: {observed}
Decision: {decision}
Why: {why}
Rejected: {rejected}
Confidence: {confidence}
Affects later steps: {affects}
Needs human input: {human}
"""
EXPERIMENT_COLUMNS = ["id", "step", "description", "features", "model", "parameters", "seeds", "ndcg10_mean", "ndcg10_sd",
                      "pr_auc_mean", "pr_auc_sd", "fit_seconds", "predict_seconds", "size_mb", "notes"]


def log_step(n, title, kind, options, observed, decision, why, rejected, confidence, affects, human="no"):
    entry = TEMPLATE.format(n=n, title=title, kind=kind, options=options, observed=observed, decision=decision,
                            why=why, rejected=rejected, confidence=confidence, affects=affects, human=human)
    head = "# Modelling log\n\nOne entry per step of `_doc/modeling_guide.md`.\n\n"
    text = LOG.read_text() if LOG.exists() else head
    parts = re.split(r"(?m)^(?=### Step \d+\. )", text)
    entries = {int(re.match(r"### Step (\d+)\.", p).group(1)): p for p in parts[1:]}
    entries[n] = entry
    LOG.write_text(parts[0] + "\n".join(entries[k] for k in sorted(entries)))
    return entry


def log_experiments(rows):
    """rows: list of dicts with (some of) EXPERIMENT_COLUMNS; an experiment id that is logged again is replaced."""
    new = pl.DataFrame([{c: r.get(c) for c in EXPERIMENT_COLUMNS} for r in rows], schema={c: (pl.Float64 if c.endswith(("mean", "sd", "seconds", "mb")) else pl.Utf8) for c in EXPERIMENT_COLUMNS})
    if EXPERIMENTS.exists():
        old = pl.read_csv(EXPERIMENTS, schema=new.schema)
        new = pl.concat([old.filter(~pl.col("id").is_in(new["id"].to_list())), new])
    new.write_csv(EXPERIMENTS)
    return new.height


def save_table(name, df):
    df.write_csv(TABLES / f"{name}.csv")
    return df


def cached(name, fn):
    path = TABLES / f"{name}.parquet"
    if path.exists() and not FORCE:
        return pl.read_parquet(path)
    df = fn()
    df.write_parquet(path)
    df.write_csv(TABLES / f"{name}.csv")
    return df
