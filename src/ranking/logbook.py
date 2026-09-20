"""Logging helpers for the modelling phase (`_doc/modeling_guide.md`): the step log, the experiment table and cached tables.

- `log_step` writes one entry per step to `modeling/model_log.md` (replaced when the step is rerun).
- `log_experiments` appends every experiment, including failures, to `modeling/experiments.csv`; every row carries the seed,
  the data snapshot id (hash of the events sample) and the code commit SHA (rule 10).
- `save_table` and `cached` keep comparison tables in `modeling/tables/` (MODEL_FORCE=1 recomputes everything).
"""

import json
import os
import re
import subprocess
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
MODELING = ROOT / "modeling"
TABLES, FIGURES, ARTIFACTS = MODELING / "tables", MODELING / "figures", MODELING / "artifacts"
LOG, EXPERIMENTS = MODELING / "model_log.md", MODELING / "experiments.csv"
SEED = 42
FORCE = os.environ.get("MODEL_FORCE") == "1"
for _d in (TABLES, FIGURES, ARTIFACTS):
    _d.mkdir(parents=True, exist_ok=True)

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
EXPERIMENT_COLUMNS = ["id", "step", "description", "features", "model", "parameters", "seeds", "metrics", "fit_seconds", "predict_seconds",
                      "size_mb", "notes", "snapshot_id", "commit_sha"]


def snapshot_id():
    manifest = json.loads((ROOT / "data" / "sample" / "sample_manifest.json").read_text())
    return manifest["sha256"]["events.parquet"][:12]


def commit_sha():
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def log_step(n, title, kind, options, observed, decision, why, rejected, confidence, affects, human="no"):
    entry = TEMPLATE.format(n=n, title=title, kind=kind, options=options, observed=observed, decision=decision, why=why,
                            rejected=rejected, confidence=confidence, affects=affects, human=human)
    text = LOG.read_text() if LOG.exists() else "# Modelling log\n\nOne entry per step of `_doc/modeling_guide.md`.\n\n"
    parts = re.split(r"(?m)^(?=### Step \d+\. )", text)
    entries = {int(re.match(r"### Step (\d+)\.", p).group(1)): p for p in parts[1:]}
    entries[n] = entry
    LOG.write_text(parts[0] + "\n".join(entries[k] for k in sorted(entries)))
    return entry


def log_experiments(rows):
    """rows: dicts with (some of) EXPERIMENT_COLUMNS; `metrics` may be a dict (stored as JSON). An id logged again is replaced."""
    snap, sha = snapshot_id(), commit_sha()
    clean = []
    for r in rows:
        r = {c: r.get(c) for c in EXPERIMENT_COLUMNS} | {"snapshot_id": snap, "commit_sha": sha}
        r["metrics"] = json.dumps(r["metrics"]) if isinstance(r["metrics"], dict) else r["metrics"]
        clean.append({k: (None if v is None else str(v)) for k, v in r.items()})
    new = pl.DataFrame(clean, schema={c: pl.Utf8 for c in EXPERIMENT_COLUMNS})
    if EXPERIMENTS.exists() and EXPERIMENTS.stat().st_size > 0:
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
