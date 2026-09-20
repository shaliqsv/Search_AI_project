"""Helpers for the EDA notebook (`notebooks/EDA.ipynb`), following `_doc/eda_guide.md`.

- `log_step` writes the decision log entry of a step to `eda/eda_log.md` in the guide's exact template
  (a step's entry is replaced when the step is rerun).
- `save_table` stores every comparison table in `eda/tables/`, so losing options are kept.
- `cached` runs an expensive computation once and reloads it afterwards; set the environment variable
  EDA_FORCE=1 to recompute everything from the raw data.
"""

import os
import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
EDA = ROOT / "eda"
TABLES, FIGURES = EDA / "tables", EDA / "figures"
LOG = EDA / "eda_log.md"
SEED = 42
FORCE = os.environ.get("EDA_FORCE") == "1"
for d in (TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

TEMPLATE = """### Step {n}. {title}
Kind: {kind}
Options run: {options}
Observed: {observed}
Decision: {decision}
Why: {why}
Rejected: {rejected}
Confidence: {confidence}
Affects later steps: {affects}
Needs human input: {human}
"""


def log_step(n, title, kind, options, observed, decision, why, rejected, confidence, affects, human="no"):
    """Write (or replace) the log entry of step n. Returns the entry text."""
    entry = TEMPLATE.format(n=n, title=title, kind=kind, options=options, observed=observed, decision=decision,
                            why=why, rejected=rejected, confidence=confidence, affects=affects, human=human)
    text = LOG.read_text() if LOG.exists() else "# EDA log\n\nOne entry per step of `_doc/eda_guide.md`, in the guide's template.\n\n"
    parts = re.split(r"(?m)^(?=### Step \d+\. )", text)
    head, entries = parts[0], {int(re.match(r"### Step (\d+)\.", p).group(1)): p for p in parts[1:]}
    entries[n] = entry + "\n"
    LOG.write_text(head + "".join(entries[k] for k in sorted(entries)))
    return entry


def save_table(name, df):
    """Save a comparison table (polars DataFrame) as CSV in eda/tables/ and return it."""
    path = TABLES / f"{name}.csv"
    df.write_csv(path)
    return df


def cached(name, fn):
    """Run fn() once and store the resulting polars DataFrame in eda/tables/; later runs reload it."""
    path = TABLES / f"{name}.parquet"
    if path.exists() and not FORCE:
        return pl.read_parquet(path)
    df = fn()
    df.write_parquet(path)
    df.write_csv(TABLES / f"{name}.csv")
    return df
