# D1: OTTO dataset license and source

**Phase**: 1, Dataset creation — step 1 (inventory sources), gating step for committing any OTTO-derived data.

**Decision**: Source the data via the Kaggle *dataset* `otto/recsys-dataset` (`kaggle datasets download -d otto/recsys-dataset`), not the `otto-recommender-system` *competition* page. The dataset is released by OTTO GmbH under **CC-BY 4.0** (attribution required); the companion code at `github.com/otto-de/recsys-dataset` is MIT. CC-BY 4.0 permits redistributing derived/processed data (samples, aggregates, snapshots) in this public repo, provided attribution is included.

**Why**: the competition page's rules may carry competition-specific restrictions on external use that don't apply to the dataset's own CC-BY 4.0 release. Using the dataset endpoint directly avoids that ambiguity and gives an unambiguous, permissive license to point to.

**Alternatives considered**: downloading from the competition page instead (rejected — same underlying files, but tangled with competition rules instead of the cleaner CC-BY 4.0 grant); treating the license as unresolved and keeping all OTTO-derived tables out of git indefinitely (rejected now that the license is confirmed permissive — this is what the old `.gitignore` comment was hedging against).

**What this unlocks**: OTTO-derived samples, snapshots, category/query synthesis outputs, and eval tables can be committed to this repo, as long as attribution to OTTO GmbH / the CC-BY 4.0 dataset is recorded somewhere in the repo (README or a NOTICE file). Raw full-scale event dumps still shouldn't be committed — not a license issue, just repo hygiene (`data/` stays gitignored; only the frozen, sampled snapshot and its derived tables go in git).

**Source**: https://github.com/otto-de/recsys-dataset, https://www.kaggle.com/datasets/otto/recsys-dataset
