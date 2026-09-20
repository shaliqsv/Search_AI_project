"""Shim: the metrics moved to `src/ranking/eval/metrics.py` (plan layout). Kept so older notebooks keep importing `metrics`."""
from ranking.eval.metrics import *
from ranking.eval.metrics import _dcg, _dedupe  # noqa: F401
