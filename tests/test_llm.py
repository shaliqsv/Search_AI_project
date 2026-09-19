import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from llm import CachedClient, Classifier, FakeClient, normalize, validate

NAMES = ["Dresses", "Skirts", "Watches", "Rugs"]
GOOD = json.dumps({"categories": [{"name": "Dresses", "confidence": 0.8}, {"name": "Skirts", "confidence": 0.1}], "out_of_scope": False})


def make(tmp_path, answers, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "llm.json"
    cfg.write_text(json.dumps({"classifier_model": "m", "classifier_prompt_version": "v1", "max_tokens": 100, "cache_dir": str(tmp_path / "cache")}))
    client = FakeClient(answers)
    cache = CachedClient(client, tmp_path / "cache", "v1", **kw)
    return Classifier(client, NAMES, config_path=cfg, cache=cache), client


def test_same_query_makes_one_call(tmp_path):
    clf, client = make(tmp_path, [GOOD])
    a = clf.classify("Maxi Dress")
    b = clf.classify("  maxi   dress ")          # same normalised query
    assert a == b and client.calls == 1


def test_output_is_validated(tmp_path):
    ok, err = validate(GOOD, NAMES)
    assert err is None and ok["categories"][0]["name"] == "Dresses" and ok["out_of_scope"] is False
    for bad in [
        '{"categories": [{"name": "Hats", "confidence": 0.5}], "out_of_scope": false}',           # unknown category
        '{"categories": [{"name": "Dresses", "confidence": 0.9}, {"name": "Skirts", "confidence": 0.5}], "out_of_scope": false}',  # sum > 1
        '{"categories": [], "out_of_scope": "no"}',                                               # flag is not a bool
        '{"categories": [{"name": "Dresses", "confidence": 1.5}], "out_of_scope": false}',        # confidence range
        "sorry, I cannot help with that",                                                          # no JSON
    ]:
        assert validate(bad, NAMES)[0] is None
    four = json.dumps({"categories": [{"name": n, "confidence": 0.1} for n in NAMES], "out_of_scope": False})
    assert validate(four, NAMES)[0] is None                                                        # more than 3


def test_malformed_answer_is_retried_once_then_out_of_scope(tmp_path):
    clf, client = make(tmp_path, ["not json", GOOD])
    assert clf.classify("dress")["categories"][0]["name"] == "Dresses" and client.calls == 2       # retry succeeds
    clf2, client2 = make(tmp_path / "b", ["not json", "still not json"])
    res = clf2.classify("dress")
    assert res["out_of_scope"] is True and "error" in res and client2.calls == 2                   # gives up after one retry


def test_empty_and_long_queries(tmp_path):
    clf, client = make(tmp_path, [GOOD])
    assert clf.classify("   ")["error"] == "empty query" and client.calls == 0
    clf.classify("dress " * 500)
    assert client.calls == 1


def test_fixtures_only_mode_never_calls_the_api(tmp_path):
    clf, _client = make(tmp_path, [GOOD])
    clf.classify("dress")                                  # records the response
    clf2, client2 = make(tmp_path, [GOOD], fixtures_only=True)
    assert clf2.classify("dress") == clf.classify("dress") and client2.calls == 0
    with pytest.raises(KeyError):
        clf2.classify("a query with no recording")


def test_normalize():
    assert normalize("  Maxi   DRESS ") == "maxi dress"


def test_no_api_key_in_the_repository():
    pattern = re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}")
    hits = [str(p) for p in ROOT.rglob("*") if p.is_file() and ".venv" not in p.parts and ".git" not in p.parts
            and p.suffix in {".py", ".json", ".md", ".ipynb", ".toml", ".yaml", ".yml", ".jsonl"} and pattern.search(p.read_text(errors="ignore"))]
    assert not hits, hits
