"""LLM client interface, local cache and the query classifier (issue #34).

Import from a notebook in this folder with `from llm import ...`.
Stays here, not in `src/ranking/`, until issue #52.

- The provider sits behind `LLMClient.complete(model, prompt, max_tokens) -> str`, so the
  Anthropic API (now) and Amazon Bedrock (issue #45) are swappable without touching the callers.
- Model IDs and prompt versions live in `configs/llm.json`, never in code.
- Every call goes through `CachedClient`: results are stored in a local file keyed by
  (model, prompt version, normalised input), so a repeated query costs nothing and tests can run
  on recorded fixtures.
- The API key is read from the ANTHROPIC_API_KEY environment variable only.
"""

import hashlib
import json
import os
import re
from pathlib import Path


def normalize(text):
    """Lowercase, trim and collapse whitespace, so trivial variants share a cache entry."""
    return re.sub(r"\s+", " ", text.strip().lower())


class LLMClient:
    """Interface: return the model's text answer for a prompt."""

    def complete(self, model, prompt, max_tokens):  # pragma: no cover - interface
        raise NotImplementedError


class AnthropicClient(LLMClient):
    """Anthropic API. The key comes from the ANTHROPIC_API_KEY environment variable."""

    def __init__(self):
        import anthropic

        if "ANTHROPIC_API_KEY" not in os.environ:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic()

    def complete(self, model, prompt, max_tokens):
        msg = self.client.messages.create(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")


class FakeClient(LLMClient):
    """Test double: returns queued answers (or a function of the prompt) and counts the calls."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = 0

    def complete(self, model, prompt, max_tokens):
        self.calls += 1
        if callable(self.answers):
            return self.answers(prompt)
        return self.answers[min(self.calls - 1, len(self.answers) - 1)]


class CachedClient(LLMClient):
    """Wrap a client with a file cache keyed by (model, prompt_version, normalised key text)."""

    def __init__(self, client, cache_dir, prompt_version, fixtures_only=False):
        self.client, self.dir, self.version, self.fixtures_only = client, Path(cache_dir), prompt_version, fixtures_only
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    def _path(self, model, key):
        digest = hashlib.sha256(f"{model}|{self.version}|{normalize(key)}".encode()).hexdigest()
        return self.dir / f"{digest}.json"

    def complete_cached(self, model, prompt, key, max_tokens):
        path = self._path(model, key)
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text())["answer"]
        if self.fixtures_only:
            raise KeyError(f"no recorded response for {key!r} and fixtures-only mode is on")
        self.misses += 1
        answer = self.client.complete(model, prompt, max_tokens)
        path.write_text(json.dumps({"model": model, "prompt_version": self.version, "key": normalize(key), "answer": answer}))
        return answer

    def complete(self, model, prompt, max_tokens):
        return self.complete_cached(model, prompt, prompt, max_tokens)


PROMPT = """You classify a shopper's search term into the product categories of a shop.

Categories (synthetic names):
{categories}

Search term: "{query}"

Answer with JSON only, in this exact form:
{{"categories": [{{"name": "<category>", "confidence": <0 to 1>}}, ...], "out_of_scope": <true or false>}}
Rules: at most 3 categories, best first, names copied exactly from the list, confidences between 0 and 1
that add up to at most 1. Set "out_of_scope" to true if the term is about something the shop does not sell
(then "categories" may be empty)."""


def validate(answer, category_names):
    """Parse and check a classifier answer. Returns (result dict, None) or (None, error message)."""
    m = re.search(r"\{.*\}", answer, flags=re.DOTALL)
    if not m:
        return None, "no JSON object in the answer"
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as err:
        return None, f"invalid JSON: {err}"
    cats, oos = data.get("categories"), data.get("out_of_scope")
    if not isinstance(oos, bool):
        return None, "out_of_scope must be true or false"
    if not isinstance(cats, list) or len(cats) > 3:
        return None, "categories must be a list of at most 3"
    names = set(category_names)
    total = 0.0
    for c in cats:
        if not isinstance(c, dict) or c.get("name") not in names:
            return None, f"unknown category {c!r}"
        conf = c.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not 0 <= conf <= 1:
            return None, f"bad confidence {conf!r}"
        total += conf
    if total > 1.001:
        return None, f"confidences add up to {total:.3f} (more than 1)"
    return {"categories": [{"name": c["name"], "confidence": float(c["confidence"])} for c in cats], "out_of_scope": oos}, None


class Classifier:
    def __init__(self, client, category_names, config_path="configs/llm.json", cache=None):
        self.cfg = json.loads(Path(config_path).read_text())
        self.names = list(category_names)
        self.cache = cache or CachedClient(client, self.cfg["cache_dir"], self.cfg["classifier_prompt_version"])
        self.calls_failed = 0

    def classify(self, query):
        """Top categories with confidences and an out-of-scope flag. A malformed answer is retried once."""
        if not query or not query.strip():
            return {"categories": [], "out_of_scope": True, "error": "empty query"}
        query = query.strip()[:300]
        prompt = PROMPT.format(categories="\n".join(f"- {n}" for n in self.names), query=query)
        model = self.cfg["classifier_model"]
        error = None
        for attempt in range(2):
            key = query if attempt == 0 else f"{query} (retry)"
            answer = self.cache.complete_cached(model, prompt, key, self.cfg["max_tokens"])
            result, error = validate(answer, self.names)
            if result is not None:
                return result
        self.calls_failed += 1
        return {"categories": [], "out_of_scope": True, "error": error}
