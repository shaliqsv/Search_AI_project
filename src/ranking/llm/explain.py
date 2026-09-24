"""Explainer support (plan D18): the numeric payload, the templated fallback and the faithfulness test.

Claude sees ONLY the payload of a result and must cite only what it contains. The faithfulness test parses numbers and feature-group names
from an explanation and checks them against the payload; the templated explanation is both the fallback and the baseline.
"""

import re

GROUP_VOCAB = ("co-visitation", "popularity", "category popularity", "recent popularity", "cart rate", "order rate", "cart and order rates", "session length", "session", "recency", "two-tower similarity")
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%?)")


def make_payload(rank, item_id, category, p_click, p_cart, p_order, blended, groups):
    """`groups`: {feature group name: contribution to the blended score}, largest first. Everything Claude may cite is in here."""
    return {"rank": int(rank), "item_id": int(item_id), "category": category, "p_click": round(float(p_click), 3), "p_cart": round(float(p_cart), 3), "p_order": round(float(p_order), 3),
            "score": round(float(blended), 3), "top_groups": {k: round(float(v), 3) for k, v in list(groups.items())[:3]}}


def _allowed_numbers(payload):
    vals = {float(payload["rank"]), float(payload["item_id"]), payload["p_click"], payload["p_cart"], payload["p_order"], payload["score"], *payload["top_groups"].values()}
    vals = vals | {abs(v) for v in vals}                                             # the sign is written next to the number, not part of it
    return vals | {round(v * 100, 1) for v in vals if 0 <= v <= 1}                     # a probability may be quoted as a percentage


def templated_explanation(payload):
    g = ", ".join(f"{k} ({v:+.3f})" for k, v in payload["top_groups"].items())
    return (f"Rank {payload['rank']} in {payload['category']}: click {payload['p_click']:.3f}, cart {payload['p_cart']:.3f}, order {payload['p_order']:.3f}, "
            f"blended score {payload['score']:.3f}. Main drivers: {g}.")


def faithfulness(text, payload, tol=0.0006):
    """Returns (passed, unsupported numbers, unsupported feature-group names). A number is supported if it equals a payload value within `tol`
    (or its percentage form); a group name is supported if it is one of the payload's top groups."""
    allowed = _allowed_numbers(payload); bad_numbers = []
    for m in _NUMBER.finditer(text):
        v = float(m.group(1))
        if not any(abs(v - a) <= max(tol, 0.0006 * abs(a)) or (m.group(2) and abs(v - a * 100) <= 0.15) for a in allowed): bad_numbers.append(m.group(0))
    low = text.lower(); named = {g for g in GROUP_VOCAB if g in low}; used = {g.lower() for g in payload["top_groups"]}
    bad_groups = sorted(g for g in named if not any(g in u or u in g for u in used))
    return (not bad_numbers and not bad_groups), bad_numbers, bad_groups
