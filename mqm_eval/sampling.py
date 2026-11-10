"""Deterministic stratified sampling of source sentences from the held-out test split.

Strata: templated vs non-templated sources (the README notes templated sentences inflate
automatic scores) x sentence-length bins. Selection is by hash order, so the same seed and
input always give the same sample.
"""
import hashlib
import json
import re
from collections import Counter, defaultdict


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def src_id(src):
    return "s" + hashlib.md5(src.strip().lower().encode("utf-8")).hexdigest()[:8]


def _order(seed, key):
    return hashlib.md5(f"{seed}|{key}".encode("utf-8")).hexdigest()


def _tokens(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def template_prefixes(pool_rows, prefix_len=3, min_count=5):
    """Opening word sequences that occur at least `min_count` times in the pool."""
    cnt = Counter(tuple(_tokens(r["src"])[:prefix_len]) for r in pool_rows)
    return {p for p, c in cnt.items() if len(p) == prefix_len and c >= min_count}


def _length_bins(items, n_bins):
    ordered = sorted(items, key=lambda it: it["n_words"])
    size = max(1, -(-len(ordered) // n_bins))
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


def _take(items, quota, seed, n_bins):
    """Round-robin over length bins, hash order inside each bin."""
    bins = [sorted(b, key=lambda it: _order(seed, it["src"])) for b in _length_bins(items, n_bins)]
    out, i = [], 0
    while len(out) < quota and any(bins):
        b = bins[i % len(bins)]
        if b:
            out.append(b.pop(0))
        i += 1
    return out


def stratified_sample(rows, n, seed=13, templated_share=0.3, pool_rows=None,
                      prefix_len=3, min_prefix=5, n_len_bins=3):
    """Pick `n` distinct sources. Returns dicts with src_id, src, tgt, n_words, templated."""
    pref = template_prefixes(pool_rows if pool_rows is not None else rows, prefix_len, min_prefix)
    seen, items = set(), []
    for r in rows:
        sid = src_id(r["src"])
        if sid in seen:
            continue
        seen.add(sid)
        toks = _tokens(r["src"])
        items.append({"src_id": sid, "src": r["src"], "tgt": r["tgt"], "n_words": len(toks),
                      "templated": tuple(toks[:prefix_len]) in pref})
    if len(items) < n:
        raise ValueError(f"only {len(items)} distinct sources available, need {n}")
    tmpl = [it for it in items if it["templated"]]
    non = [it for it in items if not it["templated"]]
    n_t = min(round(n * templated_share), len(tmpl))
    n_n = min(n - n_t, len(non))
    picked = _take(tmpl, n_t, seed, n_len_bins) + _take(non, n_n, seed, n_len_bins)
    if len(picked) < n:      # one group ran short: top up from whatever is left
        chosen = {it["src_id"] for it in picked}
        rest = [it for it in items if it["src_id"] not in chosen]
        picked += _take(rest, n - len(picked), seed, n_len_bins)
    return sorted(picked, key=lambda it: _order(seed, it["src_id"]))


def strata_summary(sample):
    c = defaultdict(int)
    for it in sample:
        c[("templated" if it["templated"] else "non-templated")] += 1
    return dict(c)
