"""Relexification: turn Hindi text into pseudo-Kumaoni with a reviewed lexicon + rewrite rules.

Pure python (no torch) so it can be unit-tested and run anywhere.
"""
import re
from .text import DEV_RUN, is_dev

_SPLIT = re.compile("(" + DEV_RUN + ")")


def load_lexicon(path, require_reviewed=False):
    """TSV columns: hindi, kumaoni, count, prob, verdict.  verdict: 'n' = reject, 'y' = accept,
    blank = unreviewed (accepted unless require_reviewed=True)."""
    lex = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            hi, km = cols[0], cols[1]
            verdict = cols[4].strip().lower() if len(cols) > 4 else ""
            if verdict == "n" or (require_reviewed and verdict != "y"):
                continue
            if hi and km and hi != km:
                lex[hi] = km
    return lex


def load_rewrites(path):
    """YAML list under key `rewrites`: [{pattern, repl, note}] applied in order after lexicon."""
    if not path:
        return []
    import yaml
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return [(re.compile(r["pattern"]), r["repl"]) for r in (data.get("rewrites") or [])]


def apply_lexicon(text, lex):
    parts = _SPLIT.split(text)
    n_dev = n_changed = 0
    for i, p in enumerate(parts):
        if i % 2 == 1 and is_dev(p):  # odd indices are the captured Devanagari runs
            n_dev += 1
            new = lex.get(p)
            if new is not None:
                parts[i] = new
                n_changed += 1
    return "".join(parts), n_changed, n_dev


def relexify(text, lex, rewrites=()):
    """Return (pseudo_kumaoni, fraction_of_devanagari_words_changed)."""
    out, changed, total = apply_lexicon(text, lex)
    for pat, repl in rewrites:
        new = pat.sub(repl, out)
        if new != out:
            changed += 1
        out = new
    return out, changed / max(1, total)
