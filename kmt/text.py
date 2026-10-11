"""Pure-python text utilities: normalisation, Hindi-leakage metric, translation-memory guard."""
import re
import unicodedata
from collections import Counter

_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff]")
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})
_WS = re.compile(r"\s+")
# Devanagari letters/signs, excluding danda (U+0964/0965) so punctuation is not a "word".
DEV_RUN = r"[\u0900-\u0963\u0966-\u097F]+"
_TOKEN = re.compile(DEV_RUN + r"|\w+|[^\w\s]")
_DEV_ONLY = re.compile(r"^" + DEV_RUN + r"$")


def normalize(text: str, strip_nukta: bool = False) -> str:
    """Conservative orthographic normalisation.

    NFC already turns precomposed nukta letters (U+0958..095F) into base + U+093C, so the
    same letter is always one byte sequence. `strip_nukta` additionally removes U+093C; only
    enable it if your corpus uses the nukta inconsistently (count it first!).
    Deliberately NOT done: merging chandrabindu/anusvara or any spelling "correction" --
    those are linguistic decisions to make with a native speaker, not blindly in code.
    """
    t = unicodedata.normalize("NFC", text or "")
    t = _ZERO_WIDTH.sub("", t)
    t = t.translate(_DEV_DIGITS).translate(_QUOTES)
    if strip_nukta:
        t = t.replace("\u093c", "")
    return _WS.sub(" ", t).strip()


def words(text: str):
    """Tokenise into Devanagari words, latin/number words and single punctuation marks."""
    return _TOKEN.findall(text)


def is_dev(tok: str) -> bool:
    return bool(_DEV_ONLY.match(tok))


# ---------------------------------------------------------------- Hindi leakage ----------
def build_hindi_only(hindi_sents, kumaoni_sents, min_hi=30, max_ratio=0.1):
    """Data-driven list of words that are common in Hindi but (almost) absent in Kumaoni.

    A word qualifies when its relative frequency in the Kumaoni corpus is <= max_ratio times
    its relative frequency in the Hindi corpus. No hand-written word list: it adapts to your
    corpus (and you should still eyeball the output with a native speaker).
    """
    hi = Counter(w for s in hindi_sents for w in words(s) if is_dev(w))
    km = Counter(w for s in kumaoni_sents for w in words(s) if is_dev(w))
    nh, nk = max(1, sum(hi.values())), max(1, sum(km.values()))
    out = set()
    for w, c in hi.items():
        if c >= min_hi and (km.get(w, 0) / nk) <= max_ratio * (c / nh):
            out.add(w)
    return out


def leak_rate(text: str, hindi_only) -> float:
    toks = [w for w in words(text) if is_dev(w)]
    return sum(t in hindi_only for t in toks) / max(1, len(toks))


# ---------------------------------------------------------------- TM guard ---------------
_NEG = {"not", "no", "never", "none", "nobody", "nothing", "nowhere", "without",
        "cannot", "neither", "nor"}


def _nums(s):
    return re.findall(r"\d+(?:[.,]\d+)*", s)


def _neg(s):
    ws = re.findall(r"[a-z']+", s.lower())
    return any(w in _NEG or w.endswith("n't") for w in ws)


def _caps(s):
    ws = re.findall(r"[A-Za-z']+", s)
    return {w for w in ws[1:] if w[:1].isupper()}


def tm_compatible(query: str, cached_src: str) -> bool:
    """Embedding similarity ignores numbers, names and negation; require them to match."""
    return (_nums(query) == _nums(cached_src)
            and _neg(query) == _neg(cached_src)
            and _caps(query) == _caps(cached_src))
