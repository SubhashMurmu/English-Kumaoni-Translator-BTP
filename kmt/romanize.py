"""Convert the Roman-script Kumaoni side of the corpus to Devanagari (word-level, human-in-the-loop)."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

_ROMAN = re.compile(r"[A-Za-z]+")
_cache = {}


def auto(word: str) -> str:
    """ITRANS on the lower-cased word (ITRANS is case-sensitive, so lower-casing is required)."""
    w = word.lower()
    if w not in _cache:
        from indic_transliteration import sanscript
        d = sanscript.transliterate(w, sanscript.ITRANS, sanscript.DEVANAGARI)
        _cache[w] = re.sub("\u094d$", "", d)   # no trailing halant
    return _cache[w]


def load_overrides(path):
    ov = {}
    if path and Path(path).exists():
        for line in open(path, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].strip() and parts[1].strip() and not line.startswith("#"):
                ov[parts[0].strip().lower()] = parts[1].strip()
    return ov


def to_devanagari(text, overrides=None):
    ov = overrides or {}
    out = _ROMAN.sub(lambda m: ov.get(m.group(0).lower()) or auto(m.group(0)), text)
    return re.sub(r"\.(\s*)$", "।", out)       # sentence-final full stop -> danda


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cmd_vocab(a):
    cnt = Counter()
    for sp in ("train", "dev", "test"):
        for r in _read(f"{a.data}/{sp}.jsonl"):
            cnt.update(w.lower() for w in _ROMAN.findall(r["tgt"]))
    total = sum(cnt.values())
    rows = cnt.most_common()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# roman\tfreq\tauto_devanagari\n")
        for w, c in rows:
            f.write(f"{w}\t{c}\t{auto(w)}\n")
    print(f"{len(rows)} distinct words, {total} tokens -> {a.out}")
    run, cum = 0, {}
    for i, (_, c) in enumerate(rows, 1):
        run += c
        if i in (100, 300, 500, 1000, 2000, 3000):
            cum[i] = round(100 * run / total, 1)
    print("token coverage of the top-N words (%):", cum)
    leftover = [w for w, _ in rows if re.search("[A-Za-z]", auto(w))]
    print(f"{len(leftover)} words still contain Latin letters after conversion, e.g. {leftover[:10]}")


def cmd_convert(a):
    ov = load_overrides(a.overrides)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_tok = n_ov = 0
    for sp in ("train", "dev", "test"):
        rows = _read(f"{a.data}/{sp}.jsonl")
        for r in rows:
            ws = [w.lower() for w in _ROMAN.findall(r["tgt"])]
            n_tok += len(ws)
            n_ov += sum(w in ov for w in ws)
            r["tgt"] = to_devanagari(r["tgt"], ov)
        with open(out / f"{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(ov)} overrides loaded; {100 * n_ov / max(1, n_tok):.1f}% of Kumaoni tokens use a reviewed form, "
          f"the rest use automatic conversion -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("vocab")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="synth/roman_vocab.tsv")
    p.set_defaults(fn=cmd_vocab)
    p = sub.add_parser("convert")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="data_dev")
    p.add_argument("--overrides", default="synth/overrides.tsv")
    p.set_defaults(fn=cmd_convert)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
