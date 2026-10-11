"""Clean + split the English-Kumaoni data WITHOUT leakage.

  python -m kmt.prep_data --out data

Differences from the original notebook: deterministic split (hash of the English source, so
the same sentence can never be in train and test), exact-duplicate removal, near-duplicate
test items moved back into train, length-ratio filter, script statistics printed.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .text import is_dev, normalize, words


def split_of(key: str) -> str:
    b = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100
    return "train" if b < 90 else "dev" if b < 95 else "test"


def _key(en: str) -> str:
    return re.sub(r"[^\w\s]", "", en.lower()).strip()


def clean_pairs(rows, strip_nukta=False, drop_non_dev=False):
    seen, kept, stats = set(), [], Counter()
    for r in rows:
        en = normalize(r.get("english") or "", strip_nukta)
        km = normalize(r.get("kumaoni") or "", strip_nukta)
        if not en or not km:
            stats["empty"] += 1
            continue
        if en.lower() == km.lower():
            stats["untranslated"] += 1
            continue
        nen, nkm = len(en.split()), len(km.split())
        if nen > 100 or nkm > 100 or not (0.4 <= nkm / nen <= 2.5):
            stats["length_ratio"] += 1
            continue
        dev = sum(is_dev(w) for w in words(km))
        if dev == 0:
            stats["no_devanagari_in_target"] += 1
            if drop_non_dev:
                continue
        if (en, km) in seen:
            stats["duplicate"] += 1
            continue
        seen.add((en, km))
        kept.append({"src": en, "tgt": km})
    return kept, stats


def move_near_dups(splits, thr=0.9):
    """dev/test items whose word set is >= thr Jaccard-similar to a train item go to train."""
    tr_sets = [set(_key(r["src"]).split()) for r in splits["train"]]
    moved = 0
    for name in ("dev", "test"):
        keep = []
        for r in splits[name]:
            a = set(_key(r["src"]).split())
            dup = False
            for b in tr_sets:
                if not a or not b or len(a) > len(b) / thr + 1 or len(b) > len(a) / thr + 1:
                    continue
                if len(a & b) / len(a | b) >= thr:
                    dup = True
                    break
            if dup:
                splits["train"].append(r)
                moved += 1
            else:
                keep.append(r)
        splits[name] = keep
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_name", default="sourabsb/english-kumaoni_translation_dataset")
    ap.add_argument("--out", default="data")
    ap.add_argument("--strip_nukta", action="store_true")
    ap.add_argument("--drop_non_devanagari", action="store_true")
    a = ap.parse_args()

    from datasets import load_dataset
    rows = list(load_dataset(a.hf_name, split="train"))
    kept, stats = clean_pairs(rows, a.strip_nukta, a.drop_non_devanagari)
    splits = {"train": [], "dev": [], "test": []}
    for r in kept:
        splits[split_of(_key(r["src"]))].append(r)
    moved = move_near_dups(splits)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in splits.items():
        with open(out / f"{k}.jsonl", "w", encoding="utf-8") as f:
            for r in v:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    nukta = sum("\u093c" in r["tgt"] for r in kept)
    print(f"input rows: {len(rows)}  kept: {len(kept)}  dropped: {dict(stats)}")
    print(f"near-duplicate dev/test items moved to train: {moved}")
    print({k: len(v) for k, v in splits.items()})
    print(f"target sentences containing a nukta: {nukta} (decide on --strip_nukta using this)")


if __name__ == "__main__":
    main()
