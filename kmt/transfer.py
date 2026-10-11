"""Sister-language (Hindi) transfer pipeline.

  1. pivot    English -> Hindi for your real training pairs (gives Hindi<->Kumaoni pairs)
  2. lexicon  word-align Hindi<->Kumaoni, write a Hindi->Kumaoni lexicon TSV for REVIEW
  3. synth    relexify a big English-Hindi corpus into pseudo-Kumaoni training data
  4. leaklist build the list of "Hindi-only" words used to measure/penalise Hindi leakage

  python -m kmt.transfer pivot   --data data/train.jsonl --out synth/pivot.jsonl
  python -m kmt.transfer lexicon --pivot synth/pivot.jsonl --out synth/lexicon.tsv
      -> a native speaker marks column 5 with n (wrong) / y (right); blank = unreviewed
  python -m kmt.transfer synth   --lexicon synth/lexicon.tsv --rules kmt/rules.yaml --out synth
  python -m kmt.transfer leaklist --hindi synth/hi_aux.jsonl --kmn data/train.jsonl --out synth/hindi_only.txt
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from .relex import load_lexicon, load_rewrites, relexify
from .text import build_hindi_only, is_dev, words


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_pivot(a):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model).to(dev).eval()
    if dev == "cuda":
        model.half()
    hin = tok.convert_tokens_to_ids("hin_Deva")
    rows, out = _read(a.data), []
    for i in range(0, len(rows), a.bs):
        batch = rows[i:i + a.bs]
        enc = tok([r["src"] for r in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to(dev)
        with torch.inference_mode():
            gen = model.generate(**enc, forced_bos_token_id=hin, num_beams=4, max_new_tokens=128)
        for r, h in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
            out.append({"en": r["src"], "hi": h, "kmn": r["tgt"]})
    _write(a.out, out)
    print(f"wrote {len(out)} pivot rows -> {a.out}")


def cmd_lexicon(a):
    from simalign import SentenceAligner
    aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="i")
    pair, hi_count = Counter(), Counter()
    for r in _read(a.pivot):
        h, k = words(r["hi"]), words(r["kmn"])
        if not h or not k:
            continue
        for i, j in aligner.get_word_aligns(h, k)["itermax"]:
            if is_dev(h[i]) and is_dev(k[j]):
                pair[(h[i], k[j])] += 1
        hi_count.update(w for w in h if is_dev(w))
    best = {}
    for (h, k), c in pair.items():
        p = c / hi_count[h]
        if h != k and c >= a.min_count and p >= a.min_prob and (h not in best or p > best[h][1]):
            best[h] = (k, p, c)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# hindi\tkumaoni\tcount\tprob\tverdict(y/n/blank)  -- sorted by Hindi frequency; review top rows first\n")
        for h in sorted(best, key=lambda w: -hi_count[w]):
            k, p, c = best[h]
            f.write(f"{h}\t{k}\t{c}\t{p:.2f}\t\n")
    print(f"{len(best)} lexicon candidates -> {a.out}  (REVIEW THEM before using)")


def cmd_synth(a):
    from datasets import load_dataset
    lex = load_lexicon(a.lexicon, a.require_reviewed)
    rules = load_rewrites(a.rules)
    ds = load_dataset(a.hf_name, split=a.split).shuffle(seed=0)
    synth, hi_aux = [], []
    for ex in ds:
        t = ex["translation"]
        en, hi = t["en"].strip(), t["hi"].strip()
        if not (3 <= len(en.split()) <= 40 and 3 <= len(hi.split()) <= 45):
            continue
        km, frac = relexify(hi, lex, rules)
        if frac >= a.min_changed:   # skip pairs the lexicon did not touch (would teach "Hindi")
            synth.append({"src": en, "tgt": km, "lang": "kmn"})
        hi_aux.append({"src": en, "tgt": hi, "lang": "hi"})
        if len(synth) >= a.max:
            break
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "synth.jsonl", synth)
    _write(out / "hi_aux.jsonl", hi_aux[: a.max])
    print(f"synthetic en->kmn: {len(synth)}  hindi aux: {min(len(hi_aux), a.max)} -> {out}")


def cmd_leaklist(a):
    hindi = [r["tgt"] for r in _read(a.hindi)]
    kmn = [r["tgt"] for r in _read(a.kmn)]
    lst = sorted(build_hindi_only(hindi, kmn, a.min_hi, a.max_ratio))
    Path(a.out).write_text("\n".join(lst) + "\n", encoding="utf-8")
    print(f"{len(lst)} Hindi-only words -> {a.out}  (skim it: remove anything that IS Kumaoni)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pivot")
    p.add_argument("--data", default="data/train.jsonl")
    p.add_argument("--out", default="synth/pivot.jsonl")
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--bs", type=int, default=32)
    p.set_defaults(fn=cmd_pivot)
    p = sub.add_parser("lexicon")
    p.add_argument("--pivot", default="synth/pivot.jsonl")
    p.add_argument("--out", default="synth/lexicon.tsv")
    p.add_argument("--min_count", type=int, default=3)
    p.add_argument("--min_prob", type=float, default=0.4)
    p.set_defaults(fn=cmd_lexicon)
    p = sub.add_parser("synth")
    p.add_argument("--lexicon", default="synth/lexicon.tsv")
    p.add_argument("--rules", default="kmt/rules.yaml")
    p.add_argument("--out", default="synth")
    p.add_argument("--hf_name", default="cfilt/iitb-english-hindi")
    p.add_argument("--split", default="train")
    p.add_argument("--max", type=int, default=100000)
    p.add_argument("--min_changed", type=float, default=0.1)
    p.add_argument("--require_reviewed", action="store_true")
    p.set_defaults(fn=cmd_synth)
    p = sub.add_parser("leaklist")
    p.add_argument("--hindi", default="synth/hi_aux.jsonl")
    p.add_argument("--kmn", default="data/train.jsonl")
    p.add_argument("--out", default="synth/hindi_only.txt")
    p.add_argument("--min_hi", type=int, default=30)
    p.add_argument("--max_ratio", type=float, default=0.1)
    p.set_defaults(fn=cmd_leaklist)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
