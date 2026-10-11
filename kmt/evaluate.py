"""Evaluate on the held-out test set: chrF++, BLEU, Hindi-leakage; dump outputs for human review.

  python -m kmt.evaluate --model runs/B/final --preset mbart --hindi_only synth/hindi_only.txt --alpha 1.0
"""
import argparse
import json

import sacrebleu

from .infer import Translator, load_hindi_only
from .text import leak_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--preset", default="mbart")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--hindi_only", default=None)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--out", default="eval_outputs.tsv")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    lo = load_hindi_only(a.hindi_only)
    tr = Translator(a.model, a.preset, lo)
    hyp = tr.translate([r["src"] for r in rows], num_beams=a.beams, n_best=a.beams, alpha=a.alpha)
    ref = [r["tgt"] for r in rows]
    chrf = sacrebleu.corpus_chrf(hyp, [ref], word_order=2)
    bleu = sacrebleu.corpus_bleu(hyp, [ref])
    print(f"chrF++ {chrf.score:.2f} | BLEU {bleu.score:.2f}")
    if lo:
        print(f"Hindi-leak rate: system {sum(leak_rate(h, lo) for h in hyp) / len(hyp):.4f}"
              f" vs reference {sum(leak_rate(r, lo) for r in ref) / len(ref):.4f}")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("source\treference\thypothesis\n")
        for r, h in zip(rows, hyp):
            f.write(f"{r['src']}\t{r['tgt']}\t{h}\n")
    print(f"wrote {a.out} - give 100 random rows to a native speaker (adequacy/fluency 1-5, 'is this Kumaoni?')")


if __name__ == "__main__":
    main()
