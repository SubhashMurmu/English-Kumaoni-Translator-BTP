"""Automatic, rule-based diagnostics of the S1 outputs. No annotators and no Kumaoni knowledge needed.

These are NOT human judgments and NOT MQM. They flag patterns that are suspicious in any
language, so a person can look at them first:

  empty              the output is empty
  copy_of_source     the output is the English source unchanged
  copied_english     content words (4+ letters) that appear in the English source but NOT in the
                     dataset reference. Names and loanwords can be legitimate, so these are candidates
  repetition         a word 3-gram repeated 3+ times, or one word repeated 3+ times in a row
  number_mismatch    the source and the reference contain the same numbers but the output does not
  length_outlier     output/reference length ratio below 0.5 or above 2.0 (reference of 4+ words only)

Overlap scores (chrF++, BLEU) are computed against the dataset references, which are unverified
and spelled inconsistently, so differences between groups are suggestive, not conclusive.
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .inference import bootstrap_ci
from .load import read_jsonl

TOKEN = re.compile(r"[^\W\d_]+|\d+(?:[.,]\d+)*", re.UNICODE)
FLAGS = ["empty", "copy_of_source", "copied_english", "repetition", "number_mismatch", "length_outlier"]


def _tokens(s):
    return TOKEN.findall(s.lower())


def _words(s):
    return [t for t in _tokens(s) if not t[0].isdigit()]


def _numbers(s):
    return sorted(t for t in _tokens(s) if t[0].isdigit())


def segment_diagnostics(src, ref, hyp):
    """Flags and details for one segment."""
    hw, sw, rw = _words(hyp), set(_words(src)), set(_words(ref))
    copied = sorted({w for w in hw if len(w) >= 4 and w in sw and w not in rw})
    tri = Counter(zip(hw, hw[1:], hw[2:]))
    run = best = 1
    for a, b in zip(hw, hw[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    n_ref = len(_words(ref))
    ratio = len(hw) / n_ref if n_ref else None
    flags = {
        "empty": not hyp.strip(),
        "copy_of_source": bool(hyp.strip()) and hyp.strip().lower() == src.strip().lower(),
        "copied_english": bool(copied),
        "repetition": (max(tri.values()) >= 3 if tri else False) or (len(hw) >= 3 and best >= 3),
        "number_mismatch": _numbers(src) == _numbers(ref) and _numbers(hyp) != _numbers(src),
        "length_outlier": n_ref >= 4 and ratio is not None and (ratio < 0.5 or ratio > 2.0),
    }
    return {"flags": flags, "copied_tokens": copied, "length_ratio": ratio}


def _sacrebleu():
    try:
        import sacrebleu
    except ImportError as e:
        raise SystemExit("sacrebleu is required: pip install sacrebleu") from e
    return sacrebleu


def _corpus_ci(hyps, refs, kind, B, seed):
    sb = _sacrebleu()
    fn = (lambda h, r: sb.corpus_chrf(h, [r], word_order=2).score) if kind == "chrF++" \
        else (lambda h, r: sb.corpus_bleu(h, [r]).score)
    rng = np.random.default_rng(seed)
    n = len(hyps)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        vals.append(fn([hyps[k] for k in i], [refs[k] for k in i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return fn(hyps, refs), float(lo), float(hi)


def _groups(rows):
    n = sorted(r["n_words"] for r in rows)
    t1, t2 = n[len(n) // 3], n[2 * len(n) // 3]
    return {
        "all segments": rows,
        "templated sources": [r for r in rows if r["templated"]],
        "non-templated sources": [r for r in rows if not r["templated"]],
        f"short sources (<= {t1} words)": [r for r in rows if r["n_words"] <= t1],
        f"medium sources ({t1 + 1}-{t2} words)": [r for r in rows if t1 < r["n_words"] <= t2],
        f"long sources (> {t2} words)": [r for r in rows if r["n_words"] > t2],
    }


def run_diagnostics(run_dir, out_dir, B=500, seed=13):
    from .report import BLUE, GRID, INK, INK2, SURFACE, _md, _num, _style, _write_csv
    import matplotlib.pyplot as plt

    run, out = Path(run_dir), Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    sample = {r["src_id"]: r for r in read_jsonl(run / "sample.jsonl")}
    outputs = {r["src_id"]: r["text"] for r in read_jsonl(run / "s1_outputs.jsonl")}
    sb = _sacrebleu()
    rows = []
    for sid, s in sample.items():
        hyp = outputs[sid]
        d = segment_diagnostics(s["src"], s["tgt"], hyp)
        rows.append({"src_id": sid, "src": s["src"], "ref": s["tgt"], "hyp": hyp, "n_words": s["n_words"],
                     "templated": bool(s["templated"]), "length_ratio": d["length_ratio"],
                     "chrF++": sb.sentence_chrf(hyp, [s["tgt"]], word_order=2).score,
                     "copied_tokens": " ".join(d["copied_tokens"]), **d["flags"],
                     "n_flags": sum(d["flags"].values())})

    summary = []
    for name, g in _groups(rows).items():
        if len(g) < 3:
            continue
        hyps, refs = [r["hyp"] for r in g], [r["ref"] for r in g]
        chrf, c_lo, c_hi = _corpus_ci(hyps, refs, "chrF++", B, seed)
        bleu, b_lo, b_hi = _corpus_ci(hyps, refs, "BLEU", B, seed)
        row = {"group": name, "n": len(g), "chrF++": chrf, "chrF++_ci_low": c_lo, "chrF++_ci_high": c_hi,
               "BLEU": bleu, "BLEU_ci_low": b_lo, "BLEU_ci_high": b_hi}
        for f in FLAGS:
            row[f"share_{f}"] = sum(r[f] for r in g) / len(g)
        row["share_any_flag"] = sum(r["n_flags"] > 0 for r in g) / len(g)
        summary.append(row)

    flagged = sorted((r for r in rows if r["n_flags"] > 0), key=lambda r: (-r["n_flags"], r["chrF++"]))
    _write_csv(out / "tables" / "diagnostics_summary.csv", summary)
    _write_csv(out / "tables" / "segment_diagnostics.csv", rows)
    _write_csv(out / "tables" / "flagged_segments.csv", flagged)

    # figure: sentence-level chrF++ by group (dots) with mean and 95% CI
    groups = [(n, g) for n, g in _groups(rows).items() if len(g) >= 3]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.4, 0.9 + 0.62 * len(groups)), facecolor=SURFACE)
    _style(ax)
    for y, (name, g) in enumerate(groups[::-1]):
        v = np.array([r["chrF++"] for r in g])
        ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=14, color=BLUE, alpha=0.45,
                   edgecolor=SURFACE, linewidth=0.5, zorder=2)
        est, lo, hi = bootstrap_ci(v, B=2000)
        ax.plot([lo, hi], [y, y], color=INK, linewidth=2, zorder=3)
        ax.scatter([est], [y], s=44, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=4)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{n} (n={len(g)})" for n, g in groups[::-1]], color=INK, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Sentence-level chrF++ against the dataset reference", color=INK2)
    ax.set_title("S1 overlap with references (dots: segments; black: mean with 95% CI)", loc="left",
                 color=INK, fontsize=10)
    fig.tight_layout()
    (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures" / "chrf_by_group.png", dpi=160)
    plt.close(fig)

    res = {"n_segments": len(rows), "n_flagged": len(flagged)}
    (out / "diagnostics_report.md").write_text(_report(res, summary, flagged, B, seed, _md, _num), encoding="utf-8")
    return res


def _report(res, summary, flagged, B, seed, _md, _num):
    share_cols = ["group", "n"] + [f"share_{f}" for f in FLAGS] + ["share_any_flag"]
    score_cols = ["group", "n", "chrF++", "chrF++_ci_low", "chrF++_ci_high", "BLEU", "BLEU_ci_low", "BLEU_ci_high"]
    L = ["# Automatic diagnostics of S1 output\n",
         "> These are rule-based checks and overlap scores. They are **not human judgments and not MQM**. "
         "Flags are candidates for a human to look at: names and numbers can be copied legitimately, and the "
         "dataset references are unverified and spelled inconsistently.\n",
         f"Segments: {res['n_segments']}; with at least one flag: {res['n_flagged']}. Bootstrap: {B} resamples of "
         f"segments, seed {seed}, 95% percentile intervals; scores are corpus-level within each group.\n",
         "## Overlap with the dataset references\n", _md(summary, score_cols),
         "![chrF++](figures/chrf_by_group.png)\n",
         "## Share of segments flagged\n", _md(summary, share_cols),
         "Flag definitions: empty output; unchanged copy of the English source; content words copied from the "
         "source that are absent from the reference; a repeated 3-gram (3+ times) or one word repeated 3+ times; "
         "numbers in the source and reference that the output lacks; output/reference length ratio outside 0.5-2.0.\n",
         "## Segments to look at first\n",
         f"{len(flagged)} flagged segment(s) are listed with their texts in `tables/flagged_segments.csv`, most "
         "flags and lowest chrF++ first. Showing the first few:\n"]
    L.append(_md([{"src_id": r["src_id"], "chrF++": r["chrF++"], "n_flags": r["n_flags"],
                   "flags": ", ".join(f for f in FLAGS if r[f])} for r in flagged[:10]]))
    L += ["## How to use this\n",
          "- Report it as an automatic diagnostic next to, not instead of, the human MQM evaluation.",
          "- If a flag type is common (for example repetition), tell the annotators nothing about it in advance, "
          "to keep the annotation independent; compare afterwards whether annotators marked those segments.",
          "- Group differences in chrF++ are small-sample and reference-dependent; do not over-read them."]
    return "\n".join(L) + "\n"
