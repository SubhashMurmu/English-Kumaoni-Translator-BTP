"""Do automatic metrics track human MQM judgments of this model's Kumaoni output?

Segment-level only: with a single system there is no system-level correlation to compute.
"Quality" is the NEGATIVE of the mean MQM penalty, so a positive correlation is the good case.

Metric support is checked, not assumed:
  * BLEU and chrF++ are character/word overlap and need no language model.
  * COMET and relatives sit on XLM-R. Its README language list covers Hindi but not Kumaoni,
    and the README warns that results for uncovered languages are unreliable. COMET is
    therefore skipped unless you pass force=True, and a forced run must be labelled
    exploratory in the report.
"""
import numpy as np

from .agreement import _spearman, _tau_c
from .inference import bootstrap_pair_stat

# Languages verified to be covered by the encoder. Kumaoni (kfy) is NOT listed in the COMET
# README at the time of writing; re-check before adding it.
COMET_VERIFIED_LANGS = set()


def comet_allowed(lang="kfy", force=False):
    if lang in COMET_VERIFIED_LANGS:
        return True, "language listed as supported"
    if force:
        return True, f"{lang} is not in the verified list; results are exploratory only"
    return False, f"{lang} is not in the verified XLM-R/COMET language list; COMET skipped"


def sentence_scores(hyps, refs):
    """Sentence-level chrF++ and smoothed BLEU against the dataset references."""
    try:
        import sacrebleu
    except ImportError as e:
        raise SystemExit("sacrebleu is required: pip install sacrebleu") from e
    chrf = [sacrebleu.sentence_chrf(h, [r], word_order=2).score for h, r in zip(hyps, refs)]
    bleu = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hyps, refs)]
    return {"chrF++": np.array(chrf), "BLEU": np.array(bleu)}


def metric_correlations(items, B=2000, seed=13, use_comet=False, force_comet=False):
    s1 = [it for it in items if it["kind"] == "S1"]
    if len(s1) < 5:
        return [], ["too few S1 items for a correlation"]
    texts = [next(iter(it["by_ann"].values()))["text"] for it in s1]
    refs = [it["ref"] for it in s1]
    quality = -np.array([it["wmt"] for it in s1])
    notes = []
    scores = sentence_scores(texts, refs)
    if use_comet:
        ok, why = comet_allowed("kfy", force_comet)
        notes.append(f"COMET: {why}")
        if ok:
            notes.append("COMET scoring is not implemented in this version.")
    else:
        notes.append(comet_allowed("kfy")[1])
    rows = []
    for name, m in scores.items():
        for stat_name, fn in (("Kendall tau-c", _tau_c), ("Spearman rho", _spearman)):
            est, lo, hi = bootstrap_pair_stat(m, quality, fn, B, seed)
            rows.append({"metric": name, "statistic": stat_name, "estimate": est,
                         "ci_low": lo, "ci_high": hi, "n_items": len(s1)})
    notes.append("References are unverified dataset references and spellings vary, so overlap metrics "
                 "are penalised for harmless variation; interpret low correlations with that in mind.")
    return rows, notes
