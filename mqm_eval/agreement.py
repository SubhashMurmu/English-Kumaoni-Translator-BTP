"""Inter-annotator agreement on the items BOTH annotators saw (the shared S1 items).

Statistics (all project choices; MQM itself does not prescribe one):
  * Spearman rho and Kendall tau-c on segment scores      (tau-c as in Kocmi et al. 2024)
  * Krippendorff's alpha, interval metric, on segment scores
  * quadratic-weighted kappa on the worst severity per item (none/minor/major/critical)
  * Cohen's kappa on "any penalised error present"
  * mean span overlap F1 between the two annotators' marked spans
Every statistic gets a bootstrap CI over items. With about 40 shared items the CIs are wide;
report them rather than the point estimate alone.
"""
import warnings

import numpy as np
from scipy import stats as sps

from .inference import bootstrap_ci, bootstrap_pair_stat


def qwk(a, b, k):
    """Quadratic-weighted Cohen's kappa for integer categories 0..k-1 (k=2 gives plain kappa)."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    O /= O.sum()
    E = np.outer(O.sum(axis=1), O.sum(axis=0))
    W = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    den = (W * E).sum()
    return float("nan") if den == 0 else float(1 - (W * O).sum() / den)


def krippendorff_interval(x, y):
    """Krippendorff's alpha, interval metric, two raters, no missing values."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    N = len(x)
    v = np.concatenate([x, y])
    n = len(v)
    Do = ((x - y) ** 2).sum() / N
    De = 2 * (n * (v ** 2).sum() - v.sum() ** 2) / (n * (n - 1))
    return float("nan") if De == 0 else float(1 - Do / De)


def _tau_c(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(sps.kendalltau(x, y, variant="c").statistic)


def _spearman(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(sps.spearmanr(x, y).statistic)


def span_tokens(errors, text):
    """Character positions (and a marker for omissions) covered by penalised errors."""
    toks = set()
    for e in errors:
        if e["severity"] == "neutral":
            continue
        s = e["span"].strip()
        if s.lower() == "[missing]":
            toks.add(("M",))
            continue
        i = text.find(s) if s else -1
        if i < 0:
            toks.add(("U", s.lower()))          # span not found in the text: compare by string
        else:
            toks.update(("c", j) for j in range(i, i + len(s)))
    return toks


def span_f1(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    p, r = inter / len(a), inter / len(b)
    return 2 * p * r / (p + r)


def shared_s1(items):
    return [it for it in items if it["kind"] == "S1" and it["n_annotators"] == 2]


def agreement_table(items, B=2000, seed=13):
    sh = shared_s1(items)
    if len(sh) < 3:
        return []
    ann = sorted(sh[0]["by_ann"])
    a1, a2 = ann[0], ann[1]
    s1 = np.array([it["by_ann"][a1]["wmt"] for it in sh])
    s2 = np.array([it["by_ann"][a2]["wmt"] for it in sh])
    m1 = np.array([it["by_ann"][a1]["max_sev"] for it in sh])
    m2 = np.array([it["by_ann"][a2]["max_sev"] for it in sh])
    f1 = np.array([span_f1(span_tokens(it["by_ann"][a1]["errors"], it["by_ann"][a1]["text"]),
                           span_tokens(it["by_ann"][a2]["errors"], it["by_ann"][a2]["text"]))
                   for it in sh])
    rows = []

    def add(name, est, lo, hi):
        rows.append({"statistic": name, "estimate": est, "ci_low": lo, "ci_high": hi, "n_items": len(sh)})

    add("Spearman rho (segment scores)", *bootstrap_pair_stat(s1, s2, _spearman, B, seed))
    add("Kendall tau-c (segment scores)", *bootstrap_pair_stat(s1, s2, _tau_c, B, seed))
    add("Krippendorff alpha, interval (segment scores)", *bootstrap_pair_stat(s1, s2, krippendorff_interval, B, seed))
    add("Weighted kappa (worst severity, 4 levels)", *bootstrap_pair_stat(m1, m2, lambda x, y: qwk(x, y, 4), B, seed))
    add("Cohen kappa (any penalised error)", *bootstrap_pair_stat((m1 > 0).astype(int), (m2 > 0).astype(int),
                                                                  lambda x, y: qwk(x, y, 2), B, seed))
    add("Mean span overlap F1", *bootstrap_ci(f1, B=B, seed=seed))
    return rows


def disagreements(items, min_gap=1):
    """Shared items where the worst severity differs by at least `min_gap` levels (review list)."""
    out = []
    for it in shared_s1(items):
        a1, a2 = sorted(it["by_ann"])[:2]
        gap = abs(it["by_ann"][a1]["max_sev"] - it["by_ann"][a2]["max_sev"])
        if gap >= min_gap:
            out.append({"src_id": it["src_id"], "gap": gap,
                        **{f"{a}_max_severity": it["by_ann"][a]["max_sev"] for a in (a1, a2)},
                        **{f"{a}_item_id": it["by_ann"][a]["item_id"] for a in (a1, a2)}})
    return sorted(out, key=lambda r: -r["gap"])
