"""Bootstrap confidence intervals and significance tests. Everything is seeded.

The unit of resampling is the SOURCE SENTENCE (an item), never the individual error, because
errors within one sentence are not independent.
"""
import warnings

import numpy as np
from scipy import stats as sps


def bootstrap_ci(x, stat=np.mean, B=10000, seed=13, level=0.95):
    """Percentile bootstrap CI of stat(x). Returns (estimate, low, high)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    boots = x[idx].mean(axis=1) if stat is np.mean else np.array([stat(x[i]) for i in idx])
    lo, hi = np.nanpercentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(stat(x)), float(lo), float(hi)


def bootstrap_ratio(num, den, scale=1000.0, B=10000, seed=13, level=0.95):
    """CI of sum(num)/sum(den)*scale, resampling items (used for the normalised penalty NPT)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(B, len(num)))
    boots = num[idx].sum(axis=1) / den[idx].sum(axis=1) * scale
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(num.sum() / den.sum() * scale), float(lo), float(hi)


def bootstrap_pair_stat(x, y, fn, B=2000, seed=13, level=0.95):
    """CI of fn(x, y) (e.g. a correlation), resampling the paired items."""
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        vals.append(fn(x[i], y[i]))
    vals = np.array(vals, float)
    if np.all(np.isnan(vals)):
        return fn(x, y), float("nan"), float("nan")
    lo, hi = np.nanpercentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(fn(x, y)), float(lo), float(hi)


def s1_vs_href(items, B=10000, seed=13):
    """Paired comparison on sources that have both an S1 item and a human-reference item.

    By design the two come from DIFFERENT annotators (a reference control uses a source the
    annotator has no S1 output for), so annotator strictness is confounded with the system
    difference. Report it as such. diff > 0 means S1 has MORE penalty (is worse).
    """
    s1 = {it["src_id"]: it["wmt"] for it in items if it["kind"] == "S1"}
    hr = {it["src_id"]: it["wmt"] for it in items if it["kind"] == "HREF"}
    ids = sorted(set(s1) & set(hr))
    out = {"n": len(ids)}
    if len(ids) < 2:
        return out
    d = np.array([s1[i] - hr[i] for i in ids])
    out["mean_diff"], out["lo"], out["hi"] = bootstrap_ci(d, B=B, seed=seed)
    out["s1_mean"], out["href_mean"] = float(np.mean([s1[i] for i in ids])), float(np.mean([hr[i] for i in ids]))
    if np.any(d != 0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["p_wilcoxon"] = float(sps.wilcoxon(d).pvalue)
    return out


def templated_vs_not(items, B=10000, seed=13):
    """Difference in mean S1 score, templated minus non-templated (unpaired bootstrap)."""
    t = np.array([it["wmt"] for it in items if it["kind"] == "S1" and it["templated"]])
    n = np.array([it["wmt"] for it in items if it["kind"] == "S1" and not it["templated"]])
    out = {"n_templated": len(t), "n_non_templated": len(n)}
    if len(t) < 2 or len(n) < 2:
        return out
    rng = np.random.default_rng(seed)
    bt = t[rng.integers(0, len(t), (B, len(t)))].mean(axis=1)
    bn = n[rng.integers(0, len(n), (B, len(n)))].mean(axis=1)
    lo, hi = np.percentile(bt - bn, [2.5, 97.5])
    out.update(mean_templated=float(t.mean()), mean_non_templated=float(n.mean()),
               diff=float(t.mean() - n.mean()), lo=float(lo), hi=float(hi),
               p_mannwhitney=float(sps.mannwhitneyu(t, n).pvalue))
    return out
