"""Run the whole analysis and write tables, figures and a Markdown report.

  python -m mqm_eval.cli analyze --run mqm_run --filled path/to/filled_sheets --out mqm_run/results
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import agreement, inference, metrics_corr, scoring  # noqa: E402
from .load import load_annotations, load_key, load_sample  # noqa: E402
from .simulate import MARKER  # noqa: E402

# Colours: slots 1 (blue) and 2 (orange) of the reference categorical palette; the severity
# ramp is three steps of one hue (ordinal data -> sequential, light to dark).
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEV_COL = {"minor": "#a9c8f2", "major": "#2a78d6", "critical": "#14396b"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2dd", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _num(x):
    if isinstance(x, (float, np.floating)):
        return "n/a" if np.isnan(x) else f"{x:.2f}"
    return str(x)


def _write_csv(path, rows):
    if not rows:
        return
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _md(rows, cols=None):
    if not rows:
        return "_none_\n"
    cols = cols or list(dict.fromkeys(c for r in rows for c in r))
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(_num(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _row(measure, group, vals, B, seed, primary=False):
    est, lo, hi = inference.bootstrap_ci(vals, B=B, seed=seed)
    return {"measure": measure, "group": group, "estimate": est, "ci_low": lo, "ci_high": hi,
            "n_items": len(vals), "primary": "yes" if primary else ""}


def build_tables(items, primary, B, seed):
    s1 = scoring.of_kind(items, "S1")
    groups = {"S1 all": s1,
              "S1 templated": [i for i in s1 if i["templated"]],
              "S1 non-templated": [i for i in s1 if not i["templated"]]}
    summary = []
    for g, its in groups.items():
        if len(its) < 2:
            continue
        summary.append(_row("Mean segment penalty, WMT-style (0 = perfect)", g, [i["wmt"] for i in its],
                            B, seed, primary == "wmt_style"))
        est, lo, hi = inference.bootstrap_ratio([i["apt"] for i in its], [i["n_words"] for i in its],
                                                B=B, seed=seed)
        summary.append({"measure": "Normed penalty total per 1000 source words, official-style",
                        "group": g, "estimate": est, "ci_low": lo, "ci_high": hi, "n_items": len(its),
                        "primary": "yes" if primary == "official_style" else ""})
        summary.append(_row("Share of segments with no penalised error", g,
                            [float(all(v["max_sev"] == 0 for v in i["by_ann"].values())) for i in its], B, seed))
        summary.append(_row("Share of segments with a major/critical error (any annotator)", g,
                            [float(any(v["max_sev"] >= 2 for v in i["by_ann"].values())) for i in its], B, seed))
    hr = scoring.of_kind(items, "HREF")
    if len(hr) >= 2:
        summary.append(_row("Mean segment penalty, WMT-style (0 = perfect)", "Human reference (control)",
                            [i["wmt"] for i in hr], B, seed))

    # error distribution over S1 annotator-segments
    n_seg = sum(len(i["by_ann"]) for i in s1)
    dist, sub = Counter(), Counter()
    for i in s1:
        for v in i["by_ann"].values():
            for e in v["errors"]:
                dist[(e["dimension"], e["severity"])] += 1
                sub[(e["dimension"], e["subtype"], e["severity"])] += 1
    dims = sorted({d for d, _ in dist})
    err_dist = [{"dimension": d, **{s: dist[(d, s)] for s in ("neutral", "minor", "major", "critical")},
                 "total": sum(dist[(d, s)] for s in ("neutral", "minor", "major", "critical")),
                 "per_100_segments": 100 * sum(dist[(d, s)] for s in ("minor", "major", "critical")) / max(1, n_seg)}
                for d in dims]
    err_dist.sort(key=lambda r: -r["total"])
    subtypes = [{"dimension": d, "subtype": s, "severity": v, "count": c}
                for (d, s, v), c in sorted(sub.items(), key=lambda kv: -kv[1])]

    chk = scoring.check_detection(items)
    checks = [{"annotator": a, "checks_flagged_major_or_critical": h, "checks_total": n,
               "rate": h / n if n else float("nan")} for a, (h, n) in sorted(chk.items())]
    by_ann_href = {}
    for i in hr:
        for a, v in i["by_ann"].items():
            by_ann_href.setdefault(a, []).append(v["wmt"])
    controls = [{"annotator": a, "mean_penalty_on_human_references": float(np.mean(v)), "n": len(v)}
                for a, v in sorted(by_ann_href.items())]
    return summary, err_dist, subtypes, checks, controls, n_seg


def figures(items, out):
    out.mkdir(parents=True, exist_ok=True)
    s1, hr = scoring.of_kind(items, "S1"), scoring.of_kind(items, "HREF")
    rng = np.random.default_rng(0)

    # 1. segment penalties by group
    groups = [("S1, all", s1, BLUE), ("S1, templated", [i for i in s1 if i["templated"]], BLUE),
              ("S1, non-templated", [i for i in s1 if not i["templated"]], BLUE),
              ("Human reference (control)", hr, ORANGE)]
    groups = [g for g in groups if len(g[1]) >= 2]
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.8 * len(groups)), facecolor=SURFACE)
    _style(ax)
    for y, (name, its, col) in enumerate(groups[::-1]):
        v = np.array([i["wmt"] for i in its])
        ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=14, color=col, alpha=0.45,
                   edgecolor=SURFACE, linewidth=0.5, zorder=2)
        est, lo, hi = inference.bootstrap_ci(v, B=2000)
        ax.plot([lo, hi], [y, y], color=INK, linewidth=2, zorder=3)
        ax.scatter([est], [y], s=48, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=4)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{n} (n={len(i)})" for n, i, _ in groups[::-1]], color=INK)
    ax.set_xlabel("Segment penalty, WMT-style (0 = no penalised errors)", color=INK2)
    ax.set_title("Segment penalties\n(dots: segments, jittered; black: mean with 95% CI)",
                 loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "segment_penalties.png", dpi=160)
    plt.close(fig)

    # 2. error distribution by dimension and severity
    cnt = Counter()
    n_seg = sum(len(i["by_ann"]) for i in s1)
    for i in s1:
        for v in i["by_ann"].values():
            for e in v["errors"]:
                if e["severity"] != "neutral":
                    cnt[(e["dimension"], e["severity"])] += 1
    dims = sorted({d for d, _ in cnt}, key=lambda d: sum(cnt[(d, s)] for s in SEV_COL))
    if dims:
        fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.55 * len(dims)), facecolor=SURFACE)
        _style(ax)
        left = np.zeros(len(dims))
        for s, col in SEV_COL.items():
            w = np.array([100 * cnt[(d, s)] / max(1, n_seg) for d in dims])
            ax.barh(dims, w, left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.62, label=s)
            left += w
        for y, tot in enumerate(left):
            ax.text(tot + 0.4, y, f"{tot:.1f}", va="center", color=INK, fontsize=9)
        ax.set_xlabel("Penalised errors per 100 annotated S1 segments", color=INK2)
        ax.set_title("What kinds of errors S1 makes, by severity", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "error_distribution.png", dpi=160)
        plt.close(fig)

    # 3. annotator agreement on shared items
    sh = agreement.shared_s1(items)
    if len(sh) >= 3:
        a1, a2 = sorted(sh[0]["by_ann"])[:2]
        x = np.array([i["by_ann"][a1]["wmt"] for i in sh])
        y = np.array([i["by_ann"][a2]["wmt"] for i in sh])
        lim = max(x.max(), y.max()) + 1
        fig, ax = plt.subplots(figsize=(4.8, 4.6), facecolor=SURFACE)
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.plot([0, lim], [0, lim], color=GRID, linewidth=1.2, zorder=1)
        ax.scatter(x + rng.uniform(-.15, .15, len(x)), y + rng.uniform(-.15, .15, len(y)), s=26, color=BLUE,
                   alpha=0.7, edgecolor=SURFACE, linewidth=0.8, zorder=2)
        ax.set_xlim(-0.5, lim)
        ax.set_ylim(-0.5, lim)
        ax.set_xlabel(f"{a1} segment penalty", color=INK2)
        ax.set_ylabel(f"{a2} segment penalty", color=INK2)
        ax.set_title(f"Agreement on {len(sh)} shared items (jittered; line = identical)", loc="left", color=INK, fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "annotator_agreement.png", dpi=160)
        plt.close(fig)


def run_analysis(run_dir, sheet_paths, out_dir, primary="wmt_style", B=10000, seed=13,
                 allow_problems=False, synthetic=False, use_comet=False, force_comet=False):
    run, out = Path(run_dir), Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    key = load_key(run / "sheets" / "key_PRIVATE.csv")
    sample = load_sample(run / "sample.jsonl")
    ann, problems = load_annotations(sheet_paths, key)
    (out / "problems.txt").write_text("\n".join(problems) + ("\n" if problems else ""), encoding="utf-8")
    if problems and not allow_problems:
        raise SystemExit(f"{len(problems)} problem(s) in the sheets (see {out / 'problems.txt'}); first few:\n  "
                         + "\n  ".join(problems[:10]) + "\nFix them, or rerun with --allow-problems to ignore the invalid rows.")
    presets = scoring.load_presets()
    items = scoring.build_items(ann, key, sample, presets)
    if not scoring.of_kind(items, "S1"):
        raise SystemExit("no S1 items were annotated")

    summary, err_dist, subtypes, checks, controls, n_seg = build_tables(items, primary, B, seed)
    agree = agreement.agreement_table(items, min(B, 2000), seed)
    paired = inference.s1_vs_href(items, B, seed)
    strata = inference.templated_vs_not(items, B, seed)
    mc_rows, mc_notes = metrics_corr.metric_correlations(items, min(B, 2000), seed, use_comet, force_comet)
    disagree = agreement.disagreements(items)

    tdir = out / "tables"
    for name, rows in (("summary", summary), ("error_distribution", err_dist), ("error_subtypes", subtypes),
                       ("attention_checks", checks), ("reference_controls", controls),
                       ("agreement", agree), ("metric_correlation", mc_rows), ("disagreements", disagree)):
        _write_csv(tdir / f"{name}.csv", rows)
    figures(items, out / "figures")

    results = {"primary_preset": primary, "seed": seed, "bootstrap_resamples": B, "synthetic": synthetic,
               "n_items": len(items), "n_s1_items": len(scoring.of_kind(items, "S1")),
               "n_problems": len(problems), "s1_vs_href": paired, "templated_vs_non_templated": strata,
               "metric_notes": mc_notes}
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "report.md").write_text(_report_md(results, summary, err_dist, checks, controls, agree, mc_rows,
                                              disagree, n_seg), encoding="utf-8")
    return results


def _report_md(res, summary, err_dist, checks, controls, agree, mc_rows, disagree, n_seg):
    L = []
    if res["synthetic"]:
        L += ["> **SYNTHETIC DATA: these results come from randomly generated annotations and say nothing "
              "about Kumaoni or about any model. Do not report them.**\n"]
    L += ["# English → Kumaoni MQM evaluation report\n",
          f"Primary score: **{res['primary_preset']}** (set before annotation). Bootstrap: "
          f"{res['bootstrap_resamples']} resamples of source sentences, seed {res['seed']}, 95% percentile intervals. "
          f"S1 items scored: {res['n_s1_items']} ({n_seg} annotator-segments). "
          f"Sheet problems ignored: {res['n_problems']}.\n",
          "## 1. Quality of S1\n",
          "Penalties: WMT-style = severity-weighted error count per segment (Minor 1, Major 5, Critical 25, "
          "punctuation 0.1, at most 5 errors counted); official-style = penalty per 1000 English source words "
          "(structure of the MQM scoring-models page; its multipliers are examples, not normative).\n",
          _md(summary), "## 2. Controls\n",
          "Human references were shown as if they were system output; attention checks pair a source with another "
          "sentence's output and should be flagged major/critical.\n",
          _md(controls), _md(checks)]
    p = res["s1_vs_href"]
    if p.get("n", 0) >= 2:
        L += ["### S1 vs human reference (paired by source)\n",
              f"n = {p['n']} sources. Mean penalty S1 {_num(p['s1_mean'])} vs reference {_num(p['href_mean'])}; "
              f"difference {_num(p['mean_diff'])} (95% CI {_num(p['lo'])} to {_num(p['hi'])})"
              + (f", Wilcoxon p = {_num(p['p_wilcoxon'])}" if "p_wilcoxon" in p else "")
              + ". Caution: the two scores in each pair come from different annotators, so annotator strictness is "
                "confounded with the difference.\n"]
    t = res["templated_vs_non_templated"]
    if "diff" in t:
        L += ["### Templated vs non-templated sources\n",
              f"Templated {_num(t['mean_templated'])} (n={t['n_templated']}) vs non-templated "
              f"{_num(t['mean_non_templated'])} (n={t['n_non_templated']}); difference {_num(t['diff'])} "
              f"(95% CI {_num(t['lo'])} to {_num(t['hi'])}), Mann-Whitney p = {_num(t['p_mannwhitney'])}.\n"]
    L += ["## 3. Error distribution (S1)\n", "![errors](figures/error_distribution.png)\n", _md(err_dist),
          "![penalties](figures/segment_penalties.png)\n",
          "## 4. Inter-annotator agreement (shared S1 items)\n", _md(agree),
          "![agreement](figures/annotator_agreement.png)\n",
          f"{len(disagree)} shared item(s) differ in worst severity (see tables/disagreements.csv) for the review session.\n",
          "## 5. Automatic metrics vs MQM\n", _md(mc_rows)]
    L += [f"- {n}" for n in res["metric_notes"]]
    L += ["\n## 6. Limitations\n",
          "- Two annotators and a sample of about 100 sentences: intervals are wide and agreement estimates are rough.",
          "- Sentence-level evaluation only; no document context.",
          "- Only one system (S1), so there is no system ranking.",
          "- References are dataset references, not independently verified; spelling variation is common.",
          "- Annotator region and variety of Kumaoni are recorded separately and may differ."]
    return "\n".join(L) + "\n"
