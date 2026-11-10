"""Scoring, agreement, statistics, loader and an end-to-end run on SYNTHETIC data only."""
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mqm_eval import agreement, scoring  # noqa: E402
from mqm_eval.inference import bootstrap_ci, bootstrap_ratio  # noqa: E402
from mqm_eval.load import load_annotations  # noqa: E402
from mqm_eval.report import run_analysis  # noqa: E402
from mqm_eval.sampling import stratified_sample  # noqa: E402
from mqm_eval.schema import SHEET_COLUMNS  # noqa: E402
from mqm_eval.sheets import build_rows, plan_allocation, write_outputs  # noqa: E402
from mqm_eval.simulate import MARKER, simulate  # noqa: E402

P = scoring.load_presets()
W, O = P["wmt_style"], P["official_style"]


def E(dim, sub, sev, span="x"):
    return {"dimension": dim, "subtype": sub, "severity": sev, "span": span}


def test_penalties_wmt_style():
    assert scoring.error_penalty(E("accuracy", "omission", "major"), W) == 5
    assert scoring.error_penalty(E("accuracy", "omission", "minor"), W) == 1
    assert scoring.error_penalty(E("linguistic_conventions", "punctuation", "minor"), W) == 0.1
    assert scoring.error_penalty(E("linguistic_conventions", "punctuation", "major"), W) == 5   # override is minor-only
    assert scoring.error_penalty(E("special", "non_translation", "major"), W) == 25
    assert scoring.error_penalty(E("accuracy", "mistranslation", "critical"), W) == 25
    assert scoring.error_penalty(E("linguistic_conventions", "spelling_variant", "neutral"), W) == 0


def test_segment_cap_and_sum():
    seven_minor = [E("linguistic_conventions", "grammar", "minor")] * 7
    assert scoring.segment_penalty(seven_minor, W) == 5                       # capped at 5 errors
    mixed = [E("accuracy", "omission", "major")] + [E("linguistic_conventions", "grammar", "minor")] * 6
    assert scoring.segment_penalty(mixed, W) == 5 + 4 * 1                     # the 5 largest
    assert scoring.segment_penalty(seven_minor, O) == 7                       # official style: no cap
    assert scoring.segment_penalty([], W) == 0
    three = [E("accuracy", "omission", "major"), E("linguistic_conventions", "grammar", "minor"),
             E("linguistic_conventions", "punctuation", "minor")]
    assert scoring.segment_penalty(three, W) == pytest.approx(6.1)


def test_npt():
    items = [{"apt": 2.0, "n_words": 10}, {"apt": 3.0, "n_words": 15}]
    assert scoring.npt(items) == pytest.approx(200.0)                         # (2+3)/(10+15)*1000
    est, lo, hi = bootstrap_ratio([2, 3], [10, 15], B=500)
    assert est == pytest.approx(200.0) and lo <= est <= hi


def test_agreement_statistics_hand_checked():
    assert agreement.qwk([0, 1, 2, 3], [0, 1, 2, 3], 4) == pytest.approx(1.0)
    assert agreement.qwk([0, 1], [1, 0], 2) == pytest.approx(-1.0)            # fully opposed (computed by hand)
    assert math.isnan(agreement.qwk([1, 1], [1, 1], 4))                       # no variation: undefined
    assert agreement.krippendorff_interval([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert agreement.krippendorff_interval([0, 2], [0, 0]) == pytest.approx(0.0)   # Do = 2, De = 2 (by hand)
    a = {("c", 0), ("c", 1)}
    b = {("c", 1), ("c", 2)}
    assert agreement.span_f1(a, b) == pytest.approx(0.5)
    assert agreement.span_f1(set(), set()) == 1.0 and agreement.span_f1(a, set()) == 0.0
    toks = agreement.span_tokens([E("accuracy", "mistranslation", "major", "def"),
                                  E("accuracy", "omission", "major", "[MISSING]"),
                                  E("linguistic_conventions", "spelling_variant", "neutral", "abc")], "abc def")
    assert toks == {("c", 4), ("c", 5), ("c", 6), ("M",)}                     # neutral ignored


def test_bootstrap_is_seeded_and_sane():
    assert bootstrap_ci([3, 3, 3, 3], B=200) == (3.0, 3.0, 3.0)
    x = np.random.default_rng(1).normal(5, 1, 60)
    est, lo, hi = bootstrap_ci(x, B=1000, seed=5)
    assert lo < est < hi and (est, lo, hi) == bootstrap_ci(x, B=1000, seed=5)


def _key_and_sheet(d, rows):
    key = [{"annotator_id": "A1", "position": 1, "item_id": "i1", "src_id": "s1", "kind": "S1", "out_src_id": "s1"},
           {"annotator_id": "A1", "position": 2, "item_id": "i2", "src_id": "s2", "kind": "S1", "out_src_id": "s2"}]
    p = Path(d) / "sheet.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in SHEET_COLUMNS})
    return {k["item_id"]: k for k in key}, [p]


def test_loader_reports_problems_and_ignores_invalid_rows():
    with tempfile.TemporaryDirectory() as d:
        key, paths = _key_and_sheet(d, [
            {"item_id": "i1", "dimension": "accuracy", "subtype": "nope", "severity": "major", "span": "x"},
            {"item_id": "zzz", "no_error": "Y"}])
        ann, problems = load_annotations(paths, key)
        text = "\n".join(problems)
        assert "unknown dimension/subtype" in text and "not in key" in text
        assert "i2" in text and "not annotated" in text                       # i2 has no rows at all
        assert ann[("A1", "i1")]["errors"] == []                              # the invalid row was not scored
        key2, paths2 = _key_and_sheet(d, [
            {"item_id": "i1", "no_error": "Y"},
            {"item_id": "i2", "dimension": "accuracy", "subtype": "omission", "severity": "major", "span": "[MISSING]"}])
        ann2, problems2 = load_annotations(paths2, key2)
        assert problems2 == [] and len(ann2[("A1", "i2")]["errors"]) == 1


def _make_run(d):
    rows = [{"src": (f"Please bring the item number {i} today" if i % 4 == 0 else f"Unique sentence {i} about topic {i * 7}"),
             "tgt": f"REF-{i}"} for i in range(400)]
    sample = stratified_sample(rows, 100, 13, 0.3)
    run = Path(d) / "run"
    run.mkdir()
    with open(run / "sample.jsonl", "w", encoding="utf-8") as f:
        for it in sample:
            f.write(json.dumps(it) + "\n")
    s1 = {it["src_id"]: f"S1 {it['src_id']} output" for it in sample}
    key, sheets = build_rows(sample, s1, plan_allocation([it["src_id"] for it in sample]))
    write_outputs(run / "sheets", key, sheets)
    return run


def test_end_to_end_on_synthetic_annotations():
    pytest.importorskip("sacrebleu")
    with tempfile.TemporaryDirectory() as d:
        run = _make_run(d)
        out_dir, paths = simulate(run)
        assert (out_dir / MARKER).exists()
        res = run_analysis(run, paths, run / "results", "wmt_style", B=300, seed=3, synthetic=True)
        r = run / "results"
        assert res["n_problems"] == 0 and res["n_s1_items"] == 100
        assert "SYNTHETIC DATA" in (r / "report.md").read_text(encoding="utf-8")
        for f in ("summary", "agreement", "error_distribution", "attention_checks", "metric_correlation"):
            assert (r / "tables" / f"{f}.csv").exists()
        for f in ("segment_penalties", "error_distribution", "annotator_agreement"):
            assert (r / "figures" / f"{f}.png").stat().st_size > 1000
        rows = list(csv.DictReader(open(r / "tables" / "summary.csv", encoding="utf-8")))
        prim = [x for x in rows if x["primary"] == "yes" and x["group"] == "S1 all"]
        assert len(prim) == 1 and float(prim[0]["ci_low"]) <= float(prim[0]["estimate"]) <= float(prim[0]["ci_high"])
        agree = {x["statistic"]: float(x["estimate"]) for x in
                 csv.DictReader(open(r / "tables" / "agreement.csv", encoding="utf-8"))}
        assert len(agree) == 6 and agree["Spearman rho (segment scores)"] > 0   # shared difficulty is found
        for x in csv.DictReader(open(r / "tables" / "attention_checks.csv", encoding="utf-8")):
            assert float(x["rate"]) >= 0.7
        # the same input and seed give the same numbers
        run_analysis(run, paths, run / "results2", "wmt_style", B=300, seed=3, synthetic=True)
        assert (r / "tables" / "summary.csv").read_text() == (run / "results2" / "tables" / "summary.csv").read_text()
