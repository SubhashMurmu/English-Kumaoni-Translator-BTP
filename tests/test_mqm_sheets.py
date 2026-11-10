"""Tests for sampling, allocation and blinding. All text is synthetic placeholder text
(English sources, 'REF-n' / 'S1-n' stand-ins); no Kumaoni is invented."""
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mqm_eval.schema import SHEET_COLUMNS, load_typology, validate_row  # noqa: E402
from mqm_eval.sampling import src_id, stratified_sample  # noqa: E402
from mqm_eval.sheets import build_rows, plan_allocation, write_outputs  # noqa: E402


def _rows(n=400):
    rows = []
    for i in range(n):
        # a quarter of the pool shares one opening ("please bring the") -> counts as templated
        src = f"Please bring the item number {i} today" if i % 4 == 0 else f"Unique sentence {i} about topic {i * 7}"
        rows.append({"src": src, "tgt": f"REF-{i}"})
    return rows


def _sample(n=100, seed=13):
    return stratified_sample(_rows(), n, seed, templated_share=0.3)


def test_sampling_deterministic_distinct_and_share():
    a, b = _sample(), _sample()
    assert [x["src_id"] for x in a] == [x["src_id"] for x in b]
    assert len({x["src_id"] for x in a}) == 100
    assert 28 <= sum(x["templated"] for x in a) <= 32
    assert [x["src_id"] for x in _sample(seed=1)] != [x["src_id"] for x in a]


def test_allocation_counts_and_no_double_exposure():
    s = _sample()
    ids = [x["src_id"] for x in s]
    plan = plan_allocation(ids)
    assert {a: len(v) for a, v in plan.items()} == {"A1": 90, "A2": 90}
    for a, items in plan.items():
        seen = [it["src_id"] for it in items]
        assert len(seen) == len(set(seen))                       # a source appears once per annotator
        outs = [it["out_src_id"] for it in items if it["kind"] == "CHECK"]
        assert not set(outs) & set(seen)                         # wrong-sentence text is never a seen sentence
        assert all(it["out_src_id"] != it["src_id"] for it in items if it["kind"] == "CHECK")
    s1 = lambda p: {it["src_id"] for it in p if it["kind"] == "S1"}   # noqa: E731
    assert len(s1(plan["A1"]) & s1(plan["A2"])) == 40            # shared agreement set
    assert len(s1(plan["A1"]) | s1(plan["A2"])) == 100


def test_blinded_sheets_and_key():
    s = _sample()
    outputs = {x["src_id"]: f"S1-{x['src_id']}" for x in s}
    plan = plan_allocation([x["src_id"] for x in s])
    key, sheets = build_rows(s, outputs, plan)
    assert len({k["item_id"] for k in key}) == len(key) == 180   # item ids unique across annotators
    with tempfile.TemporaryDirectory() as d:
        out = write_outputs(d, key, sheets)
        with open(out / "sheet_A1.csv", encoding="utf-8-sig") as f:
            r = list(csv.DictReader(f))
        assert list(r[0]) == SHEET_COLUMNS and len(r) == 90
        assert "kind" not in r[0] and "system" not in r[0]       # no system information on the sheet
        k = {x["item_id"]: x for x in key}
        href = [x for x in r if k[x["item_id"]]["kind"] == "HREF"]
        assert len(href) == 12 and all(x["translation"].startswith("REF-") for x in href)
        chk = [x for x in r if k[x["item_id"]]["kind"] == "CHECK"]
        assert len(chk) == 8 and all(x["translation"].startswith("S1-") for x in chk)
    orders = [[k["src_id"] for k in key if k["annotator_id"] == a][:10] for a in ("A1", "A2")]
    assert orders[0] != orders[1]                                # independent shuffles


def test_validate_row():
    t = load_typology()
    ok = {"dimension": "accuracy", "subtype": "omission", "severity": "major", "span": "[MISSING]"}
    assert validate_row(ok, t) == []
    assert validate_row({"no_error": "Y"}, t) == []
    assert validate_row({"no_error": "Y", "severity": "minor"}, t)
    assert validate_row({}, t)                                    # empty row
    assert validate_row({**ok, "subtype": "nope"}, t)             # unknown subtype
    assert validate_row({**ok, "severity": "severe"}, t)          # unknown severity
    var = {"dimension": "linguistic_conventions", "subtype": "spelling_variant", "span": "x"}
    assert validate_row({**var, "severity": "neutral"}, t) == []
    assert validate_row({**var, "severity": "minor"}, t)          # variants can never be penalised
    assert src_id("Hello ") == src_id("hello")
