"""Diagnostics and Excel sheets. All text is synthetic placeholder text; no Kumaoni is invented."""
import csv
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from mqm_eval.diagnostics import segment_diagnostics, run_diagnostics  # noqa: E402
from mqm_eval.load import collect_sheet_paths, read_csv, read_sheet  # noqa: E402
from mqm_eval.report import run_analysis  # noqa: E402
from mqm_eval.schema import SHEET_COLUMNS  # noqa: E402
from mqm_eval.simulate import simulate  # noqa: E402
from mqm_eval.xlsx import write_xlsx  # noqa: E402
from test_mqm_scoring import _make_run  # noqa: E402


def flags(src, ref, hyp):
    return {k for k, v in segment_diagnostics(src, ref, hyp)["flags"].items() if v}


def test_diagnostic_flags_hand_checked():
    src, ref = "Please bring 3 books today", "ref 3 kitab ref"
    assert flags(src, ref, "ref 3 kitab ref") == set()
    # empty output: the "3" is missing too; the reference has only 3 words, so no length flag
    assert flags(src, ref, "") == {"empty", "number_mismatch"}
    assert flags(src, ref, "Please bring 3 books today") >= {"copy_of_source", "copied_english"}
    assert flags(src, ref, "ref 5 kitab ref") == {"number_mismatch"}               # source and reference say 3
    assert "repetition" in flags(src, ref, "kitab kitab kitab 3")                   # one word 3 times in a row
    assert "repetition" in flags("s", "a b c d", "a b c a b c a b c")               # a 3-gram three times
    assert "repetition" not in flags("s", "a b c d", "a b c a b c")                 # only twice: not flagged
    d = segment_diagnostics(src, ref, "bring kitab 3")
    assert d["copied_tokens"] == ["bring"]                                          # in source, not in reference
    assert segment_diagnostics("Meet Nikhil today", "ref Nikhil ref", "Nikhil ref")["copied_tokens"] == []  # name is in the reference
    long_ref = "a b c d e f"
    assert "length_outlier" in flags("s", long_ref, "a")                            # ratio 1/6
    assert "length_outlier" not in flags("s", "a b c", "a")                         # reference under 4 words
    assert "length_outlier" in flags("s", "a b c d", "a b c d e f g h i")           # ratio above 2


def test_run_diagnostics_end_to_end():
    pytest.importorskip("sacrebleu")
    with tempfile.TemporaryDirectory() as d:
        run = _make_run(d)
        import json
        sample = [json.loads(line) for line in open(run / "sample.jsonl", encoding="utf-8")]
        with open(run / "s1_outputs.jsonl", "w", encoding="utf-8") as f:
            for r in sample:
                f.write(json.dumps({"src_id": r["src_id"], "text": "S1 " + r["src_id"]}) + "\n")
        res = run_diagnostics(run, run / "diag", B=50, seed=1)
        assert res["n_segments"] == 100
        o = run / "diag"
        assert (o / "diagnostics_report.md").exists() and (o / "figures" / "chrf_by_group.png").stat().st_size > 1000
        rows = list(csv.DictReader(open(o / "tables" / "diagnostics_summary.csv", encoding="utf-8")))
        assert rows[0]["group"] == "all segments" and int(rows[0]["n"]) == 100
        assert "not human judgments and not MQM" in (o / "diagnostics_report.md").read_text(encoding="utf-8")


def test_xlsx_structure_and_loader_agree_with_csv():
    from openpyxl import load_workbook
    with tempfile.TemporaryDirectory() as d:
        run = _make_run(d)
        csv_path = sorted((run / "sheets").glob("sheet_*.csv"))[0]
        rows = read_csv(csv_path)
        x = write_xlsx(rows, Path(d) / "sheet_A1.xlsx")
        wb = load_workbook(x)
        assert wb.sheetnames == ["Annotate", "Guide", "Lists"] and wb["Lists"].sheet_state == "hidden"
        ws = wb["Annotate"]
        assert [c.value for c in ws[1]] == SHEET_COLUMNS + ["check"] and ws.max_row == 91
        dvs = {str(v.sqref).split(":")[0]: v.formula1 for v in ws.data_validations.dataValidation}
        assert dvs["E2"] == "=dimension_list" and dvs["F2"] == "=INDIRECT($E2)"
        for name in ("accuracy", "linguistic_conventions", "special", "dimension_list"):
            assert name in wb.defined_names
        back = read_sheet(x)
        assert len(back) == len(rows)
        for a, b in zip(rows, back):
            assert all((a[c] or "") == b[c] for c in ("item_id", "source", "translation", "annotator_id"))


def test_xlsx_text_starting_with_equals_stays_text():
    from openpyxl import load_workbook
    with tempfile.TemporaryDirectory() as d:
        row = {c: "" for c in SHEET_COLUMNS}
        row.update(item_id="i1", source="=1+1 is two", translation="=SUM(A1)", annotator_id="A1")
        write_xlsx([row], Path(d) / "s.xlsx")
        ws = load_workbook(Path(d) / "s.xlsx")["Annotate"]
        assert ws["B2"].data_type == "s" and ws["C2"].data_type == "s"


def test_analysis_from_xlsx_matches_csv_and_path_rules():
    pytest.importorskip("sacrebleu")
    with tempfile.TemporaryDirectory() as d:
        run = _make_run(d)
        out_dir, csv_paths = simulate(run)
        xdir = Path(d) / "filled_xlsx"
        xdir.mkdir()
        for p in csv_paths:
            write_xlsx(read_csv(p), xdir / (p.stem + ".xlsx"))
        (xdir / "~$sheet_A1.xlsx").write_bytes(b"lock")                      # Excel lock file: ignored
        xlsx_paths = collect_sheet_paths(xdir)
        assert [p.suffix for p in xlsx_paths] == [".xlsx", ".xlsx"]
        r1 = run_analysis(run, csv_paths, run / "res_csv", "wmt_style", B=200, seed=3, synthetic=True)
        r2 = run_analysis(run, xlsx_paths, run / "res_xlsx", "wmt_style", B=200, seed=3, synthetic=True)
        assert r1["n_problems"] == r2["n_problems"] == 0
        assert (run / "res_csv" / "tables" / "summary.csv").read_text() == \
               (run / "res_xlsx" / "tables" / "summary.csv").read_text()
        both = Path(d) / "both"
        both.mkdir()
        (both / "sheet_A1.csv").write_text("x")
        (both / "sheet_A1.xlsx").write_bytes(b"x")
        with pytest.raises(ValueError):
            collect_sheet_paths(both)
