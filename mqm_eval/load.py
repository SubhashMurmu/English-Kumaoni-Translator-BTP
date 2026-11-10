"""Load filled annotation sheets and join them to the private key.

A filled sheet has one row per error; an item with no errors has a single row with
no_error = Y. The key (key_PRIVATE.csv) says which source and kind (S1 / HREF / CHECK)
each item_id is. Problems are collected rather than raised, so you see all of them at once.
"""
import csv
import json
from pathlib import Path

from .schema import load_typology, validate_row


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_sheet(path):
    """Rows of an annotation sheet, from a .csv or an .xlsx (the 'Annotate' tab).

    xlsx rows carry '__line__', the real spreadsheet row, so problem messages point at the
    right line even when blank rows are skipped.
    """
    p = Path(path)
    if p.suffix.lower() != ".xlsx":
        return read_csv(p)
    from openpyxl import load_workbook
    wb = load_workbook(p, read_only=True, data_only=True)
    ws = wb["Annotate"] if "Annotate" in wb.sheetnames else wb.worksheets[0]
    it = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(it)]
    rows = []
    for n, r in enumerate(it, start=2):
        if r is None or all(v is None or str(v).strip() == "" for v in r):
            continue
        row = {h: ("" if v is None else str(v)) for h, v in zip(header, r) if h}
        row["__line__"] = n
        rows.append(row)
    wb.close()
    return rows


def collect_sheet_paths(folder):
    """CSV and XLSX sheets in a folder (Excel lock files ignored). A CSV and an XLSX with the
    same name would double-count one annotator's work, so that is an error."""
    folder = Path(folder)
    paths = sorted(p for p in list(folder.glob("*.csv")) + list(folder.glob("*.xlsx"))
                   if not p.name.startswith("~$"))
    stems = [p.stem for p in paths]
    dup = sorted({s for s in stems if stems.count(s) > 1})
    if dup:
        raise ValueError(f"both .csv and .xlsx exist for {dup}; keep only one of each")
    return paths


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_key(path):
    return {r["item_id"]: r for r in read_csv(path)}


def load_sample(path):
    return {r["src_id"]: r for r in read_jsonl(path)}


def _clean(row):
    g = lambda k: (row.get(k) or "").strip()      # noqa: E731
    return {"dimension": g("dimension").lower(), "subtype": g("subtype").lower(),
            "severity": g("severity").lower(), "span": g("span"),
            "explanation": g("explanation"), "correction": g("correction")}


def load_annotations(sheet_paths, key, typology=None):
    """Return ({(annotator, item_id): {"errors": [...], "text": str}}, problems).

    The annotator is taken from the key, not from the sheet, so a mistyped ID cannot
    reassign items. Items missing from a provided sheet are reported as problems.
    """
    typology = typology or load_typology()
    ann, problems, texts = {}, [], {}
    seen_annotators = set()
    for p in sheet_paths:
        for n, row in enumerate(read_sheet(p), start=2):        # row 1 is the header
            n = row.get("__line__", n)
            iid = (row.get("item_id") or "").strip()
            where = f"{Path(p).name} line {n} ({iid or 'no item_id'})"
            if iid not in key:
                problems.append(f"{where}: item_id not in key")
                continue
            a = key[iid]["annotator_id"]
            seen_annotators.add(a)
            sheet_a = (row.get("annotator_id") or "").strip()
            if sheet_a and sheet_a != a:
                problems.append(f"{where}: annotator_id {sheet_a!r} differs from the key ({a!r})")
            rec = ann.setdefault((a, iid), {"errors": [], "no_error": False, "text": ""})
            rec["text"] = row.get("translation") or rec["text"]
            msgs = validate_row(row, typology)
            if msgs:                      # invalid rows are reported and ignored, never scored
                problems.extend(f"{where}: {m}" for m in msgs)
                continue
            if (row.get("no_error") or "").strip().lower() == "y":
                rec["no_error"] = True
            elif any((row.get(c) or "").strip() for c in ("dimension", "subtype", "severity")):
                rec["errors"].append(_clean(row))
    for (a, iid), rec in list(ann.items()):
        if rec["no_error"] and rec["errors"]:
            problems.append(f"item {iid}: marked no_error=Y but also has error rows")
    for iid, k in key.items():
        if k["annotator_id"] in seen_annotators and (k["annotator_id"], iid) not in ann:
            problems.append(f"item {iid} ({k['annotator_id']}): not annotated")
    return ann, problems
