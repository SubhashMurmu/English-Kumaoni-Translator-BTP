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
        for n, row in enumerate(read_csv(p), start=2):          # row 1 is the header
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
