"""Annotation record layout and validation against the typology."""
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / "config"

# Columns of the sheet an annotator fills in (blank on delivery).
INPUT_COLUMNS = ["item_id", "source", "translation"]
ANNOTATION_COLUMNS = ["no_error", "dimension", "subtype", "severity", "span",
                      "explanation", "correction", "annotator_id"]
SHEET_COLUMNS = INPUT_COLUMNS + ANNOTATION_COLUMNS


def load_typology(path=None):
    with open(path or CONFIG_DIR / "typology.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def valid_pairs(typology):
    """{(dimension, subtype): only_severity or None}"""
    out = {}
    for dim, d in typology["dimensions"].items():
        for sub, meta in d["subtypes"].items():
            out[(dim, sub)] = (meta or {}).get("only_severity")
    return out


def validate_row(row, typology):
    """Return a list of problems for one annotation row (empty list = valid).

    A row is either a 'no error' row (no_error == Y and all error columns blank) or an
    error row with a valid dimension/subtype pair and a severity.
    """
    pairs, sevs = valid_pairs(typology), set(typology["severities"])
    g = lambda k: (row.get(k) or "").strip().lower()   # noqa: E731
    errors_cols = [g("dimension"), g("subtype"), g("severity")]
    if g("no_error") == "y":
        return ["no_error=Y but error columns are filled"] if any(errors_cols) else []
    problems = []
    if not any(errors_cols):
        return ["row is empty: mark errors, or put Y in no_error"]
    if (g("dimension"), g("subtype")) not in pairs:
        problems.append(f"unknown dimension/subtype: {g('dimension')!r}/{g('subtype')!r}")
    if g("severity") not in sevs:
        problems.append(f"unknown severity: {g('severity')!r}")
    elif (g("dimension"), g("subtype")) in pairs:
        only = pairs[(g("dimension"), g("subtype"))]
        if only and g("severity") != only:
            problems.append(f"{g('subtype')} is only allowed with severity {only}")
    if not (row.get("span") or "").strip():
        problems.append("span is empty")
    return problems
