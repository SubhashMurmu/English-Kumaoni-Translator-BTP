"""SYNTHETIC annotations, for testing the pipeline before real annotators exist.

Nothing produced here is data about Kumaoni or about any model. The output folder contains
SYNTHETIC_DO_NOT_REPORT.txt and `analyze` stamps a warning banner on every report built from it.
Errors are drawn at random with a per-sentence latent difficulty shared by both annotators, so
the agreement statistics have something to find.
"""
import csv
import hashlib
import random
from pathlib import Path

import numpy as np

from .load import read_csv
from .schema import SHEET_COLUMNS, load_typology, valid_pairs

MARKER = "SYNTHETIC_DO_NOT_REPORT.txt"


def simulate(run_dir, out_dir=None, seed=7, strictness=None):
    run = Path(run_dir)
    out = Path(out_dir or run / "synthetic_filled")
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text("These annotations are randomly generated test data.\n", encoding="utf-8")
    key = {r["item_id"]: r for r in read_csv(run / "sheets" / "key_PRIVATE.csv")}
    pairs = [p for p, only in valid_pairs(load_typology()).items() if only is None]
    strictness = strictness or {}
    paths = []
    for sheet in sorted((run / "sheets").glob("sheet_*.csv")):
        rows = read_csv(sheet)
        out_rows = []
        for r in rows:
            k = key[r["item_id"]]
            a, kind = k["annotator_id"], k["kind"]
            lat = random.Random(f"{seed}|{k['src_id']}").random() ** 2      # shared sentence difficulty
            rng = np.random.default_rng(int(hashlib.md5(f"{seed}|{r['item_id']}".encode()).hexdigest()[:8], 16))
            lam = {"S1": 2.0 * lat + 0.2, "HREF": 0.5 * lat + 0.05, "CHECK": 0.0}[kind]
            n_err = 0 if kind == "CHECK" else int(rng.poisson(lam * strictness.get(a, 1.0)))
            base = {c: "" for c in SHEET_COLUMNS}
            base.update(item_id=r["item_id"], source=r["source"], translation=r["translation"],
                        annotator_id=a)
            if kind == "CHECK":
                missed = rng.random() < 0.05
                errs = [] if missed else [("accuracy", "mistranslation", "critical" if rng.random() < .5 else "major")]
            else:
                errs = []
                for _ in range(n_err):
                    dim, sub = pairs[rng.integers(len(pairs))]
                    sev = rng.choice(["minor", "major", "critical"], p=[0.6, 0.35, 0.05])
                    errs.append((dim, sub, str(sev)))
            if not errs:
                out_rows.append({**base, "no_error": "Y"})
                continue
            words = r["translation"].split() or ["x"]
            for dim, sub, sev in errs:
                span = "[MISSING]" if sub == "omission" else words[int(rng.integers(len(words)))]
                out_rows.append({**base, "dimension": dim, "subtype": sub, "severity": sev, "span": span,
                                 "explanation": "synthetic", "correction": ""})
        p = out / sheet.name
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
            w.writeheader()
            w.writerows(out_rows)
        paths.append(p)
    return out, paths
