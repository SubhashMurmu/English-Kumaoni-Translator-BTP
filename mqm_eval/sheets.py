"""Allocation of items to two annotators and generation of blinded annotation sheets.

Per annotator (defaults give 90 items):
  * n_shared  S1 outputs seen by BOTH annotators            -> inter-annotator agreement
  * n_unique  S1 outputs seen only by this annotator        -> more distinct sentences
  * n_href    human references shown as if they were system output (hidden control)
  * n_check   attention checks: a source paired with the S1 output of a DIFFERENT sentence
              (a known-bad item that needs no invented Kumaoni)

A hidden-control or check item always uses a source the annotator has no S1 output for, so
nobody sees two translations of the same sentence. The blinded sheet has no system column;
the private key file maps item_id back to (source id, kind).
"""
import csv
import random
from pathlib import Path

from .schema import SHEET_COLUMNS

KINDS = ("S1", "HREF", "CHECK")


def plan_allocation(src_ids, annotators=("A1", "A2"), n_shared=40, n_unique=30,
                    n_href=12, n_check=8, seed=13):
    """Return {annotator: [{"src_id", "kind", "out_src_id"}]} (unshuffled)."""
    if len(annotators) != 2:
        raise ValueError("this design is for exactly two annotators")
    if len(set(src_ids)) != len(src_ids):
        raise ValueError("src_ids must be distinct")
    need = n_shared + 2 * n_unique
    if len(src_ids) != need:
        raise ValueError(f"need exactly {need} sources (n_shared + 2*n_unique), got {len(src_ids)}")
    if n_href + n_check >= n_unique:
        raise ValueError("n_href + n_check must be smaller than n_unique")
    ids = sorted(src_ids)
    random.Random(seed).shuffle(ids)
    shared = ids[:n_shared]
    uniq = {annotators[0]: ids[n_shared:n_shared + n_unique],
            annotators[1]: ids[n_shared + n_unique:]}
    plan = {}
    for a, other in ((annotators[0], annotators[1]), (annotators[1], annotators[0])):
        theirs = uniq[other]
        href = theirs[:n_href]
        chk = theirs[n_href:n_href + n_check]
        spare = theirs[n_href + n_check:]        # S1 outputs used only as wrong-sentence text
        items = [{"src_id": s, "kind": "S1", "out_src_id": s} for s in shared + uniq[a]]
        items += [{"src_id": s, "kind": "HREF", "out_src_id": s} for s in href]
        items += [{"src_id": s, "kind": "CHECK", "out_src_id": spare[i % len(spare)]}
                  for i, s in enumerate(chk)]
        plan[a] = items
    return plan


def build_rows(sample, s1_outputs, plan, seed=13):
    """Turn a plan into (private key rows, blinded sheet rows) per annotator.

    sample      list of dicts from sampling.stratified_sample (src_id, src, tgt)
    s1_outputs  {src_id: S1 translation}
    """
    by_id = {it["src_id"]: it for it in sample}
    rng = random.Random(seed)
    used, key, sheets = set(), [], {}
    for a, items in plan.items():
        items = list(items)
        random.Random(f"{seed}|{a}").shuffle(items)
        rows = []
        for pos, it in enumerate(items, 1):
            src = by_id[it["src_id"]]
            text = {"S1": lambda: s1_outputs[it["src_id"]],
                    "HREF": lambda: src["tgt"],
                    "CHECK": lambda: s1_outputs[it["out_src_id"]]}[it["kind"]]()
            while True:
                item_id = "i" + "".join(rng.choice("0123456789abcdef") for _ in range(6))
                if item_id not in used:
                    used.add(item_id)
                    break
            key.append({"annotator_id": a, "position": pos, "item_id": item_id,
                        "src_id": it["src_id"], "kind": it["kind"], "out_src_id": it["out_src_id"]})
            rows.append({"item_id": item_id, "source": src["src"], "translation": text,
                         "annotator_id": a})
        sheets[a] = rows
    return key, sheets


def write_outputs(outdir, key, sheets):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "key_PRIVATE.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    for a, rows in sheets.items():
        with open(out / f"sheet_{a}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in SHEET_COLUMNS})
    return out
