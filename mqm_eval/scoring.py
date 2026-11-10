"""MQM scoring. Two presets are always computed side by side (see config/severity_weights.yaml).

WMT-style (after Freitag et al. 2021), per annotator and segment:
    score = sum of the (at most 5) largest error penalties
    penalty(error) = weight[severity], with overrides (punctuation minor = 0.1,
                     non-translation major = 25)
    item score = mean over the annotators who saw the item          (0 = perfect)

Official-style (structure of themqm.org/mqm-pillars/the-mqm-scoring-models/):
    ETPT = sum over errors of  SPM[severity] * ETW[dimension]       (no cap)
    APT  = sum of ETPT for an item (mean over annotators)
    NPT  = (sum APT over items) / (sum source words over items) * RWC
Only NPT is implemented for the official style. The page's raw/calibrated 0-100 formulas are
published as images, so they are not reproduced here; add them only after checking the source.
"""
import numpy as np
import yaml

from .schema import CONFIG_DIR

SEV_RANK = {"neutral": 0, "minor": 1, "major": 2, "critical": 3}


def load_presets(path=None):
    with open(path or CONFIG_DIR / "severity_weights.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["presets"]


def error_penalty(err, preset):
    sev = err["severity"]
    for o in preset.get("overrides", []):
        if (o["dimension"], o["subtype"], o["severity"]) == (err["dimension"], err["subtype"], sev):
            return float(o["weight"])
    etw = float((preset.get("error_type_weights") or {}).get(err["dimension"], 1))
    return float(preset["weights"][sev]) * etw


def segment_penalty(errors, preset):
    pens = sorted((error_penalty(e, preset) for e in errors), reverse=True)
    cap = preset.get("cap_errors_per_segment")
    return float(sum(pens[:cap] if cap else pens))


def build_items(ann, key, sample, presets):
    """Group annotations into items keyed by (src_id, kind).

    Each item carries per-annotator results (`by_ann`) and the mean over annotators.
    """
    items = {}
    for (a, iid), rec in ann.items():
        k = key[iid]
        sid, kind = k["src_id"], k["kind"]
        s = sample[sid]
        it = items.setdefault((sid, kind), {
            "src_id": sid, "kind": kind, "src": s["src"], "ref": s["tgt"],
            "n_words": s["n_words"], "templated": bool(s["templated"]), "by_ann": {}})
        errs = rec["errors"]
        it["by_ann"][a] = {
            "item_id": iid, "text": rec["text"], "errors": errs,
            "wmt": segment_penalty(errs, presets["wmt_style"]),
            "apt": segment_penalty(errs, presets["official_style"]),
            "max_sev": max((SEV_RANK[e["severity"]] for e in errs), default=0)}
    for it in items.values():
        vals = it["by_ann"].values()
        it["wmt"] = float(np.mean([v["wmt"] for v in vals]))
        it["apt"] = float(np.mean([v["apt"] for v in vals]))
        it["n_annotators"] = len(it["by_ann"])
    return list(items.values())


def of_kind(items, kind):
    return [it for it in items if it["kind"] == kind]


def npt(items, rwc=1000):
    """Normed penalty total per `rwc` source words."""
    return sum(it["apt"] for it in items) / sum(it["n_words"] for it in items) * rwc


def check_detection(items, min_severity=2):
    """Per annotator: share of attention-check items where a major/critical error was marked."""
    out = {}
    for it in of_kind(items, "CHECK"):
        for a, v in it["by_ann"].items():
            hit, n = out.get(a, (0, 0))
            out[a] = (hit + (v["max_sev"] >= min_severity), n + 1)
    return out
