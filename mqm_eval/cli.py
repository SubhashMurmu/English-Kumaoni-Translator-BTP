"""Command line entry points.

  1. python -m mqm_eval.cli sample --test data/test.jsonl --train data/train.jsonl --out mqm_run
        writes mqm_run/sample.jsonl (100 sources by default)
  2. Run the existing model on the sources in sample.jsonl and save
        mqm_run/s1_outputs.jsonl with rows {"src_id": ..., "text": ...}
     (for example with kmt.infer.Translator; this package does not load any model)
  3. python -m mqm_eval.cli sheets --run mqm_run
        writes mqm_run/sheets/sheet_A1.csv, sheet_A2.csv (give these to annotators)
        and mqm_run/sheets/key_PRIVATE.csv (keep it away from annotators)
"""
import argparse
import json
from pathlib import Path

from .sampling import read_jsonl, strata_summary, stratified_sample
from .sheets import build_rows, plan_allocation, write_outputs


def cmd_sample(a):
    test = read_jsonl(a.test)
    pool = read_jsonl(a.train) if a.train else None
    n = a.n_shared + 2 * a.n_unique
    sample = stratified_sample(test, n, a.seed, a.templated_share, pool)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sample.jsonl", "w", encoding="utf-8") as f:
        for it in sample:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(sample)} sources -> {out / 'sample.jsonl'}  {strata_summary(sample)}")


def cmd_sheets(a):
    run = Path(a.run)
    sample = read_jsonl(run / "sample.jsonl")
    s1 = {r["src_id"]: r["text"] for r in read_jsonl(run / "s1_outputs.jsonl")}
    missing = [it["src_id"] for it in sample if it["src_id"] not in s1]
    if missing:
        raise SystemExit(f"s1_outputs.jsonl is missing {len(missing)} sources, e.g. {missing[:3]}")
    plan = plan_allocation([it["src_id"] for it in sample], (a.ann1, a.ann2),
                           a.n_shared, a.n_unique, a.n_href, a.n_check, a.seed)
    key, sheets = build_rows(sample, s1, plan, a.seed)
    out = write_outputs(run / "sheets", key, sheets)
    print({k: len(v) for k, v in sheets.items()}, "items per annotator ->", out)


def cmd_simulate(a):
    from .simulate import simulate
    out, paths = simulate(a.run, a.out, a.seed)
    print(f"SYNTHETIC annotations ({len(paths)} sheets) -> {out}   (test data only; never report)")


def cmd_analyze(a):
    from .report import run_analysis
    from .simulate import MARKER
    filled = Path(a.filled)
    paths = sorted(filled.glob("*.csv")) if filled.is_dir() else [filled]
    synthetic = filled.is_dir() and (filled / MARKER).exists()
    res = run_analysis(a.run, paths, a.out, a.primary, a.boot, a.seed, a.allow_problems, synthetic,
                       a.comet, a.force_comet)
    print(f"report -> {Path(a.out) / 'report.md'}" + ("   [SYNTHETIC]" if synthetic else ""))
    print({k: res[k] for k in ("n_items", "n_s1_items", "n_problems")})


def _common(p):
    p.add_argument("--n_shared", type=int, default=40)
    p.add_argument("--n_unique", type=int, default=30)
    p.add_argument("--seed", type=int, default=13)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--test", default="data/test.jsonl")
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--out", default="mqm_run")
    p.add_argument("--templated_share", type=float, default=0.3)
    _common(p)
    p.set_defaults(fn=cmd_sample)
    p = sub.add_parser("sheets")
    p.add_argument("--run", default="mqm_run")
    p.add_argument("--ann1", default="A1")
    p.add_argument("--ann2", default="A2")
    p.add_argument("--n_href", type=int, default=12)
    p.add_argument("--n_check", type=int, default=8)
    _common(p)
    p.set_defaults(fn=cmd_sheets)
    p = sub.add_parser("simulate", help="generate SYNTHETIC annotations to test the pipeline")
    p.add_argument("--run", default="mqm_run")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_simulate)
    p = sub.add_parser("analyze", help="score filled sheets and write tables, figures, report.md")
    p.add_argument("--run", default="mqm_run")
    p.add_argument("--filled", required=True, help="folder with the filled sheet CSVs (or one CSV)")
    p.add_argument("--out", default="mqm_run/results")
    p.add_argument("--primary", choices=["wmt_style", "official_style"], default="wmt_style")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--allow-problems", dest="allow_problems", action="store_true")
    p.add_argument("--comet", action="store_true", help="request COMET (skipped unless language support is verified)")
    p.add_argument("--force-comet", dest="force_comet", action="store_true")
    p.set_defaults(fn=cmd_analyze)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
