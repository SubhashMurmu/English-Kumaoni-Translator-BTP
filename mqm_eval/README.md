# mqm_eval: human evaluation of English → Kumaoni output (MQM + WMT-inspired statistics)

Evaluation only; no model training. Install: `pip install -r mqm_eval/requirements.txt`.

## Workflow

```bash
# 1. fixed, reproducible sample of held-out sources (100 by default)
python -m mqm_eval.cli sample --test data/test.jsonl --train data/train.jsonl --out mqm_run

# 2. run YOUR model on mqm_run/sample.jsonl and save mqm_run/s1_outputs.jsonl
#    rows: {"src_id": ..., "text": ...}   (do not change after annotation starts)

# 3. blinded sheets for two annotators + private key (never give the key to annotators)
python -m mqm_eval.cli sheets --run mqm_run

# 4. annotators fill sheet_A1.csv / sheet_A2.csv following docs/guideline.md

# 5. score, agreement, bootstrap CIs, tables, figures, report.md
python -m mqm_eval.cli analyze --run mqm_run --filled path/to/filled_sheets --out mqm_run/results
```

Test the pipeline without annotators using SYNTHETIC data (never report these results):
`python -m mqm_eval.cli simulate --run mqm_run` then `analyze --filled mqm_run/synthetic_filled`.

## What is official and what is a project choice

| Part | Status |
|---|---|
| Error dimensions, Neutral/Minor/Major/Critical, penalty = severity multiplier × type weight, normalising by word count | MQM (themqm.org); the 0/1/5/25 multipliers and 1000-word basis are the page's **examples** |
| Weights 5 / 1 / 0.1 / 25, cap of 5 errors per segment, mean over annotators | Freitag et al. 2021 (WMT-style MQM); Critical → 25 is our adaptation |
| `language_identity` (Hindi leakage), `spelling_variant` (neutral), `dialect_mismatch` | Project-specific |
| Paired bootstrap over source sentences, Wilcoxon, Mann-Whitney | WMT-inspired statistics (WMT24 used Wilcoxon signed-rank for system ranking) |
| Agreement statistics, human-reference control, attention checks by mismatched pairing | Project choices |
| COMET | Skipped: Kumaoni is not in the verified XLM-R language list |

## Modules

`schema` (columns, validation) · `sampling` · `sheets` · `load` · `scoring` · `agreement` ·
`inference` (bootstrap, tests) · `metrics_corr` · `report` · `simulate` · `cli`.
Configuration is in `config/` (typology, penalty presets).
