# English -> Kumaoni v2: Hindi as sister language + linguistic checks

Drop `kmt/`, `tests/`, `app.py`, `build_cache.py`, `requirements.txt` into your repo root.
**Status: unit tests for the pure-python parts pass; the training/inference scripts are
lint-checked but have NOT been run end-to-end (no GPU / Hugging Face access when written).**
Expect to fix small API-version issues on first run (needs transformers >= 4.46).

## Run order (Colab GPU recommended)
```bash
pip install -r requirements.txt
python -m pytest -q tests                                   # sanity
python -m kmt.prep_data --out data                          # leak-free split + stats
# --- baseline numbers FIRST (so you know what "better" means) ---
python -m kmt.train --preset mbart --out runs/base --train data/train.jsonl:1 --lr 5e-5 --epochs 10
python -m kmt.evaluate --model runs/base/final --preset mbart
# --- sister-language transfer ---
python -m kmt.transfer pivot   --data data/train.jsonl --out synth/pivot.jsonl
python -m kmt.transfer lexicon --pivot synth/pivot.jsonl --out synth/lexicon.tsv
#   >>> native speaker reviews synth/lexicon.tsv (col 5: y / n); fill kmt/rules.yaml <<<
python -m kmt.transfer synth   --lexicon synth/lexicon.tsv --rules kmt/rules.yaml --out synth
python -m kmt.train --preset mbart --out runs/A --train data/train.jsonl:2,synth/synth.jsonl:1,synth/hi_aux.jsonl:0.5 --epochs 6
python -m kmt.train --preset mbart --init runs/A/final --out runs/B --train data/train.jsonl:1 --lr 2e-5
python -m kmt.transfer leaklist --out synth/hindi_only.txt
python -m kmt.evaluate --model runs/B/final --hindi_only synth/hindi_only.txt --alpha 0   # then 0.5, 1, 2
python build_cache.py && python app.py
```
## Ablations to report in the thesis
base (kfy token only) -> + Hindi aux -> + synthetic relexified data -> + reviewed lexicon/rules
-> + leakage reranking -> NLLB-600M instead of mBART-50 (`--preset nllb`).
Always report chrF++ (BLEU is brittle on unstandardised spelling) plus a native-speaker rating.
