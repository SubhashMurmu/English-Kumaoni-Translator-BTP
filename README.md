# English → Kumaoni Neural Machine Translation

A neural machine translation system for **English-to-Kumaoni**, developed as a **Bachelor's Thesis Project (BTP)** at IIT Kharagpur.

Kumaoni is an Indo-Aryan language spoken mainly in the Kumaon region of Uttarakhand, India, and is low-resource for machine translation. This project fine-tunes Meta's **mBART-50** and explores using **Hindi, a closely related language, as a sister language** (shared script and much shared vocabulary and grammar) together with linguistically motivated checks.

---

## Status

| Component | Status |
| --- | --- |
| Leak-free data preparation (deterministic split, de-duplication) | Done |
| Baseline: mBART-50 fine-tuned on English → Kumaoni (Roman script) | Done, evaluated |
| Web translator (`server.py` + `web/`) | Working locally |
| Roman → Devanagari conversion tooling (`kmt/romanize.py`) | Built; reviewed word list pending |
| Hindi sister-language transfer (lexicon induction, relexified synthetic data) | Implemented, **not yet run or evaluated** |
| Retraining on Devanagari targets | Planned |
| Hindi-leakage metric and reranking | Implemented, not yet evaluated |

### Results so far (Roman-script baseline)

Model: `facebook/mbart-large-50-many-to-many-mmt`, full fine-tuning, Adafactor, learning rate 1e-4, 5 epochs, with a dedicated Kumaoni language token initialised from Hindi's.

| Split | chrF++ | BLEU |
| --- | --- | --- |
| Dev (547 sentences, greedy decoding) | 50.6 | 25.8 |
| Held-out test (beam search, 5 beams) | 50.3 | 24.4 |

How to read these numbers:

- Scores are for **Roman-script** Kumaoni, the script of the source dataset. They are **not comparable** with future Devanagari results.
- Many sentences in the dataset are short or templated, which inflates scores. A native-speaker evaluation of translation adequacy and fluency has not been done yet.
- Spelling in the dataset varies a lot (for example `chhan / chha / chhe / chho`), so part of the remaining error is inconsistent romanisation rather than wrong translation.
- The earlier LoRA version of this project reported only training loss, so no earlier score exists to compare against.

---

## Approach

1. **Leak-free evaluation.** The train/dev/test split is deterministic (hash of the English source), exact duplicates are removed, and near-duplicate dev/test sentences are moved into training. The translation memory is built from the training split only.
2. **A separate Kumaoni language token.** Kumaoni gets its own token (`kfy_IN`, from ISO 639-3 `kfy`) whose embedding starts as a copy of Hindi's. The model keeps the transfer benefit while Kumaoni and Hindi can diverge. This requires full fine-tuning rather than LoRA, because a frozen embedding could not tell the two tokens apart.
3. **Hindi as sister language** *(implemented, not yet evaluated)*:
   - English is translated to Hindi to obtain Hindi–Kumaoni pairs, and a Hindi → Kumaoni word lexicon is induced by word alignment (simalign).
   - A native speaker reviews the lexicon, and hand-written rewrite rules (`kmt/rules.yaml`) cover verb, postposition and pronoun differences.
   - A large English–Hindi corpus is relexified into pseudo-Kumaoni for a first training stage, followed by fine-tuning on the real data.
4. **Devanagari output.** The dataset is in Roman script, so `kmt/romanize.py` converts it word by word with an automatic fallback and a reviewed override list for frequent words.
5. **Linguistic checks.** A data-driven Hindi-only word list measures and penalises Hindi "leakage" in outputs, and the translation memory only accepts fuzzy matches whose numbers, names and negation agree with the query.

---

## Repository structure

```text
English-Kumaoni-Translator-BTP/
├── server.py                 Website + JSON API (FastAPI)
├── web/index.html            Translator page
├── app.py                    Optional Gradio demo
├── build_cache.py            Builds the translation memory (used by app.py)
├── requirements.txt
├── kmt/
│   ├── prep_data.py          Clean + leak-free split
│   ├── romanize.py           Roman → Devanagari conversion (word list + overrides)
│   ├── common.py             Model presets, Kumaoni language token
│   ├── train.py              Full fine-tuning, model selection by chrF++
│   ├── evaluate.py           chrF++, BLEU, Hindi-leak rate, outputs for human review
│   ├── infer.py              Translator (n-best + leak reranking), translation memory
│   ├── transfer.py           Hindi pivot, lexicon induction, synthetic data, leak list
│   ├── relex.py              Lexicon + rewrite-rule engine
│   ├── text.py               Normalisation, leakage metric, memory guard
│   └── rules.yaml            Hindi → Kumaoni rewrite rules (to be filled with a linguist)
├── tests/test_pure.py        Unit tests for the pure-Python parts
└── old/                      Original LoRA notebook and scripts (v1), kept for reference
```

Generated or downloaded locally and **not committed**: `data/`, `data_dev/`, `synth/`, `runs/` (model weights), `*.zip`.

---

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/SubhashMurmu/English-Kumaoni-Translator-BTP.git
cd English-Kumaoni-Translator-BTP
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For CPU-only inference you can use the smaller PyTorch build first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## Running the website

The trained model is **not in the repository** (about 2.4 GB). Train it as described below, or obtain the `runs/base/final/` folder from the authors, and place it at `runs/base/final/` in the project root.

```bash
uvicorn server:app --port 8000
```

Open `http://127.0.0.1:8000`. The model needs about 3–4 GB of RAM and takes a minute to load; on CPU, each translation takes a few seconds.

| Environment variable | Meaning | Default |
| --- | --- | --- |
| `KMT_MODEL` | Model directory | `runs/base/final` |
| `KMT_PRESET` | `mbart` or `nllb` | `mbart` |
| `KMT_SCRIPT` | `devanagari` (convert Roman output) or `roman` | `devanagari` |
| `KMT_ALPHA` | Hindi-leak reranking weight (0 = off) | `0` |
| `KMT_DEMO=1` | Echo mode without loading the model, for UI work | off |

Until the model is retrained on Devanagari data, the server converts the model's Roman output with `kmt/romanize.py`, and spellings will be imperfect until the override list is reviewed.

The older Gradio interface is still available: run `python build_cache.py` (needs `data/train.jsonl`), then `python app.py`.

---

## Training and evaluation

A GPU is needed for training (a Colab or Kaggle T4 is enough; the baseline took about 65 minutes for 5 epochs).

```bash
# 1. Clean, de-duplicate and split the dataset
python -m kmt.prep_data --out data

# 2. Baseline
python -m kmt.train --preset mbart --out runs/base --train data/train.jsonl:1 \
    --lr 1e-4 --epochs 5 --optim adafactor --max_len 96 --resume

# 3. Evaluate on the held-out test set
python -m kmt.evaluate --model runs/base/final --out eval_base.tsv

# 4. Devanagari conversion (after reviewing synth/overrides.tsv)
python -m kmt.romanize vocab   --data data --out synth/roman_vocab.tsv
python -m kmt.romanize convert --data data --out data_dev --overrides synth/overrides.tsv
```

The Hindi-transfer steps (`python -m kmt.transfer pivot | lexicon | synth | leaklist`) are documented at the top of `kmt/transfer.py`. Run the unit tests with `python -m pytest -q tests`.

The training and Hindi-transfer scripts need a recent `transformers` (4.46 or newer). If `--optim adafactor` runs out of memory, lower `--bs` and raise `--grad_accum`.

---

## Limitations

- The dataset is small (about 10,000 training pairs), partly templated, and its dialect and spelling conventions are not documented. Translations may differ from the dialect a given speaker uses.
- The conversion to Devanagari is only as good as the reviewed word list; unreviewed words fall back to an automatic rule-based conversion that is wrong for some words (loanwords, final long vowels).
- No human evaluation has been done yet. Treat the automatic scores as indicative.
- Translations can be wrong. Check anything important with a Kumaoni speaker.

---

## Technologies

PyTorch, Hugging Face Transformers, mBART-50, FastAPI, sacreBLEU, simalign and `indic-transliteration` (conversion and lexicon induction), FAISS and sentence-transformers (translation memory), Gradio (optional demo). The Hindi-transfer pipeline also uses NLLB-200 (distilled 600M) for English → Hindi pivoting and the IITB English–Hindi corpus for synthetic data.

---

## Acknowledgements

Meta AI's mBART-50 and NLLB-200, Hugging Face Transformers, FAISS, and the open-source authors of the tools listed above.

---

## Authors

**Subhash Murmu** & **Akshat Jain**

Bachelor's Thesis Project, Indian Institute of Technology Kharagpur

---

## License

This project is intended for **academic and research purposes**. Check the licenses of the underlying models (mBART-50, NLLB-200), the dataset and the dependencies before redistributing the model or deploying the system.
