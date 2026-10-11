"""Full fine-tuning with a dedicated Kumaoni language token and a data mixture.

Stage A (sister-language pre-training):
  python -m kmt.train --preset mbart --out runs/A \
     --train data/train.jsonl:2,synth/synth.jsonl:1,synth/hi_aux.jsonl:0.5 --lr 5e-5 --epochs 6
Stage B (real data only, low LR):
  python -m kmt.train --preset mbart --init runs/A/final --out runs/B \
     --train data/train.jsonl:1 --lr 2e-5 --epochs 10
Mixture spec "path:weight": weight = multiplier on that file's size (>1 upsample, <1 subsample).
Rows are {"src","tgt"} and optional "lang" in {"kmn","hi"} (default "kmn").
Model selection is by chrF++ on the real dev set (not by loss).
"""
import argparse
import json
import random

import numpy as np
import sacrebleu
import torch
from datasets import Dataset
from transformers import (DataCollatorForSeq2Seq, EarlyStoppingCallback, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, set_seed)

from .common import PRESETS, load_with_kmn_token


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_mix(spec, seed):
    rng, rows = random.Random(seed), []
    for item in spec.split(","):
        path, _, w = item.partition(":")
        data = read_jsonl(path)
        rng.shuffle(data)
        k = int(len(data) * float(w or 1))
        rows += [dict(data[i % len(data)], lang=data[i % len(data)].get("lang", "kmn")) for i in range(k)]
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="mbart")
    ap.add_argument("--init", default=None, help="checkpoint dir (default: HF base model)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--optim", default="adamw_torch", help="try adafactor if you hit OOM")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    set_seed(a.seed)

    tok, model, ids = load_with_kmn_token(a.init or PRESETS[a.preset]["name"], a.preset)
    lang_id = {"kmn": ids["kmn"], "hi": ids["hi"]}
    model.generation_config.forced_bos_token_id = ids["kmn"]

    def encode(b):
        tok.src_lang = PRESETS[a.preset]["src"]
        enc = tok(b["src"], max_length=a.max_len, truncation=True)
        labels = []
        for t, lg in zip(b["tgt"], b["lang"]):
            body = tok(t, add_special_tokens=False, max_length=a.max_len - 2, truncation=True)["input_ids"]
            labels.append([lang_id[lg]] + body + [tok.eos_token_id])   # [lang] X [eos]
        enc["labels"] = labels
        return enc

    def to_ds(rows):
        return Dataset.from_list([{"src": r["src"], "tgt": r["tgt"], "lang": r.get("lang", "kmn")}
                                  for r in rows]).map(encode, batched=True, remove_columns=["src", "tgt", "lang"])

    train_ds, dev_ds = to_ds(load_mix(a.train, a.seed)), to_ds(read_jsonl(a.dev))
    print(f"train examples: {len(train_ds)}  dev examples: {len(dev_ds)}")

    def metrics(ev):
        preds, labels = ev
        preds = preds[0] if isinstance(preds, tuple) else preds
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        hyp = tok.batch_decode(preds, skip_special_tokens=True)
        ref = tok.batch_decode(labels, skip_special_tokens=True)
        return {"chrf": sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score,
                "bleu": sacrebleu.corpus_bleu(hyp, [ref]).score}

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    args = Seq2SeqTrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs * 2,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, num_train_epochs=a.epochs,
        warmup_ratio=0.05, lr_scheduler_type="cosine", weight_decay=0.01, label_smoothing_factor=0.1,
        optim=a.optim, bf16=bf16, fp16=cuda and not bf16, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="chrf",
        greater_is_better=True, predict_with_generate=True, generation_max_length=a.max_len,
        generation_num_beams=4, logging_steps=50, report_to="none", seed=a.seed)
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, pad_to_multiple_of=8),
        processing_class=tok, compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.train()
    trainer.save_model(f"{a.out}/final")
    tok.save_pretrained(f"{a.out}/final")
    print("best dev:", trainer.evaluate())


if __name__ == "__main__":
    main()
