#!/usr/bin/env python3
import argparse
import os
import csv
import json
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import precision_score, recall_score, f1_score

def load_grouped_tsv(path):
    sentences, tags = [], []
    ctoks, ctags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if ctoks:
                    sentences.append(ctoks); tags.append(ctags)
                    ctoks, ctags = [], []
            else:
                tok, tag = line.split("\t")
                ctoks.append(tok); ctags.append(tag)
        if ctoks:
            sentences.append(ctoks); tags.append(ctags)
    return Dataset.from_dict({"token": sentences, "tag": tags})

def tokenize_and_align_labels(examples, tokenizer, label2id, max_length):
    tok = tokenizer(
        examples["token"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    all_labels = []
    for i, enc in enumerate(tok.encodings):
        word_ids = enc.word_ids
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            else:
                label_ids.append(label2id[ examples["tag"][i][wid] ])
        all_labels.append(label_ids)
    tok["labels"] = all_labels
    return tok

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",   required=True,
                        help="Path to BioBERT final_checkpoint directory")
    parser.add_argument("--test_file",   required=True,
                        help="Path to combined_chem_test.tsv")
    parser.add_argument("--output_tsv",  default="test_predictions.tsv",
                        help="Where to write predictions")
    parser.add_argument("--metrics_out", default="biobert_metrics.json",
                        help="Where to write metrics")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--max_length",  type=int, default=128)
    args = parser.parse_args()

    # 1) Load raw test sentences (original tokens & true tags)
    raw_ds = load_grouped_tsv(args.test_file)
    raw_sentences = raw_ds["token"]  # List[List[str]]
    raw_tags      = raw_ds["tag"]

    # 2) Build a DatasetDict for tokenization/prediction
    ds = DatasetDict({"test": raw_ds})

    # 3) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model     = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    # 4) Label maps
    label2id = model.config.label2id
    id2label = model.config.id2label

    # 5) Tokenize + align labels for prediction
    tokenized = ds.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, label2id, args.max_length),
        batched=True,
        batch_size=100,
        remove_columns=["token","tag"]
    )

    # 6) Prepare Trainer
    data_collator = DataCollatorForTokenClassification(tokenizer)
    training_args = TrainingArguments(
        output_dir="tmp",
        per_device_eval_batch_size=args.batch_size,
        logging_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 7) Run prediction
    pred_out = trainer.predict(tokenized["test"])
    logits, label_ids = pred_out.predictions, pred_out.label_ids
    pred_ids = np.argmax(logits, axis=2)

    # 8) Write out original‐token predictions
    with open(args.output_tsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sentence_idx","token_idx","token","true_tag","pred_tag"],
            delimiter="\t"
        )
        writer.writeheader()

        all_true_seqs = []
        all_pred_seqs = []

        for i, (words, true_tags, lid_seq, pid_seq) in enumerate(zip(
                raw_sentences, raw_tags, label_ids, pred_ids
            )):
            # re-tokenize to get word_ids mapping
            enc = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
            )
            wids = enc.word_ids()  # maps each subword to word index or None

            word_preds = {}
            true_seq = []
            pred_seq = []

            for j, wid in enumerate(wids):
                if wid is None:
                    continue
                token_str = words[wid]
                t_label = true_tags[wid]
                p_label = id2label[ pid_seq[j] ]

                # If it's the first time seeing the word
                if wid not in word_preds:
                    word_preds[wid] = {
                        "token":    token_str,
                        "true_tag": t_label,
                        "pred_tag": p_label,
                    }
                else:
                    # If current subword prediction does NOT match the true label, update prediction
                    if p_label != t_label:
                        word_preds[wid]["pred_tag"] = p_label  # overwrite with this incorrect one

            # Write final predictions
            for wid in sorted(word_preds):
                entry = word_preds[wid]
                writer.writerow({
                    "sentence_idx": i,
                    "token_idx":    wid,
                    "token":        entry["token"],
                    "true_tag":     entry["true_tag"],
                    "pred_tag":     entry["pred_tag"],
                })
                true_seq.append(entry["true_tag"])
                pred_seq.append(entry["pred_tag"])

            all_true_seqs.append(true_seq)
            all_pred_seqs.append(pred_seq)

    # 9) Compute entity‐level metrics
    prec = precision_score(all_true_seqs, all_pred_seqs)
    rec  = recall_score(all_true_seqs, all_pred_seqs)
    f1   = f1_score(all_true_seqs, all_pred_seqs)
    metrics = {"precision":prec, "recall":rec, "f1":f1}

    with open(args.metrics_out, "w") as mf:
        json.dump(metrics, mf, indent=2)

    print(f"▶ Predictions saved to {args.output_tsv}")
    print(f"▶ Metrics saved to   {args.metrics_out}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()
