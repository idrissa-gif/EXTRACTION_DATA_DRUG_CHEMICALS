import argparse
import time
import json
import os
import csv
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# JSON encoder for NumPy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a BioBERT-based NER model with customizable dataset paths."
    )
    parser.add_argument(
        "--train_file", type=str, required=True,
        help="Path to the training TSV file (token<TAB>tag format, blank line between sentences)."
    )
    parser.add_argument(
        "--dev_file", type=str, required=True,
        help="Path to the development/validation TSV file."
    )
    parser.add_argument(
        "--test_file", type=str, required=True,
        help="Path to the test TSV file."
    )
    parser.add_argument(
        "--output_dir", type=str, default="biobert_output",
        help="Directory where outputs (checkpoints, metrics, preds) will be saved."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load + group tokens into sentences
    def load_grouped_tsv(path):
        sentences, tags = [], []
        ctoks, ctags = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if ctoks:
                        sentences.append(ctoks); tags.append(ctags)
                        ctoks, ctags = [], []
                else:
                    parts = line.split("\t")
                    if len(parts) != 2:
                        continue
                    tok, tag = parts
                    ctoks.append(tok); ctags.append(tag)
            if ctoks:
                sentences.append(ctoks); tags.append(ctags)
        return Dataset.from_dict({"token": sentences, "tag": tags})

    files = {
        "train":      args.train_file,
        "validation": args.dev_file,
        "test":       args.test_file,
    }
    datasets = DatasetDict({split: load_grouped_tsv(path)
                            for split, path in files.items()})

    # 2. Label <-> id maps
    all_tags   = { tag for seq in datasets["train"]["tag"] for tag in seq }
    label_list = sorted(t for t in all_tags if t)
    label2id   = {lab:i for i, lab in enumerate(label_list)}
    id2label   = {i:lab for lab, i in label2id.items()}

    # 3. Tokenizer & model_init
    model_name = "alvaroalon2/biobert_genetic_ner"
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    def model_init():
        return AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
        )

    # 4. Tokenize + align labels
    def tokenize_and_align_labels(examples):
        tok = tokenizer(
            examples["token"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
        )
        all_labels = []
        for i, enc in enumerate(tok.encodings):
            wids = enc.word_ids  # use attribute, not method
            labs = [(-100 if wid is None else label2id[ examples["tag"][i][wid] ])
                    for wid in wids]
            all_labels.append(labs)
        tok["labels"] = all_labels
        return tok

    tokenized = datasets.map(
        tokenize_and_align_labels,
        batched=True,
        batch_size=500,
        remove_columns=["token","tag"]
    )

    # 5. Metrics function
    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=2)
        true_seqs, pred_seqs = [], []
        for t_seq, p_seq in zip(label_ids, preds):
            t_tags, p_tags = [], []
            for t, p in zip(t_seq, p_seq):
                if t == -100: continue
                t_tags.append(id2label[t]); p_tags.append(id2label[p])
            true_seqs.append(t_tags)
            pred_seqs.append(p_tags)
        return {
            "precision": precision_score(true_seqs, pred_seqs),
            "recall":    recall_score(true_seqs, pred_seqs),
            "f1":        f1_score(true_seqs, pred_seqs),
        }

    # 6. TrainingArguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "model_ckpt"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=100,
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    # 7. HPO space
    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8,16,32]),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        }

    data_collator = DataCollatorForTokenClassification(tokenizer)
    hp_trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 8. Run HPO
    best_run = hp_trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=10,
        compute_objective=lambda metrics: metrics["eval_f1"],
    )
    print("Best HPO parameters:", best_run.hyperparameters)

    # 9. Apply best HPO params
    for k, v in best_run.hyperparameters.items():
        setattr(hp_trainer.args, k, v)

    # 10. Final training
    t0 = time.time()
    train_out   = hp_trainer.train()
    train_time  = time.time() - t0
    train_mets  = train_out.metrics
    train_mets["train_time_sec"] = train_time
    with open(os.path.join(args.output_dir, "train_metrics.json"), "w") as f:
        json.dump(train_mets, f, indent=2, cls=NpEncoder)
    print(f"✔ Training done in {train_time:.1f}s → train_metrics.json")

    # 11. Final eval
    t0 = time.time()
    eval_mets   = hp_trainer.evaluate()
    eval_time   = time.time() - t0
    eval_mets["eval_time_sec"] = eval_time
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
        json.dump(eval_mets, f, indent=2, cls=NpEncoder)
    print(f"✔ Validation done in {eval_time:.1f}s → eval_metrics.json")

    # 12. Test predictions
    t0 = time.time()
    pred_logits, true_ids, _ = hp_trainer.predict(tokenized["test"])
    pred_time  = time.time() - t0
    pred_ids   = np.argmax(pred_logits, axis=2)

    records, true_seqs, pred_seqs = [], [], []
    for i, (t_seq, p_seq, orig) in enumerate(zip(true_ids, pred_ids, tokenized["test"])):
        t_tags, p_tags = [], []
        for j, (tid, pid) in enumerate(zip(t_seq, p_seq)):
            if tid == -100: continue
            tok = tokenizer.convert_ids_to_tokens(orig["input_ids"][j])
            tt, pt = id2label[tid], id2label[pid]
            t_tags.append(tt); p_tags.append(pt)
            records.append({
                "sentence_idx": i,
                "token_idx":     j,
                "token":         tok,
                "true_tag":      tt,
                "pred_tag":      pt,
            })
        true_seqs.append(t_tags); pred_seqs.append(p_tags)

    test_mets = {
        "precision": precision_score(true_seqs, pred_seqs),
        "recall":    recall_score(true_seqs, pred_seqs),
        "f1":        f1_score(true_seqs, pred_seqs),
        "classification_report": classification_report(true_seqs, pred_seqs, output_dict=True),
        "pred_time_sec": pred_time,
    }
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_mets, f, indent=2, cls=NpEncoder)
    print(f"✔ Test done in {pred_time:.1f}s → test_metrics.json")

    # 13. Save TSV preds
    with open(os.path.join(args.output_dir, "test_predictions.tsv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    print("✔ Saved test_predictions.tsv")

    # 14. Save model + tokenizer
    final_ckpt = os.path.join(args.output_dir, "final_checkpoint")
    hp_trainer.save_model(final_ckpt)
    tokenizer.save_pretrained(final_ckpt)
    print(f"✔ Model & tokenizer saved → {final_ckpt}")
