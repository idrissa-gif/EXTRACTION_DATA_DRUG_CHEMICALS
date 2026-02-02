#!/usr/bin/env python3
import argparse
import os
import time
import json
import csv
import sys
from collections import defaultdict

from openai import OpenAI


def upload_and_finetune(judge_jsonl, base_model, n_epochs, lr_mult):
    client = OpenAI()  # Uses OPENAI_API_KEY from environment

    # 1) Upload training file
    with open(judge_jsonl, "rb") as f:
        file_response = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    print(f"Uploaded {judge_jsonl} -> file id {file_response.id}")

    # 2) Create fine-tuning job
    ft_job = client.fine_tuning.jobs.create(
        training_file=file_response.id,
        model=base_model,
        hyperparameters={
            "n_epochs": n_epochs,
            "learning_rate_multiplier": lr_mult
        }
    )
    print(f"Started fine-tune job: {ft_job.id} on base model {base_model}")

    # 3) Poll until completion
    while True:
        job_status = client.fine_tuning.jobs.retrieve(ft_job.id)
        print(f"[{time.strftime('%X')}] status: {job_status.status}")
        if job_status.status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(30)

    if job_status.status == "succeeded":
        print("✅ Fine-tune succeeded!")
        return job_status.fine_tuned_model
    else:
        raise RuntimeError(f"Fine-tune failed with status: {job_status.status}")


def extract_spans(tokens, tags):
    spans = []
    curr = []
    for tok, tg in zip(tokens, tags):
        if tg == "B":
            if curr:
                spans.append(" ".join(curr))
            curr = [tok]
        elif tg == "I" and curr:
            curr.append(tok)
        else:
            if curr:
                spans.append(" ".join(curr))
            curr = []
    if curr:
        spans.append(" ".join(curr))
    return spans


def evaluate_judge(judge_model, predictions_tsv):
    client = OpenAI()

    by_sent = defaultdict(lambda: {"tokens":[], "pred":[], "true":[]})
    with open(predictions_tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            idx = int(r["sentence_idx"])
            by_sent[idx]["tokens"].append(r["token"])
            by_sent[idx]["pred"].append(r["pred_tag"])
            by_sent[idx]["true"].append(r["true_tag"])

    tp = fp = fn = 0
    for data in by_sent.values():
        gold_spans = set(extract_spans(data["tokens"], data["true"]))
        pred_spans = extract_spans(data["tokens"], data["pred"])

        # false negatives for gold not predicted
        for g in gold_spans:
            if g not in pred_spans:
                fn += 1

        # judge each predicted span
        for span in pred_spans:
            prompt = f'Is "{span}" a biomedical chemical? Answer yes or no.\n\nAnswer:'
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            ans = resp.choices[0].message.content.strip().lower()
            is_chem = ans.startswith("yes")
            if is_chem:
                if span in gold_spans:
                    tp += 1
                else:
                    fp += 1
            else:
                if span in gold_spans:
                    fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_judge_direct(predictions_tsv, base_model="gpt-3.5-turbo"):
    """
    Alternative evaluation function that uses a pre-trained model directly
    without fine-tuning (useful for testing or if fine-tuning fails)
    """
    client = OpenAI()

    by_sent = defaultdict(lambda: {"tokens":[], "pred":[], "true":[]})
    with open(predictions_tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            idx = int(r["sentence_idx"])
            by_sent[idx]["tokens"].append(r["token"])
            by_sent[idx]["pred"].append(r["pred_tag"])
            by_sent[idx]["true"].append(r["true_tag"])

    tp = fp = fn = 0
    for data in by_sent.values():
        gold_spans = set(extract_spans(data["tokens"], data["true"]))
        pred_spans = extract_spans(data["tokens"], data["pred"])

        # false negatives for gold not predicted
        for g in gold_spans:
            if g not in pred_spans:
                fn += 1

        # judge each predicted span
        for span in pred_spans:
            prompt = f'Is "{span}" a biomedical chemical? Answer yes or no.\n\nAnswer:'
            try:
                resp = client.chat.completions.create(
                    model=base_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                ans = resp.choices[0].message.content.strip().lower()
                is_chem = ans.startswith("yes")
                if is_chem:
                    if span in gold_spans:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if span in gold_spans:
                        fn += 1
            except Exception as e:
                print(f"Error evaluating span '{span}': {e}")
                # Default to false negative if API call fails
                if span in gold_spans:
                    fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_jsonl",
                        help="Path to chem_judge.jsonl (required for fine-tuning)")
    parser.add_argument("--test_preds", required=True,
                        help="Path to test_predictions.tsv")
    parser.add_argument("--base_model", default="gpt-3.5-turbo",
                        help="Base model (gpt-3.5-turbo or gpt-4)")
    parser.add_argument("--n_epochs", type=int, default=2,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--lr_mult", type=float, default=0.1,
                        help="Learning rate multiplier")
    parser.add_argument("--metrics_out", default="judge_metrics.json",
                        help="Output JSON for metrics")
    parser.add_argument("--no_finetune", action="store_true",
                        help="Skip fine-tuning and use base model directly")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("❌ Please set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    # Determine which evaluation method to use
    if args.no_finetune or not args.judge_jsonl:
        print(f"🔄 Using base model {args.base_model} directly (no fine-tuning)")
        metrics = evaluate_judge_direct(args.test_preds, args.base_model)
    else:
        try:
            # Fine-tune the judge
            print(f"🔄 Fine-tuning {args.base_model}...")
            judge_model = upload_and_finetune(
                args.judge_jsonl,
                args.base_model,
                args.n_epochs,
                args.lr_mult
            )

            # Run evaluation with fine-tuned model
            print(f"🔄 Evaluating with fine-tuned model: {judge_model}")
            metrics = evaluate_judge(judge_model, args.test_preds)

        except Exception as e:
            print(f"❌ Fine-tuning failed: {e}")
            print(f"🔄 Falling back to base model {args.base_model}")
            metrics = evaluate_judge_direct(args.test_preds, args.base_model)

    # Save metrics
    with open(args.metrics_out, "w") as mf:
        json.dump(metrics, mf, indent=2)

    print(f"🚀 Saved metrics to {args.metrics_out}")
    print(f"📊 Results: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")


if __name__ == "__main__":
    main()