#!/usr/bin/env python3
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",      required=True, help="chem_judge.jsonl")
    parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--out_dir",    default="judge_llama_lora")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--bs",         type=int, default=4)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--max_len",    type=int, default=128)
    args = parser.parse_args()

    # 1) Load dataset
    ds = load_dataset("json", data_files=args.jsonl, split="train")

    # 2) Tokenizer & model
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        device_map="auto",
    )
    model.config.pad_token_id = tok.eos_token_id

    # 3) Apply LoRA
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # 4) Tokenize & pad to max_len
    def tokenize_fn(ex):
        prompt, comp = ex["prompt"], ex["completion"]
        text = prompt + " " + comp
        t = tok(
            text,
            truncation=True,
            max_length=args.max_len,
            padding="max_length",
        )
        pid = tok(
            prompt,
            truncation=True,
            max_length=args.max_len,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        labels = t["input_ids"].copy()
        labels[: len(pid)] = [-100] * len(pid)
        t["labels"] = labels
        return t

    ds = ds.map(tokenize_fn, batched=False, remove_columns=["prompt", "completion"])

    # 5) Convert to torch Tensors
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 6) Train/test split
    split_ds = ds.train_test_split(test_size=0.2)
    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]

    # 7) Data collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # 8) Training args with early stopping
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bs,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # 9) Trainer with EarlyStoppingCallback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # 10) Train & save
    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"✅ Saved fine-tuned model to {args.out_dir}")

if __name__ == "__main__":
    main()