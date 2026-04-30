import argparse, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SYSTEM = "You are a precise chemical NER assistant. You ONLY output a JSON array of chemical names."
TEMPLATE = """<s>[SYSTEM]\n{system}\n[/SYSTEM]\n[USER]\n{instruction}\n\nText: {text}\n[/USER]\n[ASSISTANT]\n"""

def format_example(ex):
    prompt = TEMPLATE.format(system=SYSTEM, instruction=ex["instruction"], text=ex["input"])
    # Target is JSON list
    target = json.dumps(ex["output"], ensure_ascii=False)
    return {"prompt": prompt, "target": target}

def tokenize(tokenizer, ex):
    # We pack prompt + target and compute labels only for the target part
    prompt_ids = tokenizer(ex["prompt"])
    target_ids = tokenizer(ex["target"] + tokenizer.eos_token, add_special_tokens=False)

    input_ids  = prompt_ids["input_ids"] + target_ids["input_ids"]
    attention  = prompt_ids["attention_mask"] + target_ids["attention_mask"]

    labels = [-100]*len(prompt_ids["input_ids"]) + target_ids["input_ids"]
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="instruct_data/train.jsonl")
    ap.add_argument("--dev_jsonl",   default="instruct_data/dev.jsonl")
    ap.add_argument("--model_name",  default="mistralai/Mistral-7B-Instruct")
    ap.add_argument("--out_dir",     default="mistral-chem-instruct-lora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=1024)
    args = ap.parse_args()

    ds = load_dataset("json", data_files={"train": args.train_jsonl, "validation": args.dev_jsonl})
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = ds.map(format_example)
    ds_tok = ds.map(lambda ex: tokenize(tok, ex), remove_columns=ds["train"].column_names, batched=False)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_4bit=True, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    args_hf = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        report_to="none",
    )

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    trainer = Trainer(
        model=model,
        args=args_hf,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()
    trainer.save_model(f"{args.out_dir}/final")
    tok.save_pretrained(f"{args.out_dir}/final")

if __name__ == "__main__":
    main()
