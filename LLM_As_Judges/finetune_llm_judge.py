#!/usr/bin/env python3
import argparse, json, os, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_raw(path):
    with open(path, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

def build_prompt(ex):
    sent   = " ".join(ex["tokens"])
    tags   = ex["tags"]
    prompt = (
        f"<start> {sent} <end>\n"
        f"Tokens (space separated): {sent}\n"
        f"Total tokens: {len(ex["tokens"])}\n"
        f"Legend: B=begin, I=inside, O=outside\n"
        f"Format: O O B I O …\n"
        f"Tags:"
    )
    completion = " " + " ".join(tags) + " <end>"
    return {"prompt": prompt, "completion": completion}

class PromptDataset:
    def __init__(self, raw, tokenizer, max_length):
        self.tok = tokenizer
        self.max_length = max_length

        examples = [build_prompt(x) for x in raw]
        ds = Dataset.from_list(examples)
        self.dataset = ds.map(self._encode, remove_columns=["prompt","completion"])
        self.dataset.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    def _encode(self, ex):
        # Combine prompt and completion for full context
        full_text = ex["prompt"] + ex["completion"]

        # Tokenize the full text
        enc = self.tok(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None  # Return lists, not tensors
        )

        # Tokenize just the prompt to find where to start predicting
        prompt_enc = self.tok(
            ex["prompt"],
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
            add_special_tokens=False
        )

        input_ids = enc["input_ids"]
        labels = input_ids.copy()

        # Mask the prompt portion (set to -100)
        prompt_len = len(prompt_enc["input_ids"])
        for i in range(prompt_len):
            if i < len(labels):
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels": labels
        }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",      required=True)
    p.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.1")
    p.add_argument("--out_dir",    default="bio_judge_lora")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--max_len",    type=int,   default=256)
    args = p.parse_args()

    raw = load_raw(args.jsonl)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = PromptDataset(raw, tokenizer, args.max_len)

    print(f"→ Loading {args.base_model} in 8-bit with LoRA…")

    # Use BitsAndBytesConfig instead of deprecated load_in_8bit
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # Prepare model for k-bit training (essential for gradient computation)
    model = prepare_model_for_kbit_training(model)

    # Resize token embeddings if needed
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    # LoRA configuration with more comprehensive target modules
    lora_cfg = LoraConfig(
        r=16,  # Increased rank for better performance
        lora_alpha=32,  # Increased alpha
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # All attention projections
            "gate_proj", "up_proj", "down_proj"       # MLP projections
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False  # Ensure training mode
    )
    model = get_peft_model(model, lora_cfg)

    # Print trainable parameters
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,  # Increased for better stability
        num_train_epochs=args.epochs,
        learning_rate=1e-4,  # Slightly lower learning rate
        warmup_ratio=0.03,   # Add warmup
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        report_to=[],
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,  # Enable gradient checkpointing properly
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds.dataset,
        data_collator=data_collator,
    )

    print("🚀 Starting training...")
    trainer.train()

    # Save the LoRA adapters
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"✅ Fine-tuned model saved to {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()