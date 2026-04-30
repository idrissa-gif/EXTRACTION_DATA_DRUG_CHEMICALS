import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

# --- CONFIGURATION ---
# 1. BASE MODEL: Must be the original Mistral, NOT your folder
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# 2. YOUR ADAPTER: Point to your previous checkpoint here
PREVIOUS_ADAPTER_PATH = "../fairClinical/mistral-chem-instruct-lora/final"

# 3. NEW OUTPUT: Where to save the further improved model
OUTPUT_DIR = "./mistral-chem-v2-finetuned"
DATA_FILE = "train_mistral_multi.jsonl"

def train():
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # 1. Load Base Model (Quantized & Safe Mode for A6000)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32, # Safe mode
    )

    print(f"Loading base model: {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float32,
    )

    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load Your Existing Adapter (The Resume Step)
    print(f"Loading existing adapter from {PREVIOUS_ADAPTER_PATH}...")

    # Prepare base model for k-bit training first
    model = prepare_model_for_kbit_training(base_model)

    # Load your specific adapter and make it trainable again
    model = PeftModel.from_pretrained(
        model,
        PREVIOUS_ADAPTER_PATH,
        is_trainable=True
    )

    print("Ensuring trainable params are Float32...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    # 4. Training Arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        logging_steps=25,
        save_strategy="epoch",
        learning_rate=1e-4,

        fp16=False,
        bf16=False,

        max_grad_norm=0.3,
        warmup_steps=50,
        lr_scheduler_type="constant",
        max_length=512,
        packing=False,
        dataset_text_field="text",
    )

    print("\nStarting training (Resuming from adapter)...")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=lambda s: f"<s>[INST] {s['instruction']}\nInput: {s['input']} [/INST] Output: {s['output']}</s>",
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"Training complete. Saving new version to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()