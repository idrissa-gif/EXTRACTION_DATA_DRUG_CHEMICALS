import argparse
import json
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# MUST match the model used in training exactly
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
ADAPTER_PATH = "mistral-chem-instruct-lora/final"
TEST_FILE = "instruct_data/test.jsonl"

SYSTEM_PROMPT = "You are a precise chemical NER assistant. You ONLY output a JSON array of chemical names."
TEMPLATE = """<s>[SYSTEM]\n{system}\n[/SYSTEM]\n[USER]\n{instruction}\n\nText: {text}\n[/USER]\n[ASSISTANT]\n"""

def main():
    print(f"Loading Base Model: {BASE_MODEL_ID} ...")

    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    except:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model (4-bit to save memory)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 3. Load Adapter
    print(f"Loading Adapter: {ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    # 4. Run Inference on first 5 examples
    print("\n--- DEBUGGING GENERATION (First 5 Items) ---")

    with open(TEST_FILE, "r") as f:
        data = [json.loads(line) for line in f][:5]

    for i, ex in enumerate(data):
        prompt = TEMPLATE.format(system=SYSTEM_PROMPT, instruction=ex["instruction"], text=ex["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode only the new tokens
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        print(f"\n[Example {i+1}]")
        print(f"Ground Truth: {ex['output']}")
        print(f"Prediction:   {output_text}")
        print("-" * 40)

if __name__ == "__main__":
    main()