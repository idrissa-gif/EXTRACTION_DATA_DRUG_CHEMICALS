import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import json
import torch
import gc
import re
import numpy as np
from tqdm import tqdm
from collections import Counter
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = "You are a precise chemical NER assistant. You ONLY output a JSON array of chemical names."
TEMPLATE = """<s>[SYSTEM]\n{system}\n[/SYSTEM]\n[USER]\n{instruction}\n\nText: {text}\n[/USER]\n[ASSISTANT]\n"""

def format_prompt(instruction, text):
    return TEMPLATE.format(system=SYSTEM_PROMPT, instruction=instruction, text=text)

def extract_json_list(text):
    text = text.strip()
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return []

def calculate_metrics_from_file(filename):
    """Reads the JSONL file and calculates metrics at the end."""
    print(f"Reading results from {filename}...")
    references = []
    predictions = []
    exact_matches = 0
    macro_f1s = []

    total_tp, total_fp, total_fn = 0, 0, 0

    with open(filename, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                t_list = item["ground_truth"]
                p_list = item["prediction"]

                references.append(t_list)
                predictions.append(p_list)

                # Metrics logic
                t_clean = [str(x).strip() for x in t_list]
                p_clean = [str(x).strip() for x in p_list]

                if sorted(t_clean) == sorted(p_clean): exact_matches += 1

                t_c = Counter(t_clean)
                p_c = Counter(p_clean)

                tp = sum((t_c & p_c).values())
                fp = sum((p_c - t_c).values())
                fn = sum((t_c - p_c).values())

                total_tp += tp; total_fp += fp; total_fn += fn

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                macro_f1s.append(f1)
            except:
                continue

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    return {
        "processed_count": len(references),
        "exact_match": exact_matches / len(references) if references else 0,
        "micro_f1": micro_f1,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "macro_f1": np.mean(macro_f1s) if macro_f1s else 0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="instruct_data/test.jsonl")
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--adapter_path", default="mistral-chem-instruct-lora/final")
    parser.add_argument("--output_file", default="results_test_stream.jsonl") # JSONL format for streaming
    parser.add_argument("--batch_size", type=int, default=32) # LOWERED DEFAULT TO 4
    args = parser.parse_args()

    # 1. Check for Resume
    processed_count = 0
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            processed_count = sum(1 for _ in f)
    print(f"Found {processed_count} examples already processed. Resuming...")

    # 2. Load Tokenizer
    print(f"Loading Base Model: {args.base_model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 3. Load Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    # 4. Load Data
    with open(args.test_file, "r") as f:
        data = [json.loads(line) for line in f]

    print(f"Total data: {len(data)}. Starting from index {processed_count}...")

    # 5. Inference Loop
    batch_size = args.batch_size

    # Open file in APPEND mode
    f_out = open(args.output_file, "a", encoding="utf-8")

    for i in tqdm(range(processed_count, len(data), batch_size)):
        batch_items = data[i : i + batch_size]
        batch_prompts = [format_prompt(item["instruction"], item["input"]) for item in batch_items]

        try:
            # Move inputs to CUDA inside try block to catch OOM
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True
                )

            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_len:]
            decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            for j, text in enumerate(decoded_texts):
                pred_list = extract_json_list(text)
                true_list = batch_items[j]["output"]

                # Write immediately to disk
                result_entry = {
                    "id": i + j,
                    "ground_truth": true_list,
                    "prediction": pred_list,
                    "raw": text[:100] # Log snippet
                }
                f_out.write(json.dumps(result_entry) + "\n")

            # Force write to disk so data isn't lost on crash
            f_out.flush()

        except Exception as e:
            print(f"\n| ERROR in batch {i}: {e}")
            print("| Skipping this batch to preserve stability.")

            # Write empty entries so the count stays correct for resuming
            for j in range(len(batch_items)):
                f_out.write(json.dumps({"id": i+j, "ground_truth": batch_items[j]["output"], "prediction": [], "error": str(e)}) + "\n")
            f_out.flush()

            torch.cuda.empty_cache()
            gc.collect()

        if i % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    f_out.close()

    # 6. Final Metrics Calculation
    print("\nCalculated Metrics from saved file:")
    metrics = calculate_metrics_from_file(args.output_file)
    print("-" * 30)
    print(f"Processed:       {metrics['processed_count']}")
    print(f"Micro F1:        {metrics['micro_f1']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()