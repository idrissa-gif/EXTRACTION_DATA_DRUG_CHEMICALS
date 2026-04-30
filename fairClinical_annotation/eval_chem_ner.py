import argparse
import json
import torch
import re
import numpy as np
from tqdm import tqdm
from collections import Counter
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# -----------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS (Must match training)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = "You are a precise chemical NER assistant. You ONLY output a JSON array of chemical names."
TEMPLATE = """<s>[SYSTEM]\n{system}\n[/SYSTEM]\n[USER]\n{instruction}\n\nText: {text}\n[/USER]\n[ASSISTANT]\n"""

def format_prompt(instruction, text):
    return TEMPLATE.format(system=SYSTEM_PROMPT, instruction=instruction, text=text)

def extract_json_list(text):
    """
    Robust parsing of model output to find the JSON list.
    """
    try:
        # Attempt 1: Direct JSON parse
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        # Attempt 2: Find the first '[' and last ']'
        # This handles cases where the model adds conversational filler before/after
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
    except:
        pass

    # Fallback: Return empty list if parsing fails
    return []

def calculate_metrics(references, predictions):
    """
    Calculates Precision, Recall, F1 (Micro/Macro) and Exact Match Accuracy.
    Uses Multi-set (Counter) matching to account for duplicate entities.
    """
    # Micro-averaged counters
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Macro-averaged lists
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

    exact_matches = 0

    for true_list, pred_list in zip(references, predictions):
        # Normalize strings (strip whitespace)
        t_clean = [str(x).strip() for x in true_list]
        p_clean = [str(x).strip() for x in pred_list]

        # Check Exact Match (Order independent comparison of lists)
        if sorted(t_clean) == sorted(p_clean):
            exact_matches += 1

        # Convert to counters to handle duplicates (e.g. "lidocaine" appearing twice)
        t_counter = Counter(t_clean)
        p_counter = Counter(p_clean)

        # True Positives: Intersection of counts
        tp_counter = t_counter & p_counter
        tp = sum(tp_counter.values())

        # False Positives: Items in Pred but not in True
        fp_counter = p_counter - t_counter
        fp = sum(fp_counter.values())

        # False Negatives: Items in True but not in Pred
        fn_counter = t_counter - p_counter
        fn = sum(fn_counter.values())

        # Accumulate for Micro
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Calculate Macro (per sample)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        macro_precisions.append(precision)
        macro_recalls.append(recall)
        macro_f1s.append(f1)

    # Compute Micro Metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    results = {
        "accuracy_exact_match": exact_matches / len(references),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": np.mean(macro_precisions),
        "macro_recall": np.mean(macro_recalls),
        "macro_f1": np.mean(macro_f1s)
    }
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="instruct_data/test.jsonl")
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct")
    parser.add_argument("--adapter_path", default="mistral-chem-instruct-lora/final") # Path to your saved model
    parser.add_argument("--output_file", default="test_predictions.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Mistral usually prefers left padding for generation

    # Load Base Model (4-bit)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Load LoRA Adapter
    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    # Load Data
    print(f"Loading test data: {args.test_file}")
    with open(args.test_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    predictions = []
    references = []
    results_log = []

    print("Starting inference...")
    for entry in tqdm(data):
        # Prepare Input
        prompt = format_prompt(entry["instruction"], entry["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,      # Greedy decoding for deterministic results
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode Output
        # We slice [inputs.input_ids.shape[1]:] to only get the newly generated text
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Parse Prediction
        pred_list = extract_json_list(generated_text)
        true_list = entry["output"]

        predictions.append(pred_list)
        references.append(true_list)

        # Log individual result
        results_log.append({
            "input": entry["input"][:100] + "...",
            "ground_truth": true_list,
            "prediction": pred_list,
            "raw_output": generated_text
        })

    # Calculate Metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(references, predictions)

    # Print Results
    print("-" * 30)
    print("EVALUATION RESULTS")
    print("-" * 30)
    print(f"Exact Match Accuracy: {metrics['accuracy_exact_match']:.4f}")
    print(f"Micro F1:             {metrics['micro_f1']:.4f}")
    print(f"Micro Precision:      {metrics['micro_precision']:.4f}")
    print(f"Micro Recall:         {metrics['micro_recall']:.4f}")
    print("-" * 30)
    print(f"Macro F1:             {metrics['macro_f1']:.4f}")
    print(f"Macro Precision:      {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:         {metrics['macro_recall']:.4f}")
    print("-" * 30)

    # Save detailed results
    with open(args.output_file, "w") as f:
        json.dump({"metrics": metrics, "predictions": results_log}, f, indent=2)
    print(f"Detailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()