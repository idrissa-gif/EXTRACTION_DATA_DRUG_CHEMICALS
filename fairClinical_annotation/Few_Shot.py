import os
import json
import argparse
import torch
import random
import re
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & PROMPT TEMPLATES
# -----------------------------------------------------------------------------
# Standard Mistral format is safest for Base models
# We chain examples as history: [INST] Input [/INST] Output </s> [INST] Input...
PROMPT_TEMPLATE = "[INST] {instruction}\n\nText: {text} [/INST] {output}"
TARGET_TEMPLATE = "[INST] {instruction}\n\nText: {text} [/INST]"

# Instruction used in training data
DEFAULT_INSTRUCTION = "Extract all chemical names from the input text. Return a JSON array of strings, with no explanations."

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# -----------------------------------------------------------------------------
# 2. SELECTION STRATEGY (CRITICAL FOR GOOD RESULTS)
# -----------------------------------------------------------------------------
class ExampleSelector:
    def __init__(self, train_data, use_similarity=True):
        self.train_data = train_data
        self.use_similarity = use_similarity
        self.vectorizer = None
        self.train_vectors = None

        if use_similarity and len(train_data) > 0:
            print("Building TF-IDF index for Similarity Selection (Better results)...")
            corpus = [ex["input"] for ex in train_data]
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
            self.train_vectors = self.vectorizer.fit_transform(corpus)

    def get_shots(self, target_input, k=3):
        if k == 0:
            return []

        if self.use_similarity and self.vectorizer:
            # Find semantically similar examples
            target_vec = self.vectorizer.transform([target_input])
            scores = cosine_similarity(target_vec, self.train_vectors).flatten()
            # Get top k indices
            top_k_indices = scores.argsort()[-k:][::-1]
            return [self.train_data[i] for i in top_k_indices]
        else:
            # Fallback to Random (Standard baseline)
            return random.sample(self.train_data, k)

def build_prompt(target_ex, shots, tokenizer):
    """
    Constructs the few-shot prompt.
    Manages context window to prevent OOM.
    """
    full_prompt = "<s>"

    # Add shots
    for shot in shots:
        shot_json = json.dumps(shot["output"], ensure_ascii=False)
        full_prompt += PROMPT_TEMPLATE.format(
            instruction=DEFAULT_INSTRUCTION,
            text=shot["input"],
            output=shot_json
        ) + " </s>"

    # Add target
    full_prompt += TARGET_TEMPLATE.format(
        instruction=DEFAULT_INSTRUCTION,
        text=target_ex["input"]
    )

    return full_prompt

# -----------------------------------------------------------------------------
# 3. ROBUST PARSING & METRICS
# -----------------------------------------------------------------------------
def extract_json_list(text):
    text = text.strip()
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            try: return json.loads(match.group(0).replace("'", '"'))
            except: pass
    if '[' in text and ']' not in text: # Handle truncation
        try: return json.loads(text[text.find('['):] + '"]')
        except: pass
    return []

def calculate_metrics(references, predictions):
    total_tp, total_fp, total_fn = 0, 0, 0
    exact_matches = 0
    macro_f1s = []

    for true_list, pred_list in zip(references, predictions):
        t_clean = [str(x).strip() for x in true_list]
        p_clean = [str(x).strip() for x in pred_list]

        if sorted(t_clean) == sorted(p_clean): exact_matches += 1

        t_c = Counter(t_clean); p_c = Counter(p_clean)
        tp = sum((t_c & p_c).values())
        fp = sum((p_c - t_c).values())
        fn = sum((t_c - p_c).values())

        total_tp += tp; total_fp += fp; total_fn += fn

        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*(prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
        macro_f1s.append(f1)

    micro_prec = total_tp / (total_tp+total_fp) if (total_tp+total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp+total_fn) if (total_tp+total_fn) > 0 else 0.0
    micro_f1 = 2*(micro_prec*micro_rec)/(micro_prec+micro_rec) if (micro_prec+micro_rec) > 0 else 0.0

    return {
        "Micro F1": micro_f1,
        "Micro Recall": micro_rec,
        "Micro Precision": micro_prec,
        "Macro F1": np.mean(macro_f1s) if macro_f1s else 0.0,
        "Exact Match": exact_matches / len(references) if references else 0.0
    }

# -----------------------------------------------------------------------------
# 4. MAIN INFERENCE LOOP
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", default="instruct_data/train.jsonl")
    parser.add_argument("--test_jsonl",  default="instruct_data/test.jsonl")
    parser.add_argument("--model_name",  default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--output_file", default="results_fewshot.jsonl")
    parser.add_argument("--shots", type=int, default=3, help="Number of examples (k)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_similarity", action="store_true", help="Use TF-IDF to find best examples (Recommended)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading Model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()

    print("Loading Data...")
    train_data = load_data(args.train_jsonl)
    test_data = load_data(args.test_jsonl)

    # Initialize Selector
    selector = ExampleSelector(train_data, use_similarity=args.use_similarity)

    # Output file handling
    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            for line in f:
                try: processed_ids.add(json.loads(line)['id'])
                except: pass
    print(f"Resuming from {len(processed_ids)} processed items...")

    f_out = open(args.output_file, 'a', encoding='utf-8')

    # Batch Processing
    batch_size = args.batch_size
    all_refs, all_preds = [], []

    # Filter remaining
    remaining_indices = [i for i in range(len(test_data)) if i not in processed_ids]

    for i in tqdm(range(0, len(remaining_indices), batch_size)):
        batch_idxs = remaining_indices[i : i + batch_size]
        batch_items = [test_data[idx] for idx in batch_idxs]

        prompts = []
        for item in batch_items:
            shots = selector.get_shots(item["input"], k=args.shots)
            prompts.append(build_prompt(item, shots, tokenizer))

        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )

            decoded = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

            for j, text in enumerate(decoded):
                idx = batch_idxs[j]
                pred = extract_json_list(text)
                truth = batch_items[j]["output"]

                # Save Result
                res = {"id": idx, "ground_truth": truth, "prediction": pred, "raw": text[:200]}
                f_out.write(json.dumps(res) + "\n")

                all_preds.append(pred)
                all_refs.append(truth)

            f_out.flush()

        except Exception as e:
            print(f"Error in batch: {e}")
            torch.cuda.empty_cache()

    f_out.close()

    # Final Calculation (Read from file to be safe)
    print("\nCalculating Final Metrics...")
    full_refs, full_preds = [], []
    with open(args.output_file, 'r') as f:
        for line in f:
            obj = json.loads(line)
            full_refs.append(obj['ground_truth'])
            full_preds.append(obj['prediction'])

    metrics = calculate_metrics(full_refs, full_preds)
    print("="*40)
    print("PUBLICATION RESULTS")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()