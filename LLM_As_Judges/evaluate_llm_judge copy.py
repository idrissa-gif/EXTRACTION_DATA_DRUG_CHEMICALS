import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification
from sklearn.metrics import classification_report

def extract_tags_from_response(response):
    response = response.strip()
    if response.startswith("completion"):
        response = response.split("completion:" )[-1]
    tags = response.strip().split()
    return tags

def evaluate_llm(model, tokenizer, test_jsonl, max_len):
    gold_labels = []
    pred_labels = []

    with open(test_jsonl, "r") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            prompt = item["prompt"]
            gold = item["completion"].strip().split()

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)[0]
            decoded = tokenizer.decode(output_ids, skip_special_tokens=True)

            if "completion:" in decoded:
                prediction = decoded.split("completion:")[-1].strip()
            else:
                prediction = decoded[len(prompt):].strip()

            pred = prediction.split()

            if len(pred) > len(gold):
                pred = pred[:len(gold)]
            elif len(pred) < len(gold):
                pred += ["O"] * (len(gold) - len(pred))

            gold_labels.extend(gold)
            pred_labels.extend(pred)

            if i < 3:
                print(f"--- Example {i + 1} ---")
                print("Prompt:", prompt)
                print("Gold:", gold)
                print("Pred:", pred)
                print()

    print("\n=== LLM Classification Report ===")
    print(classification_report(gold_labels, pred_labels, digits=4))

def evaluate_biobert(model_name, test_jsonl, max_len):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name).eval()
    if torch.cuda.is_available():
        model.cuda()

    gold_labels = []
    pred_labels = []

    with open(test_jsonl, "r") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            sentence = item["prompt"].split("Sentence:")[-1].split("\n")[0].strip()
            gold = item["completion"].strip().split()

            tokens = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_len, is_split_into_words=False)
            if torch.cuda.is_available():
                tokens = {k: v.cuda() for k, v in tokens.items()}

            with torch.no_grad():
                outputs = model(**tokens)
                predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

            word_ids = tokens.word_ids(batch_index=0)
            preds = []
            for idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if idx == 0 or word_id != word_ids[idx - 1]:
                    preds.append(model.config.id2label[predictions[idx]])

            if len(preds) > len(gold):
                preds = preds[:len(gold)]
            elif len(preds) < len(gold):
                preds += ["O"] * (len(gold) - len(preds))

            gold_labels.extend(gold)
            pred_labels.extend(preds)

            if i < 3:
                print(f"--- BioBERT Example {i + 1} ---")
                print("Sentence:", sentence)
                print("Gold:", gold)
                print("Pred:", preds)
                print()

    print("\n=== BioBERT Classification Report ===")
    print(classification_report(gold_labels, pred_labels, digits=4))

def evaluate_from_json_fields(json_file, llm_key="tags", biobert_key="biobert_tags", gold_key="tags"):
    with open(json_file) as f:
        data = json.load(f)

    gold = []
    llm_preds = []
    biobert_preds = []

    for i, ex in enumerate(data):
        gold_tags = ex[gold_key]
        llm_tags = ex[llm_key]
        bio_tags = ex[biobert_key]

        assert len(gold_tags) == len(llm_tags) == len(bio_tags), f"Length mismatch in example {i}"

        gold.extend(gold_tags)
        llm_preds.extend(llm_tags)
        biobert_preds.extend(bio_tags)

        if i < 3:
            print(f"--- JSON Field Example {i+1} ---")
            print("Tokens:", ex["tokens"])
            print("Gold:", gold_tags)
            print("LLM:", llm_tags)
            print("BioBERT:", bio_tags)
            print()

    print("\n=== LLM Tags Classification Report ===")
    print(classification_report(gold, llm_preds, digits=4))

    print("\n=== BioBERT Tags Classification Report ===")
    print(classification_report(gold, biobert_preds, digits=4))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="Path to the fine-tuned LLM model directory")
    parser.add_argument("--biobert", help="BioBERT model name, e.g., dslim/bert-base-NER")
    parser.add_argument("--test_jsonl", required=False, help="Path to test JSONL file")
    parser.add_argument("--structured_json", help="Optional structured JSON (with gold/llm/biobert fields)")
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()

    if args.structured_json:
        evaluate_from_json_fields(args.structured_json)
        return

    if args.model_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_dir).eval().cuda() if torch.cuda.is_available() else AutoModelForCausalLM.from_pretrained(args.model_dir).eval()
        evaluate_llm(model, tokenizer, args.test_jsonl, args.max_len)

    if args.biobert:
        evaluate_biobert(args.biobert, args.test_jsonl, args.max_len)

if __name__ == "__main__":
    main()
