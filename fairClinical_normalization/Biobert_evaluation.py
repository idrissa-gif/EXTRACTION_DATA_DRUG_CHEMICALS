import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
import re
import requests
import time
import os
import string
from difflib import SequenceMatcher

BIOBERT_CHECKPOINT = "Models/combined/final_checkpoint"
TEST_DATA_FILE = "fairClinical_V2/test_set_with_text.json"

ID_CACHE = {}

def clean_term(term):
    clean = re.sub(r'<[^>]+>', '', term)
    clean = " ".join(clean.split())
    return clean

def normalize_text(term):
    t = term.strip().lower()
    t = t.strip(string.punctuation)
    t = re.sub(r"\s+", " ", t)
    return t

def is_valid_term(term):
    t = normalize_text(term)
    if not t:
        return False
    if len(t) == 1:
        return False
    return True

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def score_doc(term_norm, doc):
    label = normalize_text(doc.get("label", ""))
    synonyms = [normalize_text(s) for s in doc.get("synonym", []) if isinstance(s, str)]

    # strongest signals
    if term_norm == label:
        return 1.0
    if term_norm in synonyms:
        return 0.95

    # similarity with label + synonyms
    best = similarity(term_norm, label)
    for s in synonyms:
        best = max(best, similarity(term_norm, s))

    return best

def get_chebi_id_from_api(term):
    term_clean = clean_term(term)
    term_norm = normalize_text(term_clean)

    if term_norm in ID_CACHE:
        return ID_CACHE[term_norm]

    try:
        url = "https://www.ebi.ac.uk/ols/api/search"
        params = {'q': term_norm, 'ontology': 'chebi', 'exact': False, 'rows': 10}
        response = requests.get(url, params=params, timeout=5)

        if response.status_code == 200:
            data = response.json()
            docs = data.get("response", {}).get("docs", [])

            best_score = 0.0
            best_id = None

            for doc in docs:
                s = score_doc(term_norm, doc)
                if s > best_score:
                    best_score = s
                    best_id = doc.get("short_form")

            # require a minimum confidence to accept
            if best_id and best_score >= 0.72:
                fixed_id = best_id.replace("_", ":")
                ID_CACHE[term_norm] = fixed_id
                return fixed_id

    except Exception:
        pass

    ID_CACHE[term_norm] = None
    return None

def load_label_map():
    labels_path = os.path.join(BIOBERT_CHECKPOINT, "labels.json")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.json not found at {labels_path}")

    with open(labels_path, "r") as f:
        labels = json.load(f)

    if isinstance(labels, dict) and "id2label" in labels:
        id2label = {int(k): v for k, v in labels["id2label"].items()}
    elif isinstance(labels, dict):
        id2label = {int(k): v for k, v in labels.items()}
    else:
        raise ValueError("labels.json format not recognized")

    return id2label

def load_biobert():
    if not os.path.exists(BIOBERT_CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found at {BIOBERT_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_CHECKPOINT, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(BIOBERT_CHECKPOINT)
    model.eval()
    return tokenizer, model

def extract_entities_from_text(text, tokenizer, model, id2label, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )

    offset_mapping = enc["offset_mapping"][0].tolist()
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    preds = logits.argmax(dim=-1)[0].tolist()
    labels = [id2label[p] for p in preds]

    word_ids = enc.word_ids(batch_index=0)
    if word_ids is None:
        raise ValueError("Tokenizer must be fast to use word_ids.")

    word_spans = []
    seen = set()
    for i, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)

        start = offset_mapping[i][0]
        end = offset_mapping[i][1]
        j = i + 1
        while j < len(word_ids) and word_ids[j] == word_id:
            end = offset_mapping[j][1]
            j += 1

        word_spans.append({
            "start": start,
            "end": end,
            "label": labels[i],
            "text": text[start:end]
        })

    entities = []
    current = None
    for w in word_spans:
        label = w["label"]
        if label == "B" or label.startswith("B-"):
            if current is not None:
                entities.append(current["text"])
            current = {"start": w["start"], "end": w["end"], "text": w["text"]}
        elif (label == "I" or label.startswith("I-")) and current is not None:
            current["end"] = w["end"]
            current["text"] = text[current["start"]:current["end"]]
        else:
            if current is not None:
                entities.append(current["text"])
                current = None

    if current is not None:
        entities.append(current["text"])

    entities = [clean_term(e) for e in entities]
    entities = [normalize_text(e) for e in entities if is_valid_term(e)]
    # de‑dup, keep order
    return list(dict.fromkeys(entities))

def run_evaluation():
    if not os.path.exists(TEST_DATA_FILE):
        print(f"Error: {TEST_DATA_FILE} not found. Please run prepare_test_data.py first.")
        return

    tokenizer, model = load_biobert()
    id2label = load_label_map()

    with open(TEST_DATA_FILE, 'r') as f:
        test_data = json.load(f)

    print(f"\n{'='*20} FINAL BIOBERT EVALUATION {'='*20}")
    print("Extracting with BioBERT -> Normalizing with EBI API")

    total_matches = 0
    total_targets = 0

    for i, case in enumerate(test_data):
        expected_ids = set([e['id'] for e in case['expected_entities'] if "CHEBI" in e['id']])
        if not expected_ids:
            continue

        print(f"\nCase {i+1}:")

        extracted_terms = extract_entities_from_text(
            case["text"], tokenizer, model, id2label
        )
        print(f"  BioBERT Found: {extracted_terms}")

        mapped_ids = set()
        for term in extracted_terms:
            api_id = get_chebi_id_from_api(term)
            if api_id:
                mapped_ids.add(api_id)
            time.sleep(0.05)

        matches = expected_ids.intersection(mapped_ids)

        if matches:
            print(f"  ✅ MATCH: {len(matches)}/{len(expected_ids)}")
            print(f"     Expected: {sorted(list(expected_ids))}")
            print(f"     Mapped:   {sorted(list(mapped_ids))}")
        else:
            print(f"  ❌ FAIL")
            print(f"     Expected: {sorted(list(expected_ids))}")
            print(f"     Mapped:   {sorted(list(mapped_ids))}")

        total_matches += len(matches)
        total_targets += len(expected_ids)

    print(f"\n{'='*20} SUMMARY {'='*20}")
    if total_targets > 0:
        recall = total_matches / total_targets
        print(f"Global Recall (Strict ChEBI): {total_matches}/{total_targets} ({recall*100:.1f}%)")
    else:
        print("No valid ChEBI targets found in the test set.")

if __name__ == "__main__":
    run_evaluation()
