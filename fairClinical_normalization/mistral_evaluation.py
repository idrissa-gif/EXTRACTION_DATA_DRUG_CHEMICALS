import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import re
import requests
import time
import os

ADAPTER_PATH = "./mistral-chem-v2-finetuned"
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
TEST_DATA_FILE = "test_set_with_text.json"

ID_CACHE = {}

def clean_term(term):
    """
    Cleans extracted terms by removing HTML tags and extra whitespace.
    Example: "CaCO<sub>3</sub>" -> "CaCO3"
    """

    clean = re.sub(r'<[^>]+>', '', term)
    clean = " ".join(clean.split())

    return clean

def get_chebi_id_from_api(term):
    """
    Queries the EBI OLS API to find the strict ChEBI ID.
    Handles the conversion from 'CHEBI_123' to 'CHEBI:123'.
    """
    term_clean = clean_term(term.lower().strip())

    # Check cache first
    if term_clean in ID_CACHE:
        return ID_CACHE[term_clean]

    try:
        url = "https://www.ebi.ac.uk/ols/api/search"
        params = {
            'q': term_clean,
            'ontology': 'chebi',
            'exact': False, 
            'rows': 1
        }
        # Add a timeout to prevent hanging
        response = requests.get(url, params=params, timeout=5)

        if response.status_code == 200:
            data = response.json()
            if data['response']['docs']:
                # The API returns IDs like "CHEBI_15365"
                raw_id = data['response']['docs'][0]['short_form']

                # FIX: Convert underscore to colon for standard format
                fixed_id = raw_id.replace("_", ":")

                ID_CACHE[term_clean] = fixed_id
                return fixed_id
    except Exception as e:
        # Silently fail on connection errors to keep the loop running
        pass

    ID_CACHE[term_clean] = None
    return None

def load_model():
    print(f"Loading base model: {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loading adapter from: {ADAPTER_PATH}...")
    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"Adapter not found at {ADAPTER_PATH}")

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return tokenizer, model

def robust_parse_output(output_text):
    """
    Robustly parses the JSON output.
    Finds the last JSON-like list pattern in the text.
    """
    try:
        # Regex to find list patterns like ["item1", "item2"]
        # re.DOTALL allows matching across newlines
        matches = re.findall(r'\[.*?\]', output_text, re.DOTALL)
        if matches:
            # We take the LAST match, as the model might repeat the prompt examples first
            last_match = matches[-1]

            # Fix strict JSON requirement for double quotes
            last_match = last_match.replace("'", '"')

            return json.loads(last_match)
    except:
        pass
    return []

def run_evaluation():
    if not os.path.exists(TEST_DATA_FILE):
        print(f"Error: {TEST_DATA_FILE} not found. Please run prepare_test_data.py first.")
        return

    tokenizer, model = load_model()

    with open(TEST_DATA_FILE, 'r') as f:
        test_data = json.load(f)

    print(f"\n{'='*20} FINAL HYBRID EVALUATION {'='*20}")
    print("Extracting with Mistral -> Normalizing with EBI API")

    total_matches = 0
    total_targets = 0

    for i, case in enumerate(test_data):
        # Filter Expected IDs: We only evaluate on ChEBI IDs
        expected_ids = set([e['id'] for e in case['expected_entities'] if "CHEBI" in e['id']])

        # Skip cases with no ChEBI targets (e.g. only ATC codes)
        if not expected_ids: continue

        print(f"\nCase {i+1}:")

        # 1. Extraction (Mistral)
        # We enforce a strict JSON output format in the prompt
        prompt = f"<s>[INST] Extract the chemical entities from the text below. Return ONLY a JSON list of strings.\nInput: {case['text']} [/INST] Output:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # IMPORTANT: Decode ONLY the generated tokens, slicing off the prompt
        output_tokens = outputs[0][input_length:]
        output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)

        extracted_terms = robust_parse_output(output_text)
        print(f"  LLM Found: {extracted_terms}")

        # 2. Normalization (API Lookup)
        mapped_ids = set()
        for term in extracted_terms:
            # Handle if model outputs a dict instead of a string
            if isinstance(term, dict):
                term = term.get('term', '')

            # Get ID from API
            api_id = get_chebi_id_from_api(term)
            if api_id:
                mapped_ids.add(api_id)

            # Brief sleep to be polite to the public API
            time.sleep(0.05)

        # 3. Scoring
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