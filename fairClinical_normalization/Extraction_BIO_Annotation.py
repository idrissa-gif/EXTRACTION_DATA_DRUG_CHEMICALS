import os
import json
import argparse
import time
import psutil
import re
import requests
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['USE_TORCH'] = '1'

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlShutdown,
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# --- API CACHE & FUNCTIONS ---
ID_CACHE = {}

def clean_term(term):
    """Removes HTML tags and normalizes whitespace for API queries."""
    clean = re.sub(r'<[^>]+>', '', term)
    return " ".join(clean.split())

def query_ols_api(term, ontology):
    """Queries the EBI OLS API for a specific ontology (e.g., 'chebi' or 'atc')."""
    url = "https://www.ebi.ac.uk/ols/api/search"
    params = {
        'q': term,
        'ontology': ontology,
        'exact': False,
        'rows': 1
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            docs = response.json().get('response', {}).get('docs', [])
            if docs:
                raw_id = docs[0].get('short_form', '')
                return raw_id.replace("_", ":")
    except Exception:
        pass
    return None

def get_concept_id(term):
    """Tries to find a ChEBI ID. If it fails, falls back to ATC."""
    term_clean = clean_term(term.lower().strip())

    if term_clean in ID_CACHE:
        return ID_CACHE[term_clean]

    # 1. Try ChEBI
    chebi_id = query_ols_api(term_clean, "chebi")
    if chebi_id:
        ID_CACHE[term_clean] = chebi_id
        return chebi_id

    # 2. Try ATC
    atc_id = query_ols_api(term_clean, "atc")
    if atc_id:
        ID_CACHE[term_clean] = atc_id
        return atc_id

    # Not found
    ID_CACHE[term_clean] = "NOT_FOUND"
    return "NOT_FOUND"


# --- NER EXTRACTION & STITCHING ---
def extract_entities_from_pipeline(ner_results, text):
    """
    Stitches subwords (e.g., 'ch', '##lor') and multi-word entities
    back together into full entity strings using character offsets.
    """
    entities = []
    current_ent = None

    for res in ner_results:
        tag = res.get('entity', res.get('entity_group', 'O'))
        token_text = res.get('word', '')

        # Check if this is a subword token (BERT-style ## prefix or other indicators)
        is_subword = token_text.startswith('##') or (
            current_ent and res['start'] == current_ent['end']
        )

        # Skip non-entities
        if tag == 'O':
            if current_ent:
                entities.append(current_ent)
                current_ent = None
            continue

        # Condition 1: Subwords (tokens touch each other exactly without spaces)
        is_contiguous = current_ent and (res['start'] == current_ent['end'])

        # Condition 2: Multi-word entities marked with 'I' (Inside) tag
        is_i_tag = current_ent and (tag.startswith('I') or tag == 'I')

        # Condition 3: Force subword tokens to merge with previous entity
        # This handles cases where ## tokens incorrectly get B- tags
        should_merge = is_contiguous or is_i_tag or (is_subword and current_ent)

        if should_merge:
            # Extend the current entity's boundary
            current_ent['end'] = res['end']
            current_ent['text'] = text[current_ent['start']:res['end']]
        elif is_subword and not current_ent:
            # Orphan subword with no preceding entity - skip it entirely
            # This catches fragments like "##ine" that shouldn't be entities
            continue
        else:
            # Finalize previous entity and start a new one
            if current_ent:
                entities.append(current_ent)
            current_ent = {
                'start': res['start'],
                'end': res['end'],
                'text': text[res['start']:res['end']]
            }

    # Catch the last entity in the sequence
    if current_ent:
        entities.append(current_ent)

    # Post-filter: remove suspiciously short entities (likely subword fragments)
    MIN_ENTITY_LENGTH = 3
    entities = [e for e in entities if len(e['text']) >= MIN_ENTITY_LENGTH]

    return entities


def parse_args():
    parser = argparse.ArgumentParser(description="Run BIO-tag NER on BioC JSON files and append annotations.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--json_dir", type=str, required=True, help="Directory with BioC Json files.")
    return parser.parse_args()


def get_gpu_memory():
    if not GPU_AVAILABLE:
        return None
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()
    return info.used / 1024 ** 2


def main():
    args = parse_args()
    process = psutil.Process(os.getpid())

    json_dir = Path(args.json_dir)
    result_dir = json_dir.parent / f"{json_dir.name}_annotated_json"
    result_dir.mkdir(parents=True, exist_ok=True)

    log_path = result_dir / "usage_log.txt"

    with open(log_path, "w") as log:
        log.write("============================ Resource Usage Log ============================\n")

        t0 = time.time()
        import torch

        # Check if CUDA is available and set device
        device = 0 if torch.cuda.is_available() else -1

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForTokenClassification.from_pretrained(args.model_path)

        # Initialize pipeline
        ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="none", device=device)

        print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
        model_load_time = time.time() - t0

        mem_used = process.memory_info().rss / 1024 ** 2
        gpu_used = get_gpu_memory()

        log.write(f"Model load time: {model_load_time:.2f} s\n")
        log.write(f"Memory after model load: {mem_used:.2f} MB\n")
        if gpu_used is not None:
            log.write(f"GPU memory after model load: {gpu_used:.2f} MB\n")

        for json_file in json_dir.glob("*.json"):
            file_start = time.time()

            # 1. Load the original BioC JSON
            with open(json_file, 'r', encoding="utf-8") as f:
                bioc = json.load(f)

            # 2. Iterate through documents and passages
            for doc in bioc.get('documents', []):
                for passage in doc.get('passages', []):
                    passage_text = passage.get("text", "")
                    if not passage_text:
                        continue

                    # Ensure annotations list exists
                    if "annotations" not in passage:
                        passage["annotations"] = []

                    passage_offset = passage.get("offset", 0)
                    ann_id_counter = len(passage["annotations"]) + 1

                    # Run NER and get raw token results
                    ner_results = ner(passage_text)

                    # Stitch subwords and multi-words into full entities
                    entities = extract_entities_from_pipeline(ner_results, passage_text)

                    # 3. Look up IDs and append to the passage's annotations
                    for ent in entities:
                        concept_id = get_concept_id(ent['text'])

                        annotation = {
                            "id": f"A{ann_id_counter}",
                            "infons": {
                                "type": "Chemical",
                                "identifier": concept_id
                            },
                            "text": ent['text'],
                            "locations": [
                                {
                                    "offset": passage_offset + ent['start'],
                                    "length": ent['end'] - ent['start']
                                }
                            ]
                        }
                        passage["annotations"].append(annotation)
                        ann_id_counter += 1
                        time.sleep(0.05) # Polite API delay

            # 4. Save the modified BioC structure to a new JSON file
            output_file = result_dir / f"{json_file.name}"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(bioc, f, indent=2, ensure_ascii=False)

            file_time = time.time() - file_start
            file_mem = process.memory_info().rss / 1024**2
            file_gpu = get_gpu_memory()

            log.write(f"\nFile: {json_file.name}\n")
            log.write(f"Time: {file_time:.2f} s\n")
            log.write(f"Memory: {file_mem:.2f} MB\n")
            if file_gpu is not None:
                log.write(f"GPU memory: {file_gpu:.2f} MB\n")

            print(f"Annotated {json_file.name} -> {output_file.name}")

if __name__ == "__main__":
    main()