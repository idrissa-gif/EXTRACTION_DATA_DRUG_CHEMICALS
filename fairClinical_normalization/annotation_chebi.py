import os
import re
import requests
import time
from pathlib import Path

# --- CONFIGURATION ---
INPUT_DIR = "./Biobert"               # Main input folder containing PMC folders
OUTPUT_DIR = "./Biobert_ID"           # Where the new folders will be saved

# Cache to prevent spamming the API for repeated chemicals
ID_CACHE = {}

def clean_term(term):
    """Removes HTML tags and normalizes whitespace."""
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
                # Get the ID and replace underscores with colons (e.g., CHEBI_123 -> CHEBI:123)
                raw_id = docs[0].get('short_form', '')
                return raw_id.replace("_", ":")
    except Exception:
        pass
    return None

def get_concept_id(term):
    """Tries to find a ChEBI ID. If it fails, falls back to ATC."""
    term_clean = clean_term(term.lower().strip())

    # Check cache first
    if term_clean in ID_CACHE:
        return ID_CACHE[term_clean]

    # 1. Try ChEBI
    chebi_id = query_ols_api(term_clean, "chebi")
    if chebi_id:
        ID_CACHE[term_clean] = chebi_id
        time.sleep(0.05) # Polite API sleep
        return chebi_id

    # 2. Try ATC (Fallback)
    atc_id = query_ols_api(term_clean, "atc")
    if atc_id:
        ID_CACHE[term_clean] = atc_id
        time.sleep(0.05)
        return atc_id

    # Not found in either
    ID_CACHE[term_clean] = "NOT_FOUND"
    time.sleep(0.05)
    return "NOT_FOUND"

def process_tsv_file(input_path, output_path):
    """Reads a BIO-tagged TSV, groups entity spans, fetches IDs, and writes a new TSV."""
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    entities = []  # Store tuples of: (start_line_idx, end_line_idx, term_string)
    current_term = []
    start_idx = -1

    # Check if the first line is a header
    has_header = False
    if lines and "Token" in lines[0] and "BIO-Tag" in lines[0]:
        has_header = True

    # Step 1: Parse file and extract full entity spans
    for i, line in enumerate(lines):
        if has_header and i == 0:
            continue

        line_stripped = line.strip()
        if not line_stripped:
            # Sentence break, finalize current entity if it exists
            if start_idx != -1:
                entities.append((start_idx, i - 1, " ".join(current_term)))
                start_idx = -1
                current_term = []
            continue

        # Handle both Tab and Space separated files safely
        parts = line_stripped.split('\t')
        if len(parts) < 2:
            parts = line_stripped.split()
            if len(parts) < 2:
                continue

        token = parts[0]
        tag = parts[-1].strip() # The last item is the tag (B, I, or O)

        if tag == 'B':
            if start_idx != -1: # Finalize previous entity if two B's are back-to-back
                entities.append((start_idx, i - 1, " ".join(current_term)))
            start_idx = i
            current_term = [token]
        elif tag == 'I':
            if start_idx != -1:
                current_term.append(token)
            else:
                # Handle edge case where model predicted 'I' without a starting 'B'
                start_idx = i
                current_term = [token]
        else: # 'O' tag
            if start_idx != -1: # Finalize current entity
                entities.append((start_idx, i - 1, " ".join(current_term)))
                start_idx = -1
                current_term = []

    # Catch any entity left at the very end of the file
    if start_idx != -1:
        entities.append((start_idx, len(lines) - 1, " ".join(current_term)))

    # Step 2: Look up IDs for extracted spans
    line_to_id = {}
    for (start, end, term) in entities:
        concept_id = get_concept_id(term)
        for line_idx in range(start, end + 1):
            line_to_id[line_idx] = concept_id

    # Step 3: Write the new TSV with the ID column added
    with open(output_path, 'w', encoding='utf-8') as f:
        if has_header:
            f.write(lines[0].strip() + "\tConcept_ID\n")

        for i, line in enumerate(lines):
            if has_header and i == 0:
                continue

            line_stripped = line.strip()
            if not line_stripped:
                f.write("\n") # Preserve blank lines between sentences
            else:
                # Append the Concept ID if it's part of an entity, otherwise use '-'
                concept_id = line_to_id.get(i, "-")
                f.write(f"{line_stripped}\t{concept_id}\n")

def main():
    input_dir = Path(INPUT_DIR)

    if not input_dir.exists():
        print(f"Directory {INPUT_DIR} not found.")
        return

    # Iterate through all subdirectories in the Biobert folder
    for folder in input_dir.iterdir():
        if folder.is_dir() and not folder.name.endswith('_ID'):
            # Create the new folder name (e.g., PMC030XXXXX_json_ascii_results_ID)
            new_folder_name = f"{folder.name}_ID"
            output_folder_path = Path(OUTPUT_DIR) / new_folder_name

            # Create the output directory
            output_folder_path.mkdir(parents=True, exist_ok=True)
            print(f"Processing folder: {folder.name} -> {new_folder_name}")

            # Process every TSV (or TXT) file inside the folder
            for data_file in folder.glob("*.*"):
                if data_file.suffix in ['.tsv', '.txt']:
                    output_file_path = output_folder_path / data_file.name
                    process_tsv_file(data_file, output_file_path)

    print("\n✅ All folders and files processed successfully!")

if __name__ == "__main__":
    main()