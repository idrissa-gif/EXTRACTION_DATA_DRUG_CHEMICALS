import pandas as pd
import json
import random

# --- CONFIGURATION ---
CSV_FILE = "parsed_results.csv"  # The file with PMID 00000
OUTPUT_FILE = "train_mistral_synthetic.jsonl"

# --- TEMPLATES ---
# We use these to simulate real sentences. {term} will be replaced by your chemical.
TEMPLATES = [
    "The patient was treated with {term} daily.",
    "We observed that {term} inhibited the growth of the culture.",
    "The synthesis of {term} resulted in a high yield.",
    "Comparison of {term} with other compounds showed distinct profiles.",
    "Adverse effects included nausea after taking {term}.",
    "The concentration of {term} in the plasma was measured.",
    "Resistance to {term} is a growing concern.",
    "The molecular structure of {term} was analyzed using NMR.",
    "{term} is typically classified under this category.",
    "Evaluation of the efficacy of {term} in clinical trials.",
    "However, {term} demonstrated poor solubility.",
    "Primary outcome was the response to {term} therapy."
]

def generate_synthetic_data():
    # 1. Load Data
    df = pd.read_csv(CSV_FILE, sep=';', dtype=str)
    
    # 2. Filter for Dictionary Data (PMID 00000)
    # The user mentioned training data has PMID 00000
    df_train = df[df['PMID'] == '00000']
    
    print(f"Found {len(df_train)} dictionary entries. Generating synthetic sentences...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for _, row in df_train.iterrows():
            term = row['Term']
            concept_id = row['Concept ID']
            
            # Skip invalid rows
            if not term or not concept_id:
                continue

            # 3. Create Synthetic Input
            # Pick a random template to wrap the term
            template = random.choice(TEMPLATES)
            synthetic_sentence = template.format(term=term)
            
            # 4. Create the Expected Output
            # The model needs to find the term INSIDE the synthetic sentence
            entities = [{
                "term": term,
                "id": concept_id
            }]
            
            # 5. Mistral JSON Structure
            json_structure = {
                "instruction": "Identify all chemical entities and their Concept IDs (ChEBI or ATC).",
                "input": synthetic_sentence,
                "output": json.dumps(entities)
            }
            
            f.write(json.dumps(json_structure) + "\n")

    print(f"Done! Generated {len(df_train)} synthetic examples in {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_synthetic_data()