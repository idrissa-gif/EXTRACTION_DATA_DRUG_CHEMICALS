import pandas as pd
import json
import random

# --- CONFIGURATION ---
CSV_FILE = "parsed_results.csv"  # Your file with PMID 00000
OUTPUT_FILE = "train_mistral_multi.jsonl"
SAMPLES_TO_GENERATE = 30000  # How many training examples to create

# Templates that accept multiple terms
MULTI_TEMPLATES = [
    "The effects of {t1} and {t2} were compared.",
    "Patients were treated with a combination of {t1}, {t2}, and {t3}.",
    "We analyzed the interaction between {t1} and {t2}.",
    "Resistance to {t1} was observed, but {t2} remained effective.",
    "The presence of {t1} and {t2} in the sample was confirmed.",
    "Both {t1} and {t2} are inhibitors of this pathway.",
    "Synthesis of {t1} followed by derivatization with {t2}.",
    "Exposure to {t1} caused toxicity, unlike {t2}.",
    "We measured levels of {t1}, {t2}, and {t3} in plasma.",
    "Primary treatment involved {t1} whereas {t2} was used for maintenance."
]

SINGLE_TEMPLATES = [
    "The patient was treated with {t1} daily.",
    "We observed that {t1} inhibited the growth.",
    "The molecular structure of {t1} was analyzed.",
    "{t1} is typically classified under this category."
]

def create_multi_entity_dataset():
    print(f"Reading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE, sep=';', dtype=str)
        # Filter for dictionary data
        df = df[df['PMID'] == '00000']

        # Create a list of all available (Term, ID) pairs
        valid_data = []
        for _, row in df.iterrows():
            if pd.notna(row['Term']) and pd.notna(row['Concept ID']):
                valid_data.append( (row['Term'], row['Concept ID']) )

        print(f"Loaded {len(valid_data)} unique terms.")

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Generating {SAMPLES_TO_GENERATE} synthetic multi-entity examples...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for _ in range(SAMPLES_TO_GENERATE):
            # Decide: Single (30%), Double (40%), or Triple (30%) entity sentence
            count = random.choices([1, 2, 3], weights=[0.3, 0.4, 0.3])[0]

            # Pick random terms
            selection = random.sample(valid_data, count)

            entities = []
            terms = []

            for term, cid in selection:
                entities.append({"term": term, "id": cid})
                terms.append(term)

            # Select Template
            if count == 1:
                template = random.choice(SINGLE_TEMPLATES)
                text_input = template.format(t1=terms[0])
            elif count == 2:
                template = random.choice(MULTI_TEMPLATES)
                # Ensure template expects 2 or more, handle strictly
                # Simple fallback formatting for safety
                text_input = f"The interaction of {terms[0]} and {terms[1]} was studied."
            else: # count == 3
                text_input = f"We compared {terms[0]}, {terms[1]}, and {terms[2]} in this study."

            # Mistral Format
            json_structure = {
                "instruction": "Identify all chemical entities and their Concept IDs (ChEBI or ATC).",
                "input": text_input,
                "output": json.dumps(entities)
            }
            f.write(json.dumps(json_structure) + "\n")

    print(f"Success! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_multi_entity_dataset()