import json
import sys
import os 

INPUT_FILE = ''
OUTPUT_FILE = 'Filtered'
ALLOWED_TYPES = ["GeneOrGeneProduct", "ChemicalEntity"]

def Drug_Chemical_Extraction(data, allowed_types):
    for document in data.get("documents", []):
        valid_ids = set()

        # === Filter annotations in each passage ===
        for passage in document.get("passages", []):
            original_annotations = passage.get("annotations", [])
            filtered_annotations = [
                ann for ann in original_annotations
                if ann['infons'].get("type") in allowed_types
            ]
            # Keep track of valid entity identifiers
            for ann in filtered_annotations:
                valid_ids.add(ann['infons'].get("identifier"))
            passage["annotations"] = filtered_annotations

        # === Filter relations: only keep those where both entities are in the valid_ids set ===
        if "relations" in document:
            filtered_relations = [
                rel for rel in document["relations"]
                if rel["infons"].get("entity1") in valid_ids and
                   rel["infons"].get("entity2") in valid_ids
            ]
            document["relations"] = filtered_relations

    return data

def main():
    if len(sys.argv) < 2 :
        print("Use the format file.py <arg1> e.g path.json")
        sys.exit(1)
    INPUT_FILE = sys.argv[1]

    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    filtered_data = Drug_Chemical_Extraction(data, ALLOWED_TYPES)

    input_file = os.path.basename(INPUT_FILE)
    file_prefix = input_file.split(".")[0]

    with open(OUTPUT_FILE + file_prefix + ".JSON", 'w', encoding='utf-8') as outfile:
        json.dump(filtered_data, outfile, indent=2)

    print(f"Filtered data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
