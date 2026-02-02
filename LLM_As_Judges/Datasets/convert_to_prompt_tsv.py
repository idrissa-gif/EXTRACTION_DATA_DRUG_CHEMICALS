#!/usr/bin/env python3
import argparse
import json
import csv
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input TSV file (columns: sentence_idx, token_idx, token, true_tag, pred_tag)")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--include_biobert_tags", action="store_true", help="Use pred_tag as biobert_tags")
    args = parser.parse_args()

    # Data structure to group by sentence
    sentences = defaultdict(lambda: {"tokens": [], "tags": [], "biobert_tags": []})

    with open(args.input, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sent_id = int(row["sentence_idx"])
            token = row["token"]
            true_tag = row["true_tag"]
            pred_tag = row["pred_tag"]

            sentences[sent_id]["tokens"].append(token)
            sentences[sent_id]["tags"].append(true_tag)
            if args.include_biobert_tags:
                sentences[sent_id]["biobert_tags"].append(pred_tag)

    output_data = []
    for sent_data in sentences.values():
        record = {
            "tokens": sent_data["tokens"],
            "tags": sent_data["tags"]
        }
        if args.include_biobert_tags:
            record["biobert_tags"] = sent_data["biobert_tags"]
        output_data.append(record)

    with open(args.output, "w") as out_f:
        json.dump(output_data, out_f, indent=2)

    print(f"✅ Saved {len(output_data)} sentences to {args.output}")

if __name__ == "__main__":
    main()
