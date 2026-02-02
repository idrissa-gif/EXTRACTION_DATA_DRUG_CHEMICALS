#!/usr/bin/env python3
import argparse
import json

def split_bio_file(input_file):
    sentences = []
    tokens, tags = [], []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or len(line.split("\t")) < 2:
                continue
            if "\t" in line:
                token, tag = line.split("\t")
            else:
                token, tag = line.split()

            tokens.append(token)
            tags.append(tag)

            if token == ".":
                sentences.append({"tokens": tokens, "tags": tags})
                tokens, tags = [], []

    if tokens:
        sentences.append({"tokens": tokens, "tags": tags})

    return sentences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bio_file", required=True, help="Input BIO file (TSV format)")
    parser.add_argument("--out_json", default="sentences.json", help="Output JSON file")
    args = parser.parse_args()

    sentences = split_bio_file(args.bio_file)

    with open(args.out_json, "w", encoding="utf-8") as out:
        json.dump(sentences, out, indent=2)

    print(f"✅ Saved {len(sentences)} sentences to {args.out_json}")

if __name__ == "__main__":
    main()
