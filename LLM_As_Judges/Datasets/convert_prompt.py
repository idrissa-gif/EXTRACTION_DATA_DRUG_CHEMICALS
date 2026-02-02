#!/usr/bin/env python3
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True,
        help="Input JSON file (a list of examples, each with 'tokens' and 'tags')"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .jsonl file (with 'prompt' and 'completion' fields)"
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(args.output, "w", encoding="utf-8") as out:
        for i, example in enumerate(data, 1):
            tokens = example.get("tokens")
            tags = example.get("tags")

            if tokens is None or tags is None:
                raise ValueError(f"❌ Example {i} missing 'tokens' or 'tags' keys.")

            if len(tokens) != len(tags):
                raise ValueError(f"❌ Example {i} has mismatched lengths: {len(tokens)} tokens vs {len(tags)} tags")

            sentence = " ".join(tokens).replace("\n", " ").strip()
            tag_seq = " ".join(tags).strip()

            record = {
                "prompt": f"Sentence: {sentence}\nPredict the sequence of BIO tags for the sentence:",
                "completion": f" {tag_seq}"  # Note leading space
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(data)} examples to {args.output}")

if __name__ == "__main__":
    main()
