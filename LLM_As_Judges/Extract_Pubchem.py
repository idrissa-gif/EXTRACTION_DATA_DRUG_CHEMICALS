#!/usr/bin/env python3
import json
import random
import argparse
from collections import defaultdict

def load_sentences(path):
    """
    Read a BIO‐tagged TSV (token<TAB>tag, blank lines between sentences),
    returning a list of (tokens, tags) for each sentence.
    Skips any malformed lines.
    """
    sents, toks, tgs = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if toks:
                    sents.append((toks, tgs))
                    toks, tgs = [], []
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                # skip bad line
                continue
            tok, tag = parts
            toks.append(tok)
            tgs.append(tag)
    # final sentence
    if toks:
        sents.append((toks, tgs))
    return sents

def extract_spans(tokens, tags, positive=True):
    """
    If positive=True: extract contiguous B/I runs (chemical entities).
    If positive=False: extract contiguous O runs (non-chemical spans).
    Returns a set of span‐strings.
    """
    spans = set()
    cur = []
    for tok, tg in zip(tokens, tags):
        keep = (tg in ("B", "I")) if positive else (tg == "O")
        if keep:
            cur.append(tok)
        else:
            if cur:
                spans.add(" ".join(cur))
                cur = []
    if cur:
        spans.add(" ".join(cur))
    return spans

def main():
    parser = argparse.ArgumentParser(
        description="Generate JSONL for LLM judge from BIO dataset"
    )
    parser.add_argument(
        "--train_tsv",
        required=True,
        help="Path to combined_chem_train.tsv (token<TAB>tag)"
    )
    parser.add_argument(
        "--max_pos", type=int, default=50000,
        help="Max positive examples to sample"
    )
    parser.add_argument(
        "--max_neg", type=int, default=50000,
        help="Max negative examples to sample"
    )
    args = parser.parse_args()

    # 1) Load sentences
    sents = load_sentences(args.train_tsv)
    print(f"Loaded {len(sents)} sentences from {args.train_tsv}")

    # 2) Extract positive (B/I) and negative (O) spans
    pos_spans = set()
    neg_spans = set()
    for toks, tgs in sents:
        pos_spans |= extract_spans(toks, tgs, positive=True)
        neg_spans |= extract_spans(toks, tgs, positive=False)

    print(f"Found {len(pos_spans):,} positive spans and {len(neg_spans):,} negative spans")

    # 3) Sample
    pos_list = pos_spans
    neg_list = neg_spans

    # 4) Write positives JSONL
    with open("chem_positives_ner.jsonl", "w", encoding="utf-8") as fout_pos:
        for span in pos_list:
            prompt = f'Is "{span}" a biomedical chemical? Answer yes or no.\n\nAnswer:'
            fout_pos.write(json.dumps({
                "prompt": prompt,
                "completion": "yes\n"
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pos_list)} positives → chem_positives_ner.jsonl")

    # 5) Write negatives JSONL
    with open("chem_negatives_ner.jsonl", "w", encoding="utf-8") as fout_neg:
        for span in neg_list:
            prompt = f'Is "{span}" a biomedical chemical? Answer yes or no.\n\nAnswer:'
            fout_neg.write(json.dumps({
                "prompt": prompt,
                "completion": "no\n"
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(neg_list)} negatives → chem_negatives_ner.jsonl")

    # 6) Instructions to combine
    print("\nCombine for fine‐tuning:")
    print("  cat chem_positives_ner.jsonl chem_negatives_ner.jsonl > chem_judge.jsonl")

if __name__ == "__main__":
    main()
