import argparse, json, re

def load_bio(path):
    sents, tags = [], []
    cur_w, cur_t = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_w:
                    sents.append(cur_w); tags.append(cur_t)
                    cur_w, cur_t = [], []
                continue
            parts = line.split()
            if len(parts) < 2:  # skip malformed
                continue
            token, tag = parts[0], parts[-1]
            cur_w.append(token); cur_t.append(tag)
    if cur_w:
        sents.append(cur_w); tags.append(cur_t)
    return sents, tags

def merge_entities(words, labels):
    ents, cur = [], []
    for w, y in zip(words, labels):
        if y == "B":
            if cur: ents.append(" ".join(cur)); cur = []
            cur = [w]
        elif y == "I":
            if cur: cur.append(w)
            else:   cur = [w]   # tolerate stray I
        else:  # O
            if cur: ents.append(" ".join(cur)); cur = []
    if cur: ents.append(" ".join(cur))
    return ents

INSTR = "Extract all chemical names from the input text. Return a JSON array of strings, with no explanations."

def process_split(in_path, out_path):
    sents, tags = load_bio(in_path)
    with open(out_path, "w", encoding="utf-8") as out:
        for w, y in zip(sents, tags):
            text = " ".join(w)
            ents = merge_entities(w, y)
            rec = {"instruction": INSTR, "input": text, "output": ents}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--dev_file", required=True)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--out_dir", default="instruct_data")
    args = ap.parse_args()
    import os; os.makedirs(args.out_dir, exist_ok=True)

    process_split(args.train_file, f"{args.out_dir}/train.jsonl")
    process_split(args.dev_file,   f"{args.out_dir}/dev.jsonl")
    process_split(args.test_file,  f"{args.out_dir}/test.jsonl")
    print("Wrote:", f"{args.out_dir}/train.jsonl", f"{args.out_dir}/dev.jsonl", f"{args.out_dir}/test.jsonl")

if __name__ == "__main__":
    main()
