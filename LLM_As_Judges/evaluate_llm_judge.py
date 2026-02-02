#!/usr/bin/env python3
import argparse
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report, f1_score

VALID_TAGS = {"O", "B", "I"}

# ---------- Utilities ----------
def clean_tags(raw_tags):
    return [t for t in raw_tags if t in VALID_TAGS]

def normalize_tok(s):
    return s.lower()

def sliding_find_spans(tokens, entity_tokens):
    """Return start indices where entity_tokens match tokens (case-insensitive)."""
    spans = []
    T = list(map(normalize_tok, tokens))
    E = list(map(normalize_tok, entity_tokens))
    if not E:
        return spans
    for i in range(0, len(T) - len(E) + 1):
        if T[i:i+len(E)] == E:
            spans.append(i)
    return spans

def entities_to_bio(tokens, entities):
    """
    entities: list[str] with surface forms as they appear in the sentence.
    Converts predicted entities to BIO tags aligned with tokens.
    Non-overlapping preference: first match wins, later overlaps are skipped.
    """
    tags = ["O"] * len(tokens)
    for ent in entities:
        ent_toks = ent.split()
        starts = sliding_find_spans(tokens, ent_toks)
        for s in starts:
            L = len(ent_toks)
            # place only if free
            if all(tags[s+j] == "O" for j in range(L)):
                tags[s] = "B"
                for j in range(1, L):
                    tags[s+j] = "I"
    return tags

def bio_to_spans(tags):
    """Turn BIO tags into list of (start, end_exclusive) spans."""
    spans = []
    i = 0
    while i < len(tags):
        if tags[i] == "B":
            j = i + 1
            while j < len(tags) and tags[j] == "I":
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans

def span_f1(gold_tags, pred_tags):
    """Strict span-level metrics."""
    gs = set(bio_to_spans(gold_tags))
    ps = set(bio_to_spans(pred_tags))
    tp = len(gs & ps)
    fp = len(ps - gs)
    fn = len(gs - ps)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return prec, rec, f1

def build_entities_prompt(words, sentence, use_biobert, biobert_tags):
    base = [
        f"<start> {sentence} <end>",
        f"Tokens: {' '.join(words)}",
        "Task: Extract CHEMICAL entity mentions from this sentence.",
        "Return ONLY a JSON array of strings (each string must be an entity mention exactly as it appears in the sentence).",
        'Example: ["acetylsalicylic acid", "NaCl"]'
    ]
    if use_biobert:
        base.append(f"BioBERT suggests: {biobert_tags}")
    # No explanations before or after the JSON.
    return "\n".join(base) + "\n"

_JSON_OPEN_RE = re.compile(r"\[\s*(?:\"|{)", flags=re.S)

def parse_entities_json(decoded_text):
    """Extract first top-level JSON array from text and return list[str]."""
    m = _JSON_OPEN_RE.search(decoded_text)
    if not m:
        return []
    start = m.start()
    depth = 0
    end = None
    for i, ch in enumerate(decoded_text[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return []
    try:
        arr = json.loads(decoded_text[start:end])
    except Exception:
        return []
    out = []
    for x in arr:
        if isinstance(x, str):
            out.append(x.strip())
        elif isinstance(x, dict) and isinstance(x.get("text"), str):
            out.append(x["text"].strip())
    return [e for e in out if e]

# ---------- Generation ----------
def generate_llm_predictions(model, tokenizer, data, max_len, debug=False, use_biobert=True, output_mode="entities_json"):
    """
    output_mode: "entities_json" (Option 2) or "bio_tags" (legacy).
    """
    for ex in data:
        words = ex["tokens"]
        n = len(words)
        sentence = " ".join(words)
        biobert_tags = " ".join(ex.get("biobert_tags", ["O"] * n))

        if output_mode == "entities_json":
            prompt = build_entities_prompt(words, sentence, use_biobert, biobert_tags)
        else:
            # Legacy BIO-output path (kept for completeness)
            prompt = (
                f"<start> {sentence} <end>\n"
                f"Tokens: {' '.join(words)}\n"
                "Task: Tag only CHEMICAL entities using BIO (B=begin, I=inside, O=other).\n"
                + (f"BioBERT suggests: {biobert_tags}\n" if use_biobert else "")
                + f"Total tokens: {n}\n"
                f"Output EXACTLY {n} tags, space-separated, using only B I O. No explanations.\n"
                f"Tags:"
            )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
        if torch.cuda.is_available():
            model.cuda()
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=(256 if output_mode == "entities_json" else n*2),
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )[0]

        full_decoded = tokenizer.decode(out_ids, skip_special_tokens=False)
        decoded = full_decoded[len(prompt):].strip() if full_decoded.startswith(prompt) else full_decoded

        if output_mode == "entities_json":
            entities = parse_entities_json(decoded)
            tags = entities_to_bio(words, entities)
            ex["llm_entities"] = entities
            ex["llm_tags"] = tags
            if debug:
                print("--- DEBUG (entities_json) ---")
                print("Prompt:", prompt)
                print("Decoded:", decoded)
                print("Entities:", entities)
                print("Pred tags:", tags)
                print()
        else:
            # Legacy BIO extraction (unchanged)
            tags = []
            if "Tags:" in decoded:
                tags_section = decoded.split("Tags:", 1)[1]
                if "<end>" in tags_section:
                    tags_section = tags_section.split("<end>")[0]
                tags = re.findall(r"\b(?:B|I|O)\b", tags_section)
                tags = clean_tags(tags)
            else:
                decoded_before_end = decoded.split("<end>")[0].strip() if "<end>" in decoded else decoded
                tags = re.findall(r"\b(?:B|I|O)\b", decoded_before_end)
                tags = clean_tags(tags)
            if len(tags) < n:
                tags += ["O"] * (n - len(tags))
            else:
                tags = tags[:n]
            ex["llm_tags"] = tags

    return data

# ---------- Evaluation ----------
def evaluate_generated_predictions(data, output, gold_key="tags", pred_key="llm_tags", biobert_key="biobert_tags", limit=100):
    gold, pred, biobert = [], [], []
    span_metrics = []

    for i, ex in enumerate(data):
        gt = ex.get(gold_key, [])
        pt = ex.get(pred_key, ["O"] * len(gt))[: len(gt)]
        bt = ex.get(biobert_key, ["O"] * len(gt))[: len(gt)]
        gold.extend(gt)
        pred.extend(pt)
        biobert.extend(bt)

        # Span-level metrics per example
        P, R, F = span_f1(gt, pt)
        span_metrics.append((P, R, F))

        if i < 3:
            print(f"Example {i+1}")
            print("Tokens:   ", ex["tokens"])
            if "llm_entities" in ex:
                print("Entities: ", ex["llm_entities"])
            print("Gold:     ", gt)
            print("LLM pred: ", pt)
            print("BioBERT:  ", bt, "\n")

    print("\n=== LLM Classification Report (token-level) ===")
    report_str = classification_report(gold, pred, labels=["B","I","O"], digits=4)
    print(report_str)

    micro_f1 = f1_score(gold, pred, average='micro')
    print(f"\nOverall Micro F1 Score (token-level): {micro_f1:.4f}")

    # Span-level macro averages
    p_avg = sum(p for p,_,_ in span_metrics) / len(span_metrics) if span_metrics else 0.0
    r_avg = sum(r for _,r,_ in span_metrics) / len(span_metrics) if span_metrics else 0.0
    f_avg = sum(f for *_,f in span_metrics) / len(span_metrics) if span_metrics else 0.0
    print(f"\n=== Span-level (entity) Macro Averages ===")
    print(f"Precision: {p_avg:.4f}  Recall: {r_avg:.4f}  F1: {f_avg:.4f}")

    report_dict = classification_report(gold, pred, labels=["B","I","O"], digits=4, output_dict=True)
    report_dict['overall_micro_f1'] = micro_f1
    report_dict['span_level_macro'] = {"precision": p_avg, "recall": r_avg, "f1": f_avg}

    with open(output, 'w') as f:
        json.dump(report_dict, f, indent=4)

    print(f"\nDetailed results saved to: {output}")

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_biobert", action="store_true")
    parser.add_argument("--output", default="output.json")
    parser.add_argument("--output_mode", choices=["entities_json", "bio_tags"], default="entities_json",
                        help="Use 'entities_json' for Option 2 (LLM returns entities, then convert to BIO).")
    args = parser.parse_args()

    data = json.load(open(args.test_jsonl))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    model = AutoModelForCausalLM.from_pretrained(args.model_dir).eval()

    data = generate_llm_predictions(
        model,
        tokenizer,
        data,
        args.max_len,
        debug=args.debug,
        use_biobert=not args.no_biobert,
        output_mode=args.output_mode,
    )
    evaluate_generated_predictions(data, args.output)
