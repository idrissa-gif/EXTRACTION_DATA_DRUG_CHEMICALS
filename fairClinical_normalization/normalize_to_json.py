"""
Normalization step for the FairClinical V2 chemical-NER pipeline.

Reads:
  1) The original BioC JSON files in ``input/PMC*_json_ascii/*.json`` (for
     passage text and character offsets).
  2) The BIO-tagged TSVs produced by ``Extraction_BIO.py`` in
     ``input/PMC*_json_ascii_results/*_annotated.tsv`` (for entity spans).

Rebuilds character offsets by replaying the training-time tokenization
(``tokenize_biomedical_text`` from ``Extraction_BIO.py``) over the passage
text and aligning each TSV token to the first matching substring. This
avoids re-running the NER model while producing output byte-for-byte
compatible with ``Extraction_BIO_Annotation.py``.

Output: one BioC JSON per document in
``input/PMC*_json_ascii_annotated_json/<docid>.json``, where each passage
gains an ``annotations`` list of the form::

    {
      "id": "A1",
      "infons": {"type": "Chemical", "identifier": "CHEBI:12345"},
      "text": "<entity surface form>",
      "locations": [{"offset": <absolute char offset>, "length": <len>}]
    }

ChEBI IDs are tried first, ATC as a fallback; lookups are cached on disk
at ``normalization_cache.json``. Resumable: files whose output already
exists are skipped unless ``--overwrite`` is passed.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests


OLS_URL = "https://www.ebi.ac.uk/ols/api/search"
API_DELAY = 0.05
REQUEST_TIMEOUT = 5
MIN_ENTITY_LENGTH = 3
CACHE_SAVE_EVERY = 20


def clean_term(term: str) -> str:
    """Strip HTML and collapse whitespace for API queries."""
    return " ".join(re.sub(r"<[^>]+>", "", term).split())


def query_ols_api(term: str, ontology: str):
    params = {"q": term, "ontology": ontology, "exact": False, "rows": 1}
    try:
        r = requests.get(OLS_URL, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            docs = r.json().get("response", {}).get("docs", [])
            if docs:
                raw = docs[0].get("short_form", "")
                if raw:
                    return raw.replace("_", ":")
    except requests.RequestException:
        pass
    return None


def get_concept_id(term: str, cache: dict) -> str:
    """Return concept_id string (CHEBI:x / ATC:y / 'NOT_FOUND'). Uses cache."""
    key = clean_term(term.lower().strip())
    if not key:
        return "NOT_FOUND"
    if key in cache:
        cid = cache[key].get("concept_id")
        return cid if cid else "NOT_FOUND"

    cid = query_ols_api(key, "chebi")
    if cid:
        cache[key] = {"concept_id": cid, "ontology": "chebi"}
        time.sleep(API_DELAY)
        return cid

    cid = query_ols_api(key, "atc")
    if cid:
        cache[key] = {"concept_id": cid, "ontology": "atc"}
        time.sleep(API_DELAY)
        return cid

    cache[key] = {"concept_id": None, "ontology": None}
    time.sleep(API_DELAY)
    return "NOT_FOUND"


def tokenize_biomedical_text(text: str):
    """Must match Extraction_BIO.py exactly so TSV tokens align to passage chars."""
    text = re.sub(r'([.!?;:,(){}[\]"\'-])', r' \1 ', text)
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return [tok for tok in text.split() if tok]


def parse_tsv_passages(tsv_path: Path):
    """Split a TSV into per-passage blocks of (token, tag) pairs."""
    with open(tsv_path, "r", encoding="utf-8") as f:
        raw = f.read()
    lines = raw.split("\n")
    if lines and lines[0].startswith("Token"):
        lines = lines[1:]

    blocks = []
    current = []
    for line in lines:
        if line == "":
            blocks.append(current)
            current = []
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split()
            if len(parts) < 2:
                continue
        current.append((parts[0], parts[-1].strip()))
    if current:
        blocks.append(current)
    return blocks


def find_token_offsets(text: str, tokens):
    """
    Walk the passage text, locating each token as the first occurrence at or
    after the running cursor. Returns a list of (start, end) or None.
    Since tokenize_biomedical_text only inserts spaces (never alters chars),
    every token is guaranteed to be a contiguous substring of the original.
    """
    offsets = []
    cursor = 0
    for tok in tokens:
        if not tok:
            offsets.append(None)
            continue
        idx = text.find(tok, cursor)
        if idx == -1:
            offsets.append(None)
            continue
        offsets.append((idx, idx + len(tok)))
        cursor = idx + len(tok)
    return offsets


def build_entities(tokens, tags, offsets, text):
    """Group B/I spans into entity dicts {start, end, text}."""
    entities = []
    cur = None
    for tok, tag, off in zip(tokens, tags, offsets):
        if tag == "B":
            if cur:
                entities.append(cur)
            cur = {"start": off[0], "end": off[1]} if off else None
        elif tag == "I":
            if off is None:
                continue
            if cur is None:
                cur = {"start": off[0], "end": off[1]}
            else:
                cur["end"] = off[1]
        else:
            if cur:
                entities.append(cur)
                cur = None
    if cur:
        entities.append(cur)

    out = []
    for e in entities:
        e["text"] = text[e["start"]:e["end"]].strip()
        if len(e["text"]) >= MIN_ENTITY_LENGTH:
            out.append(e)
    return out


def annotate_document(bioc_path: Path, tsv_path: Path, cache: dict):
    with open(bioc_path, "r", encoding="utf-8") as f:
        bioc = json.load(f)

    blocks = parse_tsv_passages(tsv_path)
    while blocks and not blocks[-1]:
        blocks.pop()

    passages = [p for d in bioc.get("documents", []) for p in d.get("passages", [])]

    ann_counter = 1
    mismatch_warnings = 0
    for i, passage in enumerate(passages):
        if "annotations" not in passage:
            passage["annotations"] = []

        text = passage.get("text", "")
        if not text:
            continue
        if i >= len(blocks):
            continue

        block = blocks[i]
        if not block:
            continue

        tokens = [t for t, _ in block]
        tags = [tg for _, tg in block]

        # Sanity check: expected tokens (from tokenizer replay) vs TSV tokens.
        expected = tokenize_biomedical_text(text)
        if expected != tokens:
            mismatch_warnings += 1
            # Fall back to the TSV's tokens -- they are what the model saw.

        offsets = find_token_offsets(text, tokens)
        entities = build_entities(tokens, tags, offsets, text)

        passage_offset = passage.get("offset", 0)
        for ent in entities:
            identifier = get_concept_id(ent["text"], cache)
            passage["annotations"].append({
                "id": f"A{ann_counter}",
                "infons": {"type": "Chemical", "identifier": identifier},
                "text": ent["text"],
                "locations": [{
                    "offset": passage_offset + ent["start"],
                    "length": ent["end"] - ent["start"],
                }],
            })
            ann_counter += 1

    return bioc, mismatch_warnings, ann_counter - 1


def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: cache at {path} is corrupt, starting fresh.")
    return {}


def save_cache(path: Path, cache: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def discover_pairs(input_root: Path):
    """Yield (json_dir, results_dir) pairs that both exist."""
    pairs = []
    for json_dir in sorted(input_root.glob("PMC*_json_ascii")):
        if not json_dir.is_dir() or json_dir.name.endswith("_results"):
            continue
        results_dir = json_dir.parent / f"{json_dir.name}_results"
        if results_dir.is_dir():
            pairs.append((json_dir, results_dir))
    return pairs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_root", default="../input",
                   help="Folder holding PMC*_json_ascii/ and PMC*_json_ascii_results/ pairs.")
    p.add_argument("--json_dirs", nargs="*", default=None,
                   help="Specific PMC*_json_ascii/ folders to process (overrides scan).")
    p.add_argument("--output_suffix", default="_annotated_json",
                   help="Suffix appended to each json-dir name for output.")
    p.add_argument("--cache_path", default="./normalization_cache.json")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-annotate files even if the output JSON already exists.")
    return p.parse_args()


def main():
    args = parse_args()
    cache_path = Path(args.cache_path)
    cache = load_cache(cache_path)
    print(f"Loaded {len(cache)} cached lookups from {cache_path}")

    if args.json_dirs:
        pairs = []
        for d in args.json_dirs:
            jd = Path(d)
            rd = jd.parent / f"{jd.name}_results"
            if jd.is_dir() and rd.is_dir():
                pairs.append((jd, rd))
            else:
                print(f"Skipping {d}: missing BioC dir or _results dir.", file=sys.stderr)
    else:
        root = Path(args.input_root)
        if not root.exists():
            print(f"Input root not found: {root}", file=sys.stderr)
            sys.exit(1)
        pairs = discover_pairs(root)

    if not pairs:
        print("No (json_dir, results_dir) pairs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(pairs)} folder pair(s).")

    total_processed = 0
    total_skipped = 0
    total_missing_tsv = 0

    try:
        for json_dir, results_dir in pairs:
            out_dir = json_dir.parent / f"{json_dir.name}{args.output_suffix}"
            out_dir.mkdir(parents=True, exist_ok=True)

            json_files = sorted(json_dir.glob("*.json"))
            print(f"\n[{json_dir.name}] {len(json_files)} docs -> {out_dir.name}/")

            for idx, bioc_path in enumerate(json_files, 1):
                out_path = out_dir / bioc_path.name
                if out_path.exists() and not args.overwrite:
                    total_skipped += 1
                    continue

                tsv_path = results_dir / f"{bioc_path.stem}_annotated.tsv"
                if not tsv_path.exists():
                    total_missing_tsv += 1
                    if total_missing_tsv <= 5:
                        print(f"  [{idx}/{len(json_files)}] {bioc_path.stem}: no TSV, skipping")
                    continue

                bioc, mismatches, n_ann = annotate_document(bioc_path, tsv_path, cache)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(bioc, f, indent=2, ensure_ascii=False)
                total_processed += 1

                tag = f" ({mismatches} passage mismatches)" if mismatches else ""
                print(f"  [{idx}/{len(json_files)}] {bioc_path.stem}: {n_ann} annotations{tag}")

                if total_processed % CACHE_SAVE_EVERY == 0:
                    save_cache(cache_path, cache)
    finally:
        save_cache(cache_path, cache)
        print(
            f"\nDone. Processed {total_processed}, skipped {total_skipped} existing, "
            f"missing TSV for {total_missing_tsv}. Cache: {len(cache)} entries."
        )


if __name__ == "__main__":
    main()
