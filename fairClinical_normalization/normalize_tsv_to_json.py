"""
Produce *_normalized.json files from BIO-tagged TSVs.

For each ``input/PMC*_json_ascii_results/<doc>_annotated.tsv`` this writes
``input/PMC*_json_ascii_results_normalized/<doc>_normalized.json`` matching
the schema already established for PMC000:

    {
      "document_id": "<doc>",
      "source_tsv": "<absolute path to TSV>",
      "entities": [
        {"text": "...", "concept_id": "CHEBI:..."|null,
         "ontology": "chebi"|"atc"|null, "first_token_line": <int>}
      ],
      "summary": {"total_occurrences": N, "unique_terms": M,
                  "ontology_counts": {"chebi": x, "atc": y, "not_found": z}}
    }

``first_token_line`` is the 0-indexed file line (header occupies line 0,
so the first token row is line 1).

Lookups use ``normalization_cache.json`` only -- no network calls -- so the
script is fast and reproducible. Terms missing from the cache are emitted
with ``concept_id`` and ``ontology`` set to null.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def clean_term(term: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", "", term).split())


def load_cache(path: Path) -> dict:
    if not path.exists():
        print(f"Cache not found: {path}", file=sys.stderr)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup(term: str, cache: dict):
    key = clean_term(term.lower().strip())
    entry = cache.get(key)
    if not entry:
        return None, None
    return entry.get("concept_id"), entry.get("ontology")


def normalize_tsv(tsv_path: Path, cache: dict) -> dict:
    entities = []
    cur_tokens = None
    cur_line = None

    with open(tsv_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue  # header
            line = line.rstrip("\n")
            if not line:
                if cur_tokens is not None:
                    entities.append((cur_line, cur_tokens))
                    cur_tokens, cur_line = None, None
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            tok = parts[0]
            tag = parts[-1].strip()
            if tag == "B":
                if cur_tokens is not None:
                    entities.append((cur_line, cur_tokens))
                cur_tokens = [tok]
                cur_line = idx  # 0-indexed file line (header is line 0)
            elif tag == "I":
                if cur_tokens is None:
                    cur_tokens = [tok]
                    cur_line = idx
                else:
                    cur_tokens.append(tok)
            else:
                if cur_tokens is not None:
                    entities.append((cur_line, cur_tokens))
                    cur_tokens, cur_line = None, None
        if cur_tokens is not None:
            entities.append((cur_line, cur_tokens))

    out_entities = []
    chebi = atc = not_found = 0
    unique = set()
    for line, toks in entities:
        text = " ".join(toks)
        cid, ont = lookup(text, cache)
        out_entities.append({
            "text": text,
            "concept_id": cid,
            "ontology": ont,
            "first_token_line": line,
        })
        unique.add(text.lower())
        if ont == "chebi":
            chebi += 1
        elif ont == "atc":
            atc += 1
        else:
            not_found += 1

    return {
        "document_id": tsv_path.stem.replace("_annotated", ""),
        "source_tsv": str(tsv_path.resolve()),
        "entities": out_entities,
        "summary": {
            "total_occurrences": len(out_entities),
            "unique_terms": len(unique),
            "ontology_counts": {"chebi": chebi, "atc": atc, "not_found": not_found},
        },
    }


def discover_results_dirs(input_root: Path):
    return sorted(
        d for d in input_root.glob("PMC*_json_ascii_results")
        if d.is_dir() and not d.name.endswith("_normalized")
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_root", default="../input")
    p.add_argument("--results_dirs", nargs="*", default=None,
                   help="Specific PMC*_json_ascii_results/ dirs (overrides scan).")
    p.add_argument("--cache_path", default="./normalization_cache.json")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output_suffix", default="_normalized",
                   help="Suffix appended to each results-dir name for output.")
    return p.parse_args()


def main():
    args = parse_args()
    cache = load_cache(Path(args.cache_path))
    print(f"Loaded {len(cache)} cached lookups.")

    if args.results_dirs:
        dirs = [Path(d) for d in args.results_dirs]
    else:
        root = Path(args.input_root)
        dirs = discover_results_dirs(root)

    if not dirs:
        print("No PMC*_json_ascii_results/ dirs found.", file=sys.stderr)
        sys.exit(1)

    grand_processed = grand_skipped = 0
    for results_dir in dirs:
        out_dir = results_dir.parent / f"{results_dir.name}{args.output_suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        tsvs = sorted(results_dir.glob("*_annotated.tsv"))
        print(f"\n[{results_dir.name}] {len(tsvs)} tsvs -> {out_dir.name}/")
        proc = skip = 0
        for i, tsv in enumerate(tsvs, 1):
            doc_id = tsv.stem.replace("_annotated", "")
            out_path = out_dir / f"{doc_id}_normalized.json"
            if out_path.exists() and not args.overwrite:
                skip += 1
                continue
            data = normalize_tsv(tsv, cache)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            proc += 1
            if proc % 500 == 0:
                print(f"  [{i}/{len(tsvs)}] {doc_id}: {data['summary']['total_occurrences']} entities")
        print(f"  -> processed {proc}, skipped {skip}")
        grand_processed += proc
        grand_skipped += skip

    print(f"\nDone. Processed {grand_processed}, skipped {grand_skipped} existing.")


if __name__ == "__main__":
    main()
