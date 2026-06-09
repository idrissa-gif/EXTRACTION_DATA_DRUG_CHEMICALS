"""
Annotate + normalize the FAIRClinical *supplementary* BioC files in a single pass.

Unlike ``normalize_to_json.py`` (which consumes a flat ``PMC*_json_ascii/`` dir
plus pre-computed ``_annotated.tsv`` files), the supplementary archives have a
*nested* layout and ship without any model tags:

    <root>/PMC###XXXXX_json_ascii_supplementary/
        PMC<id>_supplementary/
            Processed/
                <name>_bioc.json     <- the only files we touch
                <name>_tables.json   <- ignored
            Raw/...                  <- ignored (DOC/PPT/PDF/...)

So this driver walks ``<root>`` recursively, keeps only files whose name ends
with ``bioc.json``, runs the joint Chem+Drug BioBERT NER on each passage, links
every surface form to ontology IDs (sharing the resolver registry and on-disk
cache used everywhere else), and writes the ``annotations`` list back **in
place** into the same ``*_bioc.json`` file.

Resumable: a processed collection gets ``infons.chemdrug_normalized = "1"`` and
is skipped on re-runs unless ``--overwrite`` is passed.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("USE_TORCH", "1")

# This file lives in fairClinical_normalization/, so its directory is sys.path[0]
# and the sibling modules import cleanly.
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from Extraction_BIO_Annotation import extract_entities_from_pipeline
from normalize_to_json import (
    DEFAULT_CHEM_ONTOLOGIES,
    DEFAULT_DRUG_ONTOLOGIES,
    INFON_FIELD,
    RESOLVERS,
    _parse_ontology_list,
    load_cache,
    lookup_term,
    primary_identifier,
    save_cache,
)

CACHE_SAVE_EVERY = 25
NORMALIZED_FLAG = "chemdrug_normalized"


def ner_passage(ner, text):
    """Run NER on one passage, chunking on the rare over-length passage.

    Returns pipeline results with character offsets relative to ``text``.
    The common path is a single ``ner(text)`` call (identical to the main
    pipeline); only if that raises do we fall back to whitespace-aligned
    windows with offset correction.
    """
    try:
        return ner(text)
    except Exception:
        pass

    results = []
    window = 1500  # characters; well under the 512-token limit for normal prose
    base = 0
    n = len(text)
    while base < n:
        end = min(base + window, n)
        if end < n:
            sp = text.rfind(" ", base, end)
            if sp > base:
                end = sp
        chunk = text[base:end]
        try:
            for res in ner(chunk):
                res = dict(res)
                res["start"] += base
                res["end"] += base
                results.append(res)
        except Exception:
            pass
        base = end
    return results


def annotate_collection(bioc, ner, cache, chem_ontologies, drug_ontologies):
    """Populate every passage's ``annotations`` list. Returns annotation count."""
    n_ann = 0
    for doc in bioc.get("documents", []):
        for passage in doc.get("passages", []):
            text = passage.get("text", "")
            if not text:
                continue
            passage.setdefault("annotations", [])
            passage_offset = passage.get("offset", 0)
            ann_counter = len(passage["annotations"]) + 1

            ner_results = ner_passage(ner, text)
            entities = extract_entities_from_pipeline(ner_results, text)

            for ent in entities:
                ontologies = (drug_ontologies if ent["type"] == "Drug"
                              else chem_ontologies)
                hits = lookup_term(ent["text"], ontologies, cache)
                infons = {
                    "type": ent["type"],
                    "identifier": primary_identifier(hits, ontologies),
                }
                for ont, cid in hits.items():
                    if cid:
                        infons[INFON_FIELD[ont]] = cid

                passage["annotations"].append({
                    "id": f"A{ann_counter}",
                    "infons": infons,
                    "text": ent["text"],
                    "locations": [{
                        "offset": passage_offset + ent["start"],
                        "length": ent["end"] - ent["start"],
                    }],
                })
                ann_counter += 1
                n_ann += 1
    return n_ann


def discover_bioc_files(root: Path):
    """All *bioc.json files under root, excluding *_tables.json, sorted."""
    return sorted(p for p in root.rglob("*bioc.json") if p.is_file())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="to_process",
                   help="Folder holding the extracted PMC*_supplementary trees.")
    p.add_argument("--model_path", default="Models/biobert_chemdrug/final_checkpoint")
    p.add_argument("--cache_path", default="fairClinical_normalization/normalization_cache.json")
    p.add_argument("--chem_ontologies", default=",".join(DEFAULT_CHEM_ONTOLOGIES))
    p.add_argument("--drug_ontologies", default=",".join(DEFAULT_DRUG_ONTOLOGIES))
    p.add_argument("--overwrite", action="store_true",
                   help="Re-annotate files already marked chemdrug_normalized.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process at most N files (0 = all). For smoke tests.")
    return p.parse_args()


def main():
    args = parse_args()
    chem_ontologies = _parse_ontology_list(args.chem_ontologies, DEFAULT_CHEM_ONTOLOGIES)
    drug_ontologies = _parse_ontology_list(args.drug_ontologies, DEFAULT_DRUG_ONTOLOGIES)
    print(f"Chemical ontology priority: {chem_ontologies}", flush=True)
    print(f"Drug ontology priority    : {drug_ontologies}", flush=True)

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"Root not found: {root}")

    files = discover_bioc_files(root)
    if args.limit:
        files = files[:args.limit]
    print(f"Found {len(files)} *bioc.json files under {root}", flush=True)
    if not files:
        return

    cache_path = Path(args.cache_path)
    cache = load_cache(cache_path)
    print(f"Loaded {len(cache)} cached terms from {cache_path}", flush=True)

    import torch
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading model on {'GPU' if device == 0 else 'CPU'}: {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    ner = pipeline("ner", model=model, tokenizer=tokenizer,
                   aggregation_strategy="none", device=device)

    processed = skipped = failed = total_ann = 0
    t0 = time.time()
    try:
        for i, path in enumerate(files, 1):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    bioc = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                failed += 1
                print(f"  [{i}/{len(files)}] {path.name}: read error ({e})", flush=True)
                continue

            infons = bioc.setdefault("infons", {})
            if infons.get(NORMALIZED_FLAG) and not args.overwrite:
                skipped += 1
                continue

            n_ann = annotate_collection(bioc, ner, cache,
                                        chem_ontologies, drug_ontologies)
            infons[NORMALIZED_FLAG] = "1"

            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(bioc, f, indent=2, ensure_ascii=False)
            tmp.replace(path)

            processed += 1
            total_ann += n_ann
            if processed % 50 == 0 or i == len(files):
                rate = i / max(time.time() - t0, 1e-6)
                print(f"  [{i}/{len(files)}] {path.name}: {n_ann} ann "
                      f"| done={processed} skip={skipped} fail={failed} "
                      f"| {rate:.1f} files/s", flush=True)
            if processed % CACHE_SAVE_EVERY == 0:
                save_cache(cache_path, cache)
    finally:
        save_cache(cache_path, cache)
        print(f"\nDone. processed={processed} skipped={skipped} failed={failed} "
              f"annotations={total_ann} cache={len(cache)} terms "
              f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
