"""
Normalization step for the FairClinical V2 Chemical/Drug NER pipeline.

Reads:
  1) The original BioC JSON files in ``input/PMC*_json_ascii/*.json`` (for
     passage text and character offsets).
  2) The BIO-tagged TSVs produced by ``Extraction_BIO.py`` (or the joint
     Chem+Drug annotator) in
     ``input/PMC*_json_ascii_results/*_annotated.tsv``.

Recognized BIO tag set:
  * ``B-Chem`` / ``I-Chem``  -> Chemical
  * ``B-Drug`` / ``I-Drug``  -> Drug
  * ``B`` / ``I`` (legacy)   -> Chemical (back-compat with the chem-only model)

Each surface form is then linked to one or more ontology concept IDs. The
ontology priority is configurable per entity type. Supported back-ends:

  ============  ============================================  ==========
  Key           Source                                        ID prefix
  ============  ============================================  ==========
  ``chebi``     EBI OLS, ``ontology=chebi``                   ``CHEBI:``
  ``atc``       EBI OLS, ``ontology=atc``                     ``ATC:``
  ``mesh``      EBI OLS, ``ontology=mesh``                    ``MESH:``
  ``drugbank``  EBI OLS, ``ontology=drugbank``                ``DRUGBANK:``
  ``pubchem``   PubChem PUG REST (compound name lookup)       ``CID:``
  ``rxnorm``    NLM RxNav (``rxcui`` exact-name lookup)       ``RXCUI:``
  ============  ============================================  ==========

The script walks the configured ontology list in order. The **first** hit
becomes the canonical ``infons.identifier``. **Every** hit is additionally
recorded in a per-ontology infon field (``chebi_id``, ``atc_id``,
``mesh_id``, ``drugbank_id``, ``pubchem_id``, ``rxnorm_id``) so downstream
consumers can join on any vocabulary they like.

Lookups are cached on disk at ``normalization_cache.json``. Old caches with
the flat ``{concept_id, ontology}`` shape are migrated to the new
``{ontology_key: id_or_null}`` shape on load so previously-resolved chemical
terms are not re-queried.

Output: one BioC JSON per document in
``input/PMC*_json_ascii_annotated_json/<docid>.json`` with each passage's
``annotations`` list populated.

Resumable: files whose output already exists are skipped unless
``--overwrite`` is passed.
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests


OLS_URL = "https://www.ebi.ac.uk/ols/api/search"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
RXNAV_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"

API_DELAY = 0.05
REQUEST_TIMEOUT = 5
MIN_ENTITY_LENGTH = 3
CACHE_SAVE_EVERY = 20

DEFAULT_CHEM_ONTOLOGIES = ["chebi", "atc"]
DEFAULT_DRUG_ONTOLOGIES = ["atc", "chebi"]


# ---------------------------------------------------------------------------
# Ontology resolvers. Each takes a cleaned term and returns either a prefixed
# concept id ("CHEBI:12345", "ATC:N02BE01", "CID:2244", ...) or None.
# ---------------------------------------------------------------------------

def _ols_lookup(term: str, ontology: str, prefix: str | None = None):
    params = {"q": term, "ontology": ontology, "exact": False, "rows": 1}
    try:
        r = requests.get(OLS_URL, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None
        docs = r.json().get("response", {}).get("docs", [])
        if not docs:
            return None
        raw = docs[0].get("short_form", "")
        if not raw:
            return None
        cid = raw.replace("_", ":")
        if prefix and not cid.upper().startswith(prefix.upper() + ":"):
            cid = f"{prefix}:{cid}"
        return cid
    except requests.RequestException:
        return None


def _resolve_chebi(term):    return _ols_lookup(term, "chebi", "CHEBI")
def _resolve_atc(term):      return _ols_lookup(term, "atc", "ATC")
def _resolve_mesh(term):     return _ols_lookup(term, "mesh", "MESH")
def _resolve_drugbank(term): return _ols_lookup(term, "drugbank", "DRUGBANK")


def _resolve_pubchem(term):
    try:
        url = PUBCHEM_URL.format(name=urllib.parse.quote(term, safe=""))
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None
        cids = r.json().get("IdentifierList", {}).get("CID", [])
        if cids:
            return f"CID:{cids[0]}"
    except (requests.RequestException, ValueError):
        pass
    return None


def _resolve_rxnorm(term):
    try:
        r = requests.get(RXNAV_URL, params={"name": term}, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None
        ids = r.json().get("idGroup", {}).get("rxnormId", [])
        if ids:
            return f"RXCUI:{ids[0]}"
    except (requests.RequestException, ValueError):
        pass
    return None


RESOLVERS = {
    "chebi":    _resolve_chebi,
    "atc":      _resolve_atc,
    "mesh":     _resolve_mesh,
    "drugbank": _resolve_drugbank,
    "pubchem":  _resolve_pubchem,
    "rxnorm":   _resolve_rxnorm,
}

# Per-ontology infon field name in the BioC output.
INFON_FIELD = {k: f"{k}_id" for k in RESOLVERS}


# ---------------------------------------------------------------------------
# Term cleaning + cached lookup orchestration.
# ---------------------------------------------------------------------------

def clean_term(term: str) -> str:
    """Strip HTML and collapse whitespace for API queries."""
    return " ".join(re.sub(r"<[^>]+>", "", term).split())


def lookup_term(term: str, ontologies, cache: dict) -> dict:
    """
    Resolve `term` against each ontology in order. Returns a dict
    ``{ontology_key: concept_id_or_None}`` containing entries for every
    ontology in ``ontologies`` (including the ones that returned None).

    Cache shape: ``cache[normalized_term][ontology_key] = id | None``.
    A missing key means the lookup has never been performed.
    """
    key = clean_term(term.lower().strip())
    if not key:
        return {ont: None for ont in ontologies}

    entry = cache.setdefault(key, {})
    out = {}
    for ont in ontologies:
        if ont not in RESOLVERS:
            out[ont] = None
            continue
        if ont in entry:
            out[ont] = entry[ont]
            continue
        cid = RESOLVERS[ont](key)
        entry[ont] = cid
        out[ont] = cid
        time.sleep(API_DELAY)
    return out


def primary_identifier(hits: dict, ontologies) -> str:
    """First non-None concept id following the configured priority order."""
    for ont in ontologies:
        cid = hits.get(ont)
        if cid:
            return cid
    return "NOT_FOUND"


# ---------------------------------------------------------------------------
# Tokenization and TSV/passage alignment (must match Extraction_BIO.py).
# ---------------------------------------------------------------------------

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


# Map tag suffix -> entity type. Bare B/I (no suffix) is the legacy
# chem-only model and resolves to Chemical.
_TYPE_FROM_SUFFIX = {"chem": "Chemical", "drug": "Drug", "": "Chemical"}


def _decode_tag(tag: str):
    """Return (prefix, type) where prefix is 'B'|'I'|'O'."""
    if tag == "O" or not tag:
        return ("O", None)
    head, _, suffix = tag.partition("-")
    if head not in ("B", "I"):
        return ("O", None)
    ent_type = _TYPE_FROM_SUFFIX.get(suffix.lower())
    return (head, ent_type)


def build_entities(tokens, tags, offsets, text):
    """Group typed B/I spans into entity dicts {start, end, text, type}."""
    entities = []
    cur = None
    for tok, tag, off in zip(tokens, tags, offsets):
        prefix, ent_type = _decode_tag(tag)
        if prefix == "B":
            if cur:
                entities.append(cur)
            cur = {"start": off[0], "end": off[1], "type": ent_type} if off else None
        elif prefix == "I":
            if off is None:
                continue
            if cur is None:
                cur = {"start": off[0], "end": off[1], "type": ent_type}
            else:
                # Continue the current span. If the suffix flipped mid-entity
                # (e.g. B-Chem followed by I-Drug), trust the opening tag.
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


# ---------------------------------------------------------------------------
# Per-document driver.
# ---------------------------------------------------------------------------

def annotate_document(bioc_path: Path, tsv_path: Path, cache: dict,
                      chem_ontologies, drug_ontologies):
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

        expected = tokenize_biomedical_text(text)
        if expected != tokens:
            mismatch_warnings += 1
            # Fall back to the TSV's tokens -- they are what the model saw.

        offsets = find_token_offsets(text, tokens)
        entities = build_entities(tokens, tags, offsets, text)

        passage_offset = passage.get("offset", 0)
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

    return bioc, mismatch_warnings, ann_counter - 1


# ---------------------------------------------------------------------------
# Cache I/O (with one-shot migration from the legacy flat shape).
# ---------------------------------------------------------------------------

def _migrate_cache_entry(value):
    """
    Old entries: ``{"concept_id": "CHEBI:123" | None, "ontology": "chebi" | "atc" | None}``
    The legacy code tried chebi first, then atc; so:
      - hit on chebi  -> {"chebi": "CHEBI:123"}
      - hit on atc    -> {"chebi": None, "atc": "ATC:..."}
      - both missed   -> {"chebi": None, "atc": None}
    """
    if not isinstance(value, dict):
        return {}
    if "concept_id" not in value and "ontology" not in value:
        # Already in the new shape (or unknown) -- keep as-is, dropping unknown keys.
        return {k: v for k, v in value.items() if k in RESOLVERS}

    cid = value.get("concept_id")
    ont = value.get("ontology")
    if ont == "chebi" and cid:
        return {"chebi": cid}
    if ont == "atc" and cid:
        return {"chebi": None, "atc": cid}
    # Either explicit double-miss, or an unrecognized legacy ontology.
    return {"chebi": None, "atc": None}


def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: cache at {path} is corrupt, starting fresh.")
        return {}

    migrated = 0
    out = {}
    for term, val in raw.items():
        new_val = _migrate_cache_entry(val)
        if isinstance(val, dict) and "concept_id" in val:
            migrated += 1
        out[term] = new_val
    if migrated:
        print(f"Migrated {migrated} legacy cache entries to per-ontology format.")
    return out


def save_cache(path: Path, cache: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Folder discovery + CLI.
# ---------------------------------------------------------------------------

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


def _parse_ontology_list(raw: str, default):
    if not raw:
        return list(default)
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    unknown = [x for x in items if x not in RESOLVERS]
    if unknown:
        raise SystemExit(
            f"Unknown ontology key(s): {unknown}. "
            f"Supported: {sorted(RESOLVERS)}"
        )
    return items


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_root", default="../input",
                   help="Folder holding PMC*_json_ascii/ and PMC*_json_ascii_results/ pairs.")
    p.add_argument("--json_dirs", nargs="*", default=None,
                   help="Specific PMC*_json_ascii/ folders to process (overrides scan).")
    p.add_argument("--output_suffix", default="_annotated_json",
                   help="Suffix appended to each json-dir name for output.")
    p.add_argument("--cache_path", default="./normalization_cache.json")
    p.add_argument("--chem_ontologies",
                   default=",".join(DEFAULT_CHEM_ONTOLOGIES),
                   help=("Ordered, comma-separated ontology keys for Chemical "
                         "entities. Supported: " + ",".join(sorted(RESOLVERS)) +
                         f". Default: {','.join(DEFAULT_CHEM_ONTOLOGIES)}"))
    p.add_argument("--drug_ontologies",
                   default=",".join(DEFAULT_DRUG_ONTOLOGIES),
                   help=("Ordered, comma-separated ontology keys for Drug "
                         "entities. Supported: " + ",".join(sorted(RESOLVERS)) +
                         f". Default: {','.join(DEFAULT_DRUG_ONTOLOGIES)}"))
    p.add_argument("--overwrite", action="store_true",
                   help="Re-annotate files even if the output JSON already exists.")
    return p.parse_args()


def main():
    args = parse_args()

    chem_ontologies = _parse_ontology_list(args.chem_ontologies, DEFAULT_CHEM_ONTOLOGIES)
    drug_ontologies = _parse_ontology_list(args.drug_ontologies, DEFAULT_DRUG_ONTOLOGIES)
    print(f"Chemical ontology priority: {chem_ontologies}")
    print(f"Drug ontology priority    : {drug_ontologies}")

    cache_path = Path(args.cache_path)
    cache = load_cache(cache_path)
    print(f"Loaded {len(cache)} cached terms from {cache_path}")

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

                bioc, mismatches, n_ann = annotate_document(
                    bioc_path, tsv_path, cache, chem_ontologies, drug_ontologies,
                )
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
            f"missing TSV for {total_missing_tsv}. Cache: {len(cache)} terms."
        )


if __name__ == "__main__":
    main()
