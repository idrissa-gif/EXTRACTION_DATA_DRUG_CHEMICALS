import os
import json
import argparse
import time
import psutil
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['USE_TORCH'] = '1'

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

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

try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlShutdown,
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# Map tag suffix -> entity type. Bare B/I (no suffix) defaults to Chemical
# for backward compatibility with the chem-only model.
_TYPE_FROM_SUFFIX = {"chem": "Chemical", "drug": "Drug", "": "Chemical"}


def _entity_type_from_tag(tag: str):
    """Return ('B'|'I'|'O', 'Chemical'|'Drug'|None) for a model tag."""
    if not tag or tag == "O":
        return ("O", None)
    head, _, suffix = tag.partition("-")
    if head not in ("B", "I"):
        return ("O", None)
    return (head, _TYPE_FROM_SUFFIX.get(suffix.lower()))


def extract_entities_from_pipeline(ner_results, text):
    """
    Stitches subwords and multi-word entities back into full entity strings,
    and tags each entity with its type ('Chemical' or 'Drug') based on the
    model's BIO label suffix.
    """
    entities = []
    current_ent = None

    for res in ner_results:
        tag = res.get('entity', res.get('entity_group', 'O'))
        token_text = res.get('word', '')
        head, ent_type = _entity_type_from_tag(tag)

        is_subword = token_text.startswith('##') or (
            current_ent and res['start'] == current_ent['end']
        )

        if head == "O":
            if current_ent:
                entities.append(current_ent)
                current_ent = None
            continue

        is_contiguous = current_ent and (res['start'] == current_ent['end'])
        is_i_tag = current_ent and head == "I"
        should_merge = is_contiguous or is_i_tag or (is_subword and current_ent)

        if should_merge:
            current_ent['end'] = res['end']
            current_ent['text'] = text[current_ent['start']:res['end']]
        elif is_subword and not current_ent:
            continue
        else:
            if current_ent:
                entities.append(current_ent)
            current_ent = {
                'start': res['start'],
                'end': res['end'],
                'text': text[res['start']:res['end']],
                'type': ent_type or "Chemical",
            }

    if current_ent:
        entities.append(current_ent)

    MIN_ENTITY_LENGTH = 3
    return [e for e in entities if len(e['text']) >= MIN_ENTITY_LENGTH]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BIO-tag NER on BioC JSON files and append "
                    "ontology-linked annotations in a single pass."
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to the fine-tuned model")
    parser.add_argument("--json_dir", required=True,
                        help="Directory with BioC JSON files.")
    parser.add_argument("--cache_path", default="./normalization_cache.json",
                        help="On-disk cache of (term, ontology) -> concept_id lookups.")
    parser.add_argument("--chem_ontologies",
                        default=",".join(DEFAULT_CHEM_ONTOLOGIES),
                        help=("Ordered, comma-separated ontology keys for "
                              "Chemical entities. Supported: "
                              + ",".join(sorted(RESOLVERS))))
    parser.add_argument("--drug_ontologies",
                        default=",".join(DEFAULT_DRUG_ONTOLOGIES),
                        help=("Ordered, comma-separated ontology keys for "
                              "Drug entities. Supported: "
                              + ",".join(sorted(RESOLVERS))))
    return parser.parse_args()


def get_gpu_memory():
    if not GPU_AVAILABLE:
        return None
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()
    return info.used / 1024 ** 2


def main():
    args = parse_args()
    chem_ontologies = _parse_ontology_list(args.chem_ontologies, DEFAULT_CHEM_ONTOLOGIES)
    drug_ontologies = _parse_ontology_list(args.drug_ontologies, DEFAULT_DRUG_ONTOLOGIES)
    print(f"Chemical ontology priority: {chem_ontologies}")
    print(f"Drug ontology priority    : {drug_ontologies}")

    cache_path = Path(args.cache_path)
    cache = load_cache(cache_path)
    print(f"Loaded {len(cache)} cached terms from {cache_path}")

    process = psutil.Process(os.getpid())

    json_dir = Path(args.json_dir)
    result_dir = json_dir.parent / f"{json_dir.name}_annotated_json"
    result_dir.mkdir(parents=True, exist_ok=True)

    log_path = result_dir / "usage_log.txt"

    files_processed = 0
    try:
        with open(log_path, "w") as log:
            log.write("============================ Resource Usage Log ============================\n")

            t0 = time.time()
            import torch

            device = 0 if torch.cuda.is_available() else -1

            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            model = AutoModelForTokenClassification.from_pretrained(args.model_path)

            ner = pipeline("ner", model=model, tokenizer=tokenizer,
                           aggregation_strategy="none", device=device)

            print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
            model_load_time = time.time() - t0

            mem_used = process.memory_info().rss / 1024 ** 2
            gpu_used = get_gpu_memory()

            log.write(f"Model load time: {model_load_time:.2f} s\n")
            log.write(f"Memory after model load: {mem_used:.2f} MB\n")
            if gpu_used is not None:
                log.write(f"GPU memory after model load: {gpu_used:.2f} MB\n")

            for json_file in json_dir.glob("*.json"):
                file_start = time.time()

                with open(json_file, 'r', encoding="utf-8") as f:
                    bioc = json.load(f)

                for doc in bioc.get('documents', []):
                    for passage in doc.get('passages', []):
                        passage_text = passage.get("text", "")
                        if not passage_text:
                            continue

                        if "annotations" not in passage:
                            passage["annotations"] = []

                        passage_offset = passage.get("offset", 0)
                        ann_id_counter = len(passage["annotations"]) + 1

                        ner_results = ner(passage_text)
                        entities = extract_entities_from_pipeline(ner_results, passage_text)

                        for ent in entities:
                            ontologies = (drug_ontologies if ent["type"] == "Drug"
                                          else chem_ontologies)
                            hits = lookup_term(ent['text'], ontologies, cache)
                            infons = {
                                "type": ent["type"],
                                "identifier": primary_identifier(hits, ontologies),
                            }
                            for ont, cid in hits.items():
                                if cid:
                                    infons[INFON_FIELD[ont]] = cid

                            passage["annotations"].append({
                                "id": f"A{ann_id_counter}",
                                "infons": infons,
                                "text": ent['text'],
                                "locations": [{
                                    "offset": passage_offset + ent['start'],
                                    "length": ent['end'] - ent['start'],
                                }],
                            })
                            ann_id_counter += 1

                output_file = result_dir / f"{json_file.name}"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(bioc, f, indent=2, ensure_ascii=False)

                files_processed += 1
                if files_processed % 20 == 0:
                    save_cache(cache_path, cache)

                file_time = time.time() - file_start
                file_mem = process.memory_info().rss / 1024**2
                file_gpu = get_gpu_memory()

                log.write(f"\nFile: {json_file.name}\n")
                log.write(f"Time: {file_time:.2f} s\n")
                log.write(f"Memory: {file_mem:.2f} MB\n")
                if file_gpu is not None:
                    log.write(f"GPU memory: {file_gpu:.2f} MB\n")

                print(f"Annotated {json_file.name} -> {output_file.name}")
    finally:
        save_cache(cache_path, cache)


if __name__ == "__main__":
    main()
