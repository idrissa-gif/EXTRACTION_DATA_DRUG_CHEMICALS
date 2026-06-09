# Synergizing Domain-Specific Masked Language Models and Instruction-Tuned LLMs for Chemical NER
![Methodology Flowchart](FlowChart.png)

This repository contains the official implementation of the paper **"Synergizing Domain-Specific Masked Language Models and Instruction-Tuned LLMs for Chemical NER"**, submitted to LREC 2026.

The framework implements a robust two-stage Chemical Named Entity Recognition (NER) system:
1.  **Supervised Fine-tuning:** Optimization of domain-specific MLMs (BioBERT) using automated hyperparameter tuning (Optuna).
2.  **Generative Tagging with Guidance:** A novel pipeline where instruction-tuned LLMs (Mistral, LLaMA, DeepSeek) generate BIO tags, guided by weak supervision from the specialized MLM.


### Hardware
* **GPU:** NVIDIA GPU with at least 16GB VRAM (24GB+ recommended for training).
    * *Tested on:* **2x NVIDIA RTX A6000 (48GB each)**
* **RAM:** 32GB+ System RAM

# Requirements
* **Python:** 3.12.0
* **Transformers:** 4.45.2
* **Datasets:** 3.4.1
* **PyTorch:** Compatible with CUDA 11.8/12.x
* **Seqeval:** For evaluation metrics

Install dependencies via pip:

```bash
pip install torch transformers datasets pandas scikit-learn seqeval tqdm requests
```

## Data Format

### 1. NER Training Data (TSV)
The training scripts expect data in standard **CoNLL-2003** format (token-per-line).
* Columns: `Token` `Label`
* Labels: `B`, `I`, `O`

**Example (`train.tsv`):**
```tsv
Aspirin   B
is        O
used      O
for       O
pain      O
.         O
```

For the reproducibility of this code, run the commands below.

# Supervised Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0,1,... python finetuned_biobert_args.py \
    --train_file TrainPath \
    --dev_file DevPath \
    --test_file TestPath \
    --output_dir OutputFolder
```
For train for the chemical and drug NER
```
 CUDA_VISIBLE_DEVICES=0 python finetuned_biobert_chemdrug.py \
    --chem_train ../chemdrug_chem_train.tsv \
    --chem_dev   ../chemdrug_chem_dev.tsv \
    --chem_test  ../chemdrug_chem_test.tsv \
    --drug_train ../chemdrug_drug_train.tsv \
    --drug_dev   ../chemdrug_drug_dev.tsv \
    --drug_test  ../chemdrug_drug_test.tsv \
    --output_dir biobert_chemdrug  
```
### Example

```bash
CUDA_VISIBLE_DEVICES=0 python finetuned_biobert_args.py \
    --train_file /combined_chem_train.tsv \
    --dev_file /combined_chem_dev.tsv \
    --test_file /biored_chem_test_iob.tsv \
    --output_dir biobert_biored
```

# Generative Tagging with Guidance

```bash
python llm_guided_biobert.py \
    --train_file "path/to/train.tsv" \
    --dev_file "path/to/dev.tsv" \
    --test_file "path/to/test.tsv" \
    --output_dir "./output_model" \
    --llm_model "epfl-llm/meditron-7b"
```

# Annotation (BioC JSON corpora)

Once the model is trained, you can annotate raw BioC JSON files (e.g. the FAIRClinical PMC dumps) with BIO tags.

### Annotate with the fine-tuned BioBERT

```bash
python fairClinical_annotation/Extraction_BIO.py \
    --model_path path/to/biobert_finetuned \
    --json_dir input/PMC000XXXXX_json_ascii
```

This writes one `*_annotated.tsv` file per document into `<json_dir>_results/`.

### Annotate with an instruction-tuned LLM guided by BioBERT priors

```bash
python fairClinical_annotation/Extraction_BIO_LLMS.py \
    --model_path mistralai/Mistral-7B-Instruct-v0.2 \
    --json_dir input/PMC000XXXXX_json_ascii \
    --biobert_priors_dir input/PMC000XXXXX_json_ascii_results \
    --biobert_help_mode both \
    --biobert_threshold 0.85
```

`--biobert_help_mode` accepts `both`, `prompt`, `constrain`, or `off`, depending on whether the BioBERT priors should guide the prompt, post-hoc constraints, both, or neither.

# Normalization (linking entities to ontology IDs)

After annotation, each surface form is linked to one or more ontology concept IDs. The script handles both **Chemical** and **Drug** entities (BIO tags `B-Chem`/`I-Chem` and `B-Drug`/`I-Drug`); legacy chem-only `B`/`I` tags are still treated as Chemical for back-compat.

The ontology priority is configurable per entity type. Supported back-ends:

| Key        | Source                                        | ID prefix    |
|------------|-----------------------------------------------|--------------|
| `chebi`    | EBI OLS, `ontology=chebi`                     | `CHEBI:`     |
| `atc`      | EBI OLS, `ontology=atc`                       | `ATC:`       |
| `mesh`     | EBI OLS, `ontology=mesh`                      | `MESH:`      |
| `drugbank` | EBI OLS, `ontology=drugbank`                  | `DRUGBANK:`  |
| `pubchem`  | PubChem PUG REST (compound name)              | `CID:`       |
| `rxnorm`   | NLM RxNav (`rxcui` exact-name lookup)         | `RXCUI:`     |

The script walks the configured ontology list in order. The **first** hit becomes the canonical `infons.identifier`; **every** hit is also recorded in a per-ontology infon field (`chebi_id`, `atc_id`, `mesh_id`, `drugbank_id`, `pubchem_id`, `rxnorm_id`) so downstream consumers can join on any vocabulary.

Lookups are cached on disk. Pre-existing caches in the legacy `{concept_id, ontology}` shape are migrated to the new per-ontology shape automatically on first load (no re-querying of already-resolved chemical terms).

```bash
python fairClinical_normalization/normalize_to_json.py \
    --input_root input \
    --cache_path fairClinical_normalization/normalization_cache.json \
    --chem_ontologies chebi,atc,mesh,pubchem \
    --drug_ontologies atc,rxnorm,drugbank,mesh,chebi
```

Defaults: `--chem_ontologies chebi,atc` and `--drug_ontologies atc,chebi` (mirrors the original behavior for chemicals while extending coverage to drugs).

Output is written to `input/PMC*_json_ascii_annotated_json/<docid>.json`. Each passage gains an `annotations` list of the form:

```json
{
  "id": "A1",
  "infons": {
    "type": "Drug",
    "identifier": "ATC:N02BE01",
    "atc_id": "ATC:N02BE01",
    "rxnorm_id": "RXCUI:161",
    "chebi_id": "CHEBI:46195"
  },
  "text": "paracetamol",
  "locations": [{"offset": 42, "length": 11}]
}
```

Pass `--overwrite` to re-normalize files that already have output, and `--json_dirs` to restrict the run to specific subdirectories.

If you want to run annotation and normalization in a single pass (useful for ad-hoc experiments), use:

```bash
python fairClinical_normalization/Extraction_BIO_Annotation.py \
    --model_path path/to/biobert_chemdrug \
    --json_dir input/PMC000XXXXX_json_ascii \
    --chem_ontologies chebi,atc,mesh,pubchem \
    --drug_ontologies atc,rxnorm,drugbank,mesh,chebi
```

This shares the same resolver registry, ontology-priority logic, and on-disk cache as `normalize_to_json.py`.

# Fine-tuned models

The fine-tuned models (MLMs and LLMs) are uploaded to the HuggingFace repository:
- `anonymous-research-2026/BioBert`
