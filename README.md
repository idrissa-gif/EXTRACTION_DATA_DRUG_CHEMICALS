# EXTRACTION_DATA_DRUG_CHEMICALS
![NER Flowchart](FlowChart.png)

### Hardware
* **GPU:** NVIDIA GPU with at least 16GB VRAM (24GB+ recommended for training).
    * *Tested on:* **2x NVIDIA RTX A6000 (48GB each)**
* **RAM:** 32GB+ System RAM

# Requirements
Python 3.12.0
Transformers 4.45.2
Datasets 3.4.1

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

For the reproducibility of this code run this code.

CUDA_VISIBLE_DEVICES=0,1,... python finetuned_biobert_args.py \
    --train_file `TrainPath` \
    --dev_file `DevPath` \
    --test_file `TestPath` \
    --output_dir `OutputFolder`

### Example

CUDA_VISIBLE_DEVICES=0 python finetuned_biobert_args.py \
    --train_file /combined_chem_train.tsv \
    --dev_file /combined_chem_dev.tsv \
    --test_file /biored_chem_test_iob.tsv \
    --output_dir biobert_biored
```

