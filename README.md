# EXTRACTION_DATA_DRUG_CHEMICALS
![NER Flowchart](FlowChart.png)

# Requirements
Python 3.12.0
Transformers 4.45.2
Datasets 3.4.1

For the reproducibility of this code run this code.

CUDA_VISIBLE_DEVICES=0,1,... python finetuned_biobert_args.py --train_file `TrainPath` --dev_f
ile `DevPath` --test_file `TestPath` --output_dir `OutputFolder`

### Example
```
CUDA_VISIBLE_DEVICES=0 python finetuned_biobert_args.py --train_file /combined_chem_train.tsv  --dev_f
ile /combined_chem_dev.tsv --test_file /biored_chem_test_iob.tsv --output_dir biobert_biored
```