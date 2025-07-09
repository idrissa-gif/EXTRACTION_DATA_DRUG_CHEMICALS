import json
import re
import sys
import argparse

def tokanization(text):

    tokens = []

    for match in re.finditer(r'\w+|[^\w\s]', text):
        tok = match.group(0)
        tokens.append(tok)

    return tokens

def convert_bio_red_to_iob(input_json_path, output_tsv_path):

    with open(input_json_path, 'r', encoding='utf-8') as infile, \
        open(output_tsv_path, 'w', encoding='utf-8') as outfile:
        data = json.load(infile)

        for doc in data.get("documents",[]):
            for passage in doc.get("passages",[]):
                text = passage.get("text","")
                base_offset = passage.get("offset", 0)

                chem_words = []

                for ann in passage.get("annotations", []):
                    if ann.get("infons",{}).get("type") == "ChemicalEntity":
                        chem_words.append(ann.get('text'))
                if not chem_words:
                    continue

                passage_tokens = tokanization(text)
                # print(passage_tokens)

                for tok in passage_tokens:
                    tag = "O"
                    for chem_word in chem_words:
                        dict_chem_words = tokanization(chem_word)
                        # print(chem_word,dict_chem_words)

                        if tok in dict_chem_words:
                            tag = "B" if tok == dict_chem_words[0]  else "I"
                            break
                    outfile.write(f"{tok}\t{tag}\n")
                outfile.write("\n")
def combine_tsvs(biored_tsv, bc4_tsv, bc5_tsv, output_tsv):
    with open(output_tsv, 'w', encoding="utf-8") as out:
        for path in [biored_tsv, bc4_tsv, bc5_tsv]:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    out.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert BioRED+BC4+BC5 into a single IOB-TSV for train/test."
    )
    parser.add_argument(
        "--split", choices=["train", "test"], required=True, help="Which split to process"
    )
    args = parser.parse_args()
    if args.split == "train":
        bio_red_json = "/people/dicko/Extraction_Training_Drug_Chemical/BioRED/Train.BioC.JSON"
        bc4_tsv       = "/people/dicko/Extraction_Training_Drug_Chemical/Compact-Biomedical-Transformers/datasets/NER/BC4CHEMD/train.tsv"
        bc5_tsv       = "/people/dicko/Extraction_Training_Drug_Chemical/Compact-Biomedical-Transformers/datasets/NER/BC5CDR-chem/train.tsv"
    else:
        bio_red_json = "/people/dicko/Extraction_Training_Drug_Chemical/BioRED/Test.BioC.JSON"
        bc4_tsv      = "/people/dicko/Extraction_Training_Drug_Chemical/Compact-Biomedical-Transformers/datasets/NER/BC4CHEMD/test.tsv"
        bc5_tsv      = "/people/dicko/Extraction_Training_Drug_Chemical/Compact-Biomedical-Transformers/datasets/NER/BC5CDR-chem/test.tsv"

    biored_iob = f"biored_chem_{args.split}_iob.tsv"
    combined_out = f"combined_chem_{args.split}.tsv"

    convert_bio_red_to_iob(bio_red_json, biored_iob)

    print(f"Converted BioRED ({args.split}) -> {biored_iob}", file=sys.stderr)
    combine_tsvs(biored_iob, bc4_tsv, bc5_tsv, combined_out)
    print(f"All datasets ({args.split}) combined -> {combined_out}", file=sys.stderr)
