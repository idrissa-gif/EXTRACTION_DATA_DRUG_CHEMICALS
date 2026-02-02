import os
import json
import torch
import argparse
import ast
import logging
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, DatasetDict
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from torch.nn import functional as F

        # ------------------------------ #
        #           Setup logging        #
        # ------------------------------ #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

        # ------------------------------- #
        #            Load LLM             #
        # ------------------------------- #
def load_llm(model_id, device_map="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    return tokenizer, model

# -----------------------
# LLM span extraction
# -----------------------
def get_llm_entity_spans(llm_tokenizer, llm_model, tokens):
    prompt = (
    f"Input sentence: {' '.join(tokens)}\n\n"
    "Task: Identify all *chemical* entities in the sentence.\n"
    "You must output *only* a single Python list literal of spans\n"
    "in the form [(start_idx, end_idx, \"EntityType\"), ...].\n"
    "If there are no entities, output exactly `[]` and nothing else.\n\n"
    "Answer:"
    )

    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)


    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        pad_token_id=llm_tokenizer.eos_token_id,
    )

    output_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"Output text :{output_text}")
    start = output_text.find('[')
    end = output_text.rfind(']')
    print(f"Start:{start} End:{end} List text:{output_text[start:end+1]}")

    if start == -1 or end < start:
        return []
    list_text = output_text[start:end+1]

    try:
        spans = ast.literal_eval(list_text)
    except Exception:
        return []
    clean_spans = []
    for span in spans:

        if (
            isinstance(span, tuple) and len(span) == 3
            and isinstance(span[0], int)
            and isinstance(span[1], int)
            and isinstance(span[2], str)
        ):
            clean_spans.append(span)
    return clean_spans

            # ------------------------------ #
            #       Soft label creation      #
            # ------------------------------ #

def create_soft_labels(tokens, spans, label_list, label2id, entity_prob=0.8):
    soft_labels = []
    other_prob = (1 - entity_prob) / (len(label_list) - 1)
    for idx in range(len(tokens)):
        probs = [other_prob] * len(label_list)
        matched = False
        for start, end, _ in spans:
            if start <= idx <= end:
                tag = "B" if idx == start else "I"
                probs = [other_prob] * len(label_list)
                probs[label2id[tag]] = entity_prob
                matched = True
                break
        if not matched:
            probs[label2id["O"]] = entity_prob
        soft_labels.append(probs)

    return soft_labels

            # ----------------------------- #
            #       Custom Trainer          #
            # ----------------------------- #

class SoftLabelNERTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        softs = inputs.get("soft_labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_ce = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=-100,
        )
        if softs is not None:
            soft_tensor = torch.tensor(softs, device=logits.device)
            loss_kl = F.kl_div(
                F.log_softmax(logits, dim=-1),
                soft_tensor,
                reduction="batchmean",
            )
            total = loss_ce + 0.5 * loss_kl
        else:
            total = loss_ce
        return (total, outputs) if return_outputs else total

            # --------------------------- #
            #         Load TSV            #
            # --------------------------- #
def load_tsv(path):
    sentences, tags = [], []
    ctoks, ctags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if not l:
                if ctoks:
                    sentences.append(ctoks); tags.append(ctags)
                    ctoks, ctags = [], []
            else:
                parts = l.split("\t")
                if len(parts) != 2:
                    continue
                tok, tag = parts
                ctoks.append(tok); ctags.append(tag)
        if ctoks:
            sentences.append(ctoks); tags.append(ctags)
    return Dataset.from_dict({"token": sentences, "tag": tags})

            # ----------------------- #
            #           Main          #
            # ----------------------- #

def main():
    parser = argparse.ArgumentParser("LLM-guided BioBERT NER")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--dev_file", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--llm_model", default="epfl-llm/meditron-7b")
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = DatasetDict({
        "train": load_tsv(args.train_file).select(range(10)),
        "validation": load_tsv(args.dev_file).select(range(10)),
        "test": load_tsv(args.test_file).select(range(10)),
    })

    all_tags = {t for seq in datasets["train"]["tag"] for t in seq}
    label_list = ([t for t in all_tags if t])
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}

    llm_tok, llm_model = load_llm(args.llm_model, args.device_map)

    for split in ["train", "validation"]:
        softs = []
        for tokens in tqdm(datasets[split]["token"], desc=f"LLM soft labels {split}"):
            spans = get_llm_entity_spans(llm_tok, llm_model, tokens)
            print("Spans:", spans)
            softs.append(create_soft_labels(tokens, spans, label_list, label2id))
        datasets[split] = datasets[split].add_column("soft_labels", softs)

    bert_name = "alvaroalon2/biobert_genetic_ner"
    bert_tok = AutoTokenizer.from_pretrained(bert_name)

    def tokenize_and_align(examples):
        enc = bert_tok(
            examples["token"],
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        labs_batch, soft_batch = [], []
        for i, e in enumerate(enc.encodings):
            wids = e.word_ids
            labs, softs = [], []
            for wid in wids:
                if wid is None:
                    labs.append(-100)
                    softs.append([0.0] * len(label_list))
                else:
                    labs.append(label2id[examples["tag"][i][wid]])
                    softs.append(examples["soft_labels"][i][wid])
            labs_batch.append(labs)
            soft_batch.append(softs)
        enc["labels"] = labs_batch
        enc["soft_labels"] = soft_batch
        return enc

    def tokenize_eval(examples):
        enc = bert_tok(
            examples["token"],
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        labs = []
        for i, e in enumerate(enc.encodings):
            wids = e.word_ids
            row = []
            for wid in wids:
                row.append(-100 if wid is None else label2id[examples["tag"][i][wid]])
            labs.append(row)
        enc["labels"] = labs
        return enc

    tokenized = DatasetDict()
    for split, ds in datasets.items():
        if split in ["train", "validation"]:
            tokenized[split] = ds.map(
                tokenize_and_align,
                batched=True,
                remove_columns=["token", "tag"],
            )
        else:
            tokenized[split] = ds.map(
                tokenize_eval,
                batched=True,
                remove_columns=["token", "tag"],
            )

    model = AutoModelForTokenClassification.from_pretrained(
        bert_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(p):
        logits, labels = p
        preds = np.argmax(logits, axis=-1)
        true_seqs, pred_seqs = [], []
        for t_seq, p_seq in zip(labels, preds):
            tt, pp = [], []
            for t, p in zip(t_seq, p_seq):
                if t == -100: continue
                tt.append(id2label[t]); pp.append(id2label[p])
            true_seqs.append(tt); pred_seqs.append(pp)
        return {
            "precision": precision_score(true_seqs, pred_seqs),
            "recall": recall_score(true_seqs, pred_seqs),
            "f1": f1_score(true_seqs, pred_seqs),
        }

    args_train = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    trainer = SoftLabelNERTrainer(
        model=model,
        args=args_train,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=bert_tok,
        data_collator=DataCollatorForTokenClassification(bert_tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model(args.output_dir)
    bert_tok.save_pretrained(args.output_dir)

    preds, labels, _ = trainer.predict(tokenized["test"])
    pred_ids = np.argmax(preds, axis=-1)
    true_seqs, pred_seqs = [], []
    for t_seq, p_seq in zip(labels, pred_ids):
        tt, pp = [], []
        for t, p in zip(t_seq, p_seq):
            if t == -100: continue
            tt.append(id2label[t]); pp.append(id2label[p])
        true_seqs.append(tt); pred_seqs.append(pp)

    test_metrics = {
        "precision": precision_score(true_seqs, pred_seqs),
        "recall": recall_score(true_seqs, pred_seqs),
        "f1": f1_score(true_seqs, pred_seqs),
        "report": classification_report(true_seqs, pred_seqs, output_dict=True),
    }
    def to_python(obj):
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, dict): return {k: to_python(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_python(v) for v in obj]
        return obj
    test_metrics = to_python(test_metrics)
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    with open(os.path.join(args.output_dir, "test_predictions.tsv"), "w") as out:
        out.write("sentence_idx\ttoken_idx\ttoken\ttrue_tag\tpred_tag\n")
        for i, (t_seq, p_seq, ids) in enumerate(zip(labels, pred_ids, tokenized["test"]["input_ids"])):
            for j, (tid, pid) in enumerate(zip(t_seq, p_seq)):
                if tid == -100: continue
                tok = bert_tok.convert_ids_to_tokens(ids[j])
                out.write(f"{i}\t{j}\t{tok}\t{id2label[tid]}\t{id2label[pid]}\n")
            out.write("\n")

if __name__ == "__main__":
    main()