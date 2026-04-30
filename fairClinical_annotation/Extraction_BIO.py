import os
import json
import argparse
import time
import psutil
import re
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ['USE_TORCH'] = '1'

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

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

def parse_args():
    parser = argparse.ArgumentParser(description="Run BIO-tag NER on BioC JSON files with resource tracking.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--json_dir", type=str, required=True, help="Directory with BioC Json files.")
    return parser.parse_args()

def tokenize_biomedical_text(text):
    """
    Tokenize biomedical text properly, separating punctuation and handling common patterns.
    This should match the tokenization used during training.
    """
    # First, handle common biomedical patterns and punctuation
    text = re.sub(r'([.!?;:,(){}[\]"\'-])', r' \1 ', text)  # Separate punctuation
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)  # Separate numbers from letters (e.g., "5mg" -> "5 mg")
    text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)  # Separate letters from numbers
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

    # Split and filter empty tokens
    tokens = [token.strip() for token in text.split() if token.strip()]
    return tokens

def annotate_text_bio(text, tokenizer, ner):
    # Use proper biomedical tokenization
    words = tokenize_biomedical_text(text)
    if not words:
        return []

    # Tokenize the words properly
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True
    )

    word_ids = encoded.word_ids()

    # Use the NER pipeline on the original text
    ner_results = ner(text)

    # Initialize tags for all words
    word_tags = ["O"] * len(words)

    # Map NER results back to word-level tags
    for result in ner_results:
        print(f"NER Result: {result}")  # Debug output

        # Get the token index (subtract 1 because pipeline uses 1-based indexing)
        token_index = result["index"]

        # Make sure token_index is within bounds
        if 0 <= token_index < len(word_ids):
            word_idx = word_ids[token_index]

            # If this token corresponds to a word (not special token)
            if word_idx is not None and word_idx < len(word_tags):
                current_tag = result['entity']


                if word_tags[word_idx] == "O" or current_tag.startswith('B'):
                    word_tags[word_idx] = current_tag

    return list(zip(words, word_tags))

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
    process = psutil.Process(os.getpid())

    json_dir = Path(args.json_dir)
    result_dir = json_dir.parent / f"{args.json_dir}_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    log_path = result_dir / "usage_log.txt"

    with open(log_path, "w") as log:
        log.write("============================ Resource Usage Log ============================\n")

        t0 = time.time()
        import torch

        # Check if CUDA is available and set device
        device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForTokenClassification.from_pretrained(args.model_path)
        ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="none", device=device)

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

            tsv_lines = ["Token\tBIO-Tag"]

            for doc in bioc['documents']:
                for passage in doc['passages']:
                    toks_tags = annotate_text_bio(passage["text"], tokenizer, ner)
                    for token, tag in toks_tags:
                        tsv_lines.append(f"{token}\t{tag}")
                    tsv_lines.append("")  # Empty line between passages

            output_file = result_dir / f"{json_file.stem}_annotated.tsv"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(tsv_lines))

            file_time = time.time() - file_start
            file_mem = process.memory_info().rss / 1024**2
            file_gpu = get_gpu_memory()

            log.write(f"\nFile: {json_file.name}\n")
            log.write(f"Time: {file_time:.2f} s\n")
            log.write(f"Memory: {file_mem:.2f} MB\n")
            if file_gpu is not None:
                log.write(f"GPU memory: {file_gpu:.2f} MB\n")

            print(f"Annotated {json_file.name} -> {output_file.name}")

if __name__ == "__main__":
    main()