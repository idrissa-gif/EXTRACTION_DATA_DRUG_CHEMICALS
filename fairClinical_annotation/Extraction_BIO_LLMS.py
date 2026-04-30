# Extraction_BIO_LLMS.py
import os
import json
import argparse
import time
import psutil
import re
import random
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ['USE_TORCH'] = '1'

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

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


# =========================
# Args
# =========================
def parse_args():
    p = argparse.ArgumentParser(
        description="Predict BIO (B/I/O only) with an LLM, using PRECOMPUTED BioBERT priors to guide inference."
    )
    p.add_argument("--model_path", type=str, required=True,
                   help="HF hub id or local path of the instruction LLM (e.g., Mistral/Llama Instruct).")
    p.add_argument("--json_dir", type=str, required=True,
                   help="Directory with BioC JSON files.")

    # Precomputed BioBERT priors
    p.add_argument("--biobert_priors_dir", type=str, required=True,
                   help="Dir containing BioBERT prior TSVs (one per JSON, same stem).")
    p.add_argument("--biobert_prior_suffix", type=str, default="_biobert.tsv",
                   help="Suffix for prior files (default: _biobert.tsv).")
    p.add_argument("--biobert_prior_has_score", action="store_true",
                   help="If set, TSV has a 3rd column with per-token score in [0,1].")
    p.add_argument("--biobert_entity_conf", type=float, default=0.9,
                   help="Default confidence for B/I when no score present.")
    p.add_argument("--biobert_o_conf", type=float, default=0.9,
                   help="Default confidence that tag is O when no score present.")
    p.add_argument("--align_min_coverage", type=float, default=0.2,
                   help="If < this fraction of tokens align with priors, drop priors for that passage.")

    # How priors help
    p.add_argument("--biobert_help_mode", choices=["both", "prompt", "constrain", "off"],
                   default="both", help="Use priors in the prompt, post-hoc constraints, both, or off.")
    p.add_argument("--biobert_threshold", type=float, default=0.85,
                   help="Confidence threshold for constraints (entity_conf = 1 - P(O)).")
    p.add_argument("--constrain_flip_o", action="store_true",
                   help="Allow flipping LLM B/I → O when prior is confidently O.")

    # LLM generation
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="Min max_new_tokens; auto-raised to >= n + headroom.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature; 0 for deterministic.")
    p.add_argument("--trust_remote_code", action="store_true",
                   help="Enable if a repo needs custom code.")

    # Debug / inspection
    p.add_argument("--debug_dump", action="store_true",
                   help="(Legacy) Append per-passage token/prior/LLM triplets to *_debug.txt.")
    p.add_argument("--debug_examples", type=int, default=0,
                   help="Dump up to N passages per JSON with metrics and TOK/PRIOR/LLM table.")
    p.add_argument("--debug_sample", choices=["non_o", "first", "random"], default="non_o",
                   help="Which passages to pick for debug_examples.")
    p.add_argument("--debug_seed", type=int, default=0, help="RNG seed when --debug_sample=random.")
    p.add_argument("--debug_dir", type=str, default=None, help="Directory to write *_debug.txt (default results dir).")
    return p.parse_args()


# =========================
# Tokenization + spans
# =========================
def tokenize_biomedical_text(text):
    # Separate punctuation & split alpha-num boundaries (5mg -> 5 mg, TNF1 -> TNF 1)
    text2 = re.sub(r'([.!?;:,(){}\[\]"\'-])', r' \1 ', text)
    text2 = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text2)
    text2 = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text2)
    text2 = re.sub(r'\s+', ' ', text2).strip()
    return [tok for tok in text2.split(' ') if tok]

def tokens_with_offsets_in_original(text, tokens):
    spans, cursor = [], 0
    for tok in tokens:
        idx = text.find(tok, cursor)
        if idx == -1:
            low_idx = text.lower().find(tok.lower(), cursor)
            if low_idx == -1:
                spans.append((tok, None, None))
                continue
            idx = low_idx
        start, end = idx, idx + len(tok)
        spans.append((tok, start, end))
        cursor = end
    return spans


# =========================
# Strict prompt (with BioBERT prior line)
# =========================
PROMPT_TEMPLATE = (
    "<start> {sentence} <end>\n"
    "Tokens: {tokens}\n"
    "{prior_line}"
    "Task: Tag only CHEMICAL mentions using BIO. "
    "Use 'B' for the first token of a chemical mention, 'I' for subsequent tokens inside the same chemical. "
    "Everything that is NOT a chemical (verbs, prepositions, punctuation, numbers, dates, symptoms, diseases, procedures, generic words like 'treatment', etc.) must be 'O'.\n"
    "Total tokens: {n}\n"
    "Output Exactly {n} BIO Tags, space-separated, no other text:\n"
    "Tags:"
)

def build_prompt_from_words(words, prior_tags=None, use_prior_in_prompt=False):
    sentence = " ".join(words)
    prior_line = f"BioBERT suggests: {' '.join(prior_tags)}\n" if (use_prior_in_prompt and prior_tags) else ""
    return PROMPT_TEMPLATE.format(
        sentence=sentence,
        tokens=sentence,
        prior_line=prior_line,
        n=len(words),
    )

def parse_bio_sequence_strict(generated_text, n_tokens):
    """
    Take only the first n BIO tokens (B/I/O) after 'Tags:'; pad with 'O' if fewer.
    """
    t = generated_text
    i = t.rfind("Tags:")
    if i != -1:
        t = t[i + len("Tags:"):]
    tags = []
    for tok in re.findall(r"[A-Za-z]+", t):
        u = tok.upper()
        if u in ("B", "I", "O"):
            tags.append(u)
            if len(tags) == n_tokens:
                break
    if len(tags) < n_tokens:
        tags += ["O"] * (n_tokens - len(tags))
    return tags

def llm_predict_bio_tags(gen_pipe, words, min_max_new_tokens=512, temperature=0.0,
                         prior_tags=None, use_prior_in_prompt=False):
    prompt = build_prompt_from_words(words, prior_tags, use_prior_in_prompt)
    n = len(words)
    needed = max(n + 8, min_max_new_tokens)  # small headroom
    out = gen_pipe(
        prompt,
        max_new_tokens=needed,
        do_sample=(temperature > 0),
        temperature=temperature,
        return_full_text=False,
        truncation=False,
        eos_token_id=gen_pipe.tokenizer.eos_token_id
    )[0]["generated_text"]
    return parse_bio_sequence_strict(out, n)


# =========================
# Read precomputed priors
# =========================
def find_priors_file(priors_dir: Path, json_stem: str, suffix: str):
    candidates = [
        priors_dir / f"{json_stem}{suffix}",
        priors_dir / f"{json_stem}.tsv",
        priors_dir / f"{json_stem}_priors.tsv",
        priors_dir / f"{json_stem}.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def read_biobert_priors_file(path: Path, has_score: bool):
    passages = []
    toks, tags, scores = [], [], [] if has_score else None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if toks:
                    passages.append({"tokens": toks, "tags": tags, "scores": scores[:] if has_score else None})
                    toks, tags = [], []
                    if has_score:
                        scores = []
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            tok, tag = parts[0], parts[1].strip().upper()
            toks.append(tok)
            tags.append("B" if tag.startswith("B") else ("I" if tag.startswith("I") else "O"))
            if has_score:
                if len(parts) >= 3:
                    try:
                        scores.append(float(parts[2]))
                    except:
                        scores.append(None)
                else:
                    scores.append(None)
    if toks:
        passages.append({"tokens": toks, "tags": tags, "scores": scores[:] if has_score else None})
    return passages

def align_priors_to_words(
    text,
    words,
    prior_tokens,
    prior_tags,
    prior_scores,
    default_entity_conf=0.9,
    default_o_conf=0.9,
    min_coverage=0.2
):
    """
    Align prior BIO tags (and optional scores) to our tokens via char-span overlap in this passage text.
    Returns:
      prior_tags_aligned: list[str] of B/I/O  OR None (if coverage too low)
      entity_conf: list[float] = (1 - P(O)) per token
      o_prob: list[float] = P(O) per token
      coverage_ratio: float
    """
    our_spans = tokens_with_offsets_in_original(text, words)
    prior_spans = tokens_with_offsets_in_original(text, prior_tokens)

    aligned_tags, entity_conf, o_prob = [], [], []
    coverage = 0

    for (_, s, e) in our_spans:
        if s is None or e is None or not prior_spans:
            aligned_tags.append("O")
            entity_conf.append(1.0 - default_o_conf)
            o_prob.append(default_o_conf)
            continue

        best_j, best_ov = None, 0
        for j, (_, ps, pe) in enumerate(prior_spans):
            if ps is None or pe is None:
                continue
            ov = max(0, min(e, pe) - max(s, ps))
            if ov > best_ov:
                best_ov, best_j = ov, j

        if best_j is None or best_ov == 0:
            aligned_tags.append("O")
            entity_conf.append(1.0 - default_o_conf)
            o_prob.append(default_o_conf)
            continue

        coverage += 1
        tag = prior_tags[best_j]
        aligned_tags.append(tag)

        # Confidence
        if prior_scores is not None:
            sc = prior_scores[best_j]
            if sc is None:
                if tag in ("B", "I"):
                    entity_conf.append(default_entity_conf)
                    o_prob.append(1.0 - default_entity_conf)
                else:
                    entity_conf.append(1.0 - default_o_conf)
                    o_prob.append(default_o_conf)
            else:
                # If tag == O: treat score as P(O); else as entity confidence.
                if tag == "O":
                    o = float(max(0.0, min(1.0, sc)))
                    o_prob.append(o)
                    entity_conf.append(1.0 - o)
                else:
                    ent = float(max(0.0, min(1.0, sc)))
                    entity_conf.append(ent)
                    o_prob.append(1.0 - ent)
        else:
            if tag in ("B", "I"):
                entity_conf.append(default_entity_conf)
                o_prob.append(1.0 - default_entity_conf)
            else:
                entity_conf.append(1.0 - default_o_conf)
                o_prob.append(default_o_conf)

    cov_ratio = (coverage / len(words)) if words else 0.0
    if len(words) > 0 and cov_ratio < min_coverage:
        return None, None, None, cov_ratio

    return aligned_tags, entity_conf, o_prob, cov_ratio


# =========================
# Constraints & heuristics
# =========================
def apply_biobert_constraints(llm_tags, prior_tags, entity_conf, o_prob, threshold=0.85, flip_o=False):
    """
    Nudge LLM tags:
      - If prior says entity with high conf and LLM says O → adopt prior tag
      - Optionally, if prior says O with high conf and LLM says entity → flip to O
      - Repair orphan I's
    """
    if prior_tags is None:
        # BIO repair only
        tags = llm_tags[:]
        for i in range(len(tags)):
            if tags[i] == "I" and (i == 0 or tags[i-1] == "O"):
                tags[i] = "B"
        return tags

    n = len(llm_tags)
    tags = llm_tags[:]
    for i in range(n):
        pt, lt = prior_tags[i], tags[i]
        conf = entity_conf[i] if entity_conf is not None else 0.0
        op = o_prob[i] if o_prob is not None else 0.0

        if pt in ("B", "I") and conf >= threshold and lt == "O":
            tags[i] = pt
            continue
        if flip_o and pt == "O" and op >= threshold and lt in ("B", "I"):
            tags[i] = "O"

    # BIO repair
    for i in range(n):
        if tags[i] == "I" and (i == 0 or tags[i-1] == "O"):
            tags[i] = "B"
    return tags

PUNCT_RE = re.compile(r'^[\W_]+$')
DIGIT_RE = re.compile(r'^\d+([.,]\d+)?$')
STOPWORDS = {
    "a","an","the","of","for","to","in","on","by","with","and","or","as","at","from",
    "was","were","is","are","be","been","being","this","that","these","those","it","its",
    "their","there","then","than","which","who","whom","whose","we","they","he","she","you","i",
    # domain-ish non-chem commons (extend as needed):
    "treatment","initial","chronic","compared","plus","conference","increase","activity",
    "immediately","compared","versus","vs","during","after","before","between","against",
    "patients","study","trial","dose","doses","weekly","daily","month","months","year","years"
}

def looks_like_chemical(tok):
    """
    Very permissive hints for chemical-looking tokens; prevents over-flipping to O.
    """
    t = tok.lower()
    # common drug/metabolite substrings
    chemish = (
        "ribavirin","interferon","peginterferon","alpha","beta","acetyl","methyl","sulf","chlor",
        "amine","azole","cyclo","statin","cillin","avir","dopa","phenyl","sodium","potassium","chloride",
        "hydrochloride","phosphate","sulfate","nitrate","oxide","ethanol","methanol","acetate","lactate"
    )
    if any(s in t for s in chemish):
        return True
    # alphanum patterns
    if re.search(r'[a-zA-Z]\d', t) or re.search(r'\d[a-zA-Z]', t):
        return True
    # hyphenated chemical-ish tokens
    if '-' in t and any(part.isalpha() for part in t.split('-')):
        return True
    return False

def prefer_biobert_o_and_fix_I(words, tags, prior_tags, o_prob, threshold=0.85):
    """
    Force O where BioBERT says O with high confidence; also repair orphan I→B.
    """
    if prior_tags is not None and o_prob is not None:
        for i in range(len(tags)):
            op = o_prob[i]
            if prior_tags[i] == "O" and op is not None and op >= threshold:
                tags[i] = "O"
    # BIO repair
    for i in range(len(tags)):
        if tags[i] == "I" and (i == 0 or tags[i-1] == "O"):
            tags[i] = "B"
    return tags

def force_obvious_O(words, tags, prior_tags=None, o_prob=None, strong_o=0.85):
    """
    Force O for punctuation, numbers, and stopwords unless BioBERT strongly suggests NOT O.
    """
    for i, w in enumerate(words):
        wl = w.lower()
        obvious_non_entity = (PUNCT_RE.match(w) is not None) or DIGIT_RE.match(w) or (wl in STOPWORDS)
        if obvious_non_entity and not looks_like_chemical(w):
            allow_flip = True
            if prior_tags is not None and o_prob is not None and i < len(o_prob):
                if o_prob[i] is not None and o_prob[i] < (1.0 - strong_o):
                    allow_flip = False  # BioBERT confident non-O → don't force O
            if allow_flip:
                tags[i] = "O"
    # BIO repair again
    for i in range(len(tags)):
        if tags[i] == "I" and (i == 0 or tags[i-1] == "O"):
            tags[i] = "B"
    return tags


# =========================
# Simple debug metrics
# =========================
def entities_from_bio(words, tags):
    """Return list of entity strings from BIO tags over words."""
    ents = []
    cur = []
    for w, t in zip(words, tags):
        if t == "B":
            if cur:
                ents.append(" ".join(cur))
                cur = []
            cur = [w]
        elif t == "I":
            if cur:
                cur.append(w)
            else:
                cur = [w]  # orphan I -> start new
        else:  # O
            if cur:
                ents.append(" ".join(cur))
                cur = []
    if cur:
        ents.append(" ".join(cur))
    return ents

def token_metrics_vs_prior(llm_tags, prior_tags):
    """Treat entity = (B or I). Return token-level precision/recall/F1 using priors as proxy."""
    if prior_tags is None:
        return None
    llm_ent = [1 if t in ("B","I") else 0 for t in llm_tags]
    pr_ent  = [1 if t in ("B","I") else 0 for t in prior_tags]
    tp = sum(1 for a,b in zip(llm_ent, pr_ent) if a==1 and b==1)
    fp = sum(1 for a,b in zip(llm_ent, pr_ent) if a==1 and b==0)
    fn = sum(1 for a,b in zip(llm_ent, pr_ent) if a==0 and b==1)
    prec = tp / (tp + fp) if (tp+fp)>0 else 0.0
    rec  = tp / (tp + fn) if (tp+fn)>0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
    return {"tp":tp, "fp":fp, "fn":fn, "precision":round(prec,3), "recall":round(rec,3), "f1":round(f1,3)}


# =========================
# GPU mem helper
# =========================
def get_gpu_memory():
    if not GPU_AVAILABLE:
        return None
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()
    return info.used / 1024 ** 2


# =========================
# Main
# =========================
def main():
    args = parse_args()
    process = psutil.Process(os.getpid()) if args.debug_sample == "random":
        random.seed(args.debug_seed or 0)

    json_dir = Path(args.json_dir)
    priors_dir = Path(args.biobert_priors_dir)
    result_dir = json_dir.parent / f"{json_dir.name}_results_llm_bio"
    result_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.debug_dir) if args.debug_dir else result_dir
    debug_dir.mkdir(parents=True, exist_ok=True)

    log_path = result_dir / "usage_log.txt"

    with open(log_path, "w", encoding="utf-8") as log:
        log.write("================ LLM + Precomputed BioBERT (B/I/O) ================\n")

        t0 = time.time()
        use_gpu = torch.cuda.is_available()

        # Load LLM
        tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=args.trust_remote_code)
        if tok.eos_token is None:
            tok.eos_token = tok.pad_token or "</s>"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        if use_gpu:
            dtype = getattr(torch, "bfloat16", None) or getattr(torch, "float16", None)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=args.trust_remote_code
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=torch.float32, device_map=None, trust_remote_code=args.trust_remote_code
            )

        if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
            model.config.pad_token_id = tok.pad_token_id

        # IMPORTANT: do NOT pass device when model uses device_map="auto"
        gen = pipeline("text-generation", model=model, tokenizer=tok)

        print(f"LLM device: {'GPU(auto)' if use_gpu else 'CPU'}")
        model_load_time = time.time() - t0
        mem_used = process.memory_info().rss / 1024 ** 2
        gpu_used = get_gpu_memory()
        log.write(f"Model load time: {model_load_time:.2f} s\n")
        log.write(f"Memory after model load: {mem_used:.2f} MB\n")
        if gpu_used is not None:
            log.write(f"GPU memory after model load: {gpu_used:.2f} MB\n")

        for json_file in sorted(json_dir.glob("*.json")):
            file_start = time.time()
            with open(json_file, "r", encoding="utf-8") as f:
                bioc = json.load(f)

            # locate priors file
            priors_path = find_priors_file(priors_dir, json_file.stem, args.biobert_prior_suffix)
            if priors_path is None:
                print(f"[WARN] No BioBERT priors found for {json_file.name} (looking for *{args.biobert_prior_suffix}). Proceeding without priors.")
                priors_passages = []
            else:
                priors_passages = read_biobert_priors_file(priors_path, has_score=args.biobert_prior_has_score)

            prior_idx = 0  # iterate priors passages in order

            # Standard TSV with header
            tsv_lines = ["Token\tBIO-Tag"]

            # Debug file per source JSON
            dbg_path = debug_dir / f"{json_file.stem}_debug.txt"
            if (args.debug_dump or args.debug_examples > 0) and dbg_path.exists():
                dbg_path.unlink()

            dumped = 0  # how many passages we have dumped for this JSON

            for doc in bioc.get("documents", []):
                for passage in doc.get("passages", []):
                    text = passage.get("text", "") or ""
                    if not text.strip():
                        tsv_lines.append("")
                        continue

                    words = tokenize_biomedical_text(text)
                    if not words:
                        tsv_lines.append("")
                        continue

                    # pull priors for this passage if available
                    prior_tags_aligned = entity_conf = o_prob = None
                    cov_ratio = None
                    if prior_idx < len(priors_passages) and args.biobert_help_mode in ("both", "prompt", "constrain"):
                        pp = priors_passages[prior_idx]
                        prior_idx += 1
                        prior_tokens, prior_tags, prior_scores = pp["tokens"], pp["tags"], pp["scores"]
                        prior_tags_aligned, entity_conf, o_prob, cov_ratio = align_priors_to_words(
                            text, words, prior_tokens, prior_tags, prior_scores,
                            default_entity_conf=args.biobert_entity_conf,
                            default_o_conf=args.biobert_o_conf,
                            min_coverage=args.align_min_coverage
                        )
                        if prior_tags_aligned is None:
                            with open(log_path, "a", encoding="utf-8") as log2:
                                log2.write(f"[INFO] Dropped priors for passage (coverage {cov_ratio:.3f} < {args.align_min_coverage}).\n")

                    # LLM prediction (optionally with prior in prompt)
                    use_prior_in_prompt = (prior_tags_aligned is not None) and (args.biobert_help_mode in ("both", "prompt"))
                    llm_tags = llm_predict_bio_tags(
                        gen,
                        words,
                        min_max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        prior_tags=prior_tags_aligned,
                        use_prior_in_prompt=use_prior_in_prompt
                    )

                    # Optional constraint from priors
                    if prior_tags_aligned is not None and args.biobert_help_mode in ("both", "constrain"):
                        llm_tags = apply_biobert_constraints(
                            llm_tags, prior_tags_aligned, entity_conf, o_prob,
                            threshold=args.biobert_threshold, flip_o=args.constrain_flip_o
                        )

                    # Prefer BioBERT-O where confident & fix I->B at starts
                    llm_tags = prefer_biobert_o_and_fix_I(
                        words, llm_tags, prior_tags_aligned, o_prob,
                        threshold=max(0.75, args.biobert_threshold)
                    )

                    # Heuristic cleanup (punct/nums/stopwords → O, unless strongly non-O)
                    llm_tags = force_obvious_O(
                        words, llm_tags, prior_tags_aligned, o_prob,
                        strong_o=max(0.80, args.biobert_threshold)
                    )

                    # Emit outputs
                    for tok_word, tag in zip(words, llm_tags):
                        tsv_lines.append(f"{tok_word}\t{tag}")
                    tsv_lines.append("")

                    # Decide whether to dump this passage to debug
                    want_dump = False
                    if args.debug_examples > 0 and dumped < args.debug_examples:
                        if args.debug_sample == "first":
                            want_dump = True
                        elif args.debug_sample == "non_o":
                            want_dump = any(t in ("B","I") for t in llm_tags)
                        else:  # random
                            want_dump = random.random() < 0.25  # ~25% chance
                    if args.debug_dump:
                        want_dump = True  # legacy: dump everything

                    if want_dump:
                        ents = entities_from_bio(words, llm_tags)
                        non_o_llm = sum(1 for t in llm_tags if t != "O")
                        metrics = token_metrics_vs_prior(llm_tags, prior_tags_aligned)
                        with open(dbg_path, "a", encoding="utf-8") as df:
                            df.write("== PASSAGE DEBUG ==\n")
                            if cov_ratio is not None:
                                df.write(f"prior_coverage={cov_ratio:.3f}  ")
                            df.write(f"tokens={len(words)}  llm_nonO={non_o_llm}  pred_entities={len(ents)}\n")
                            if ents:
                                df.write("pred_entity_strings: " + " | ".join(ents[:10]) + ("\n" if len(ents)<=10 else " ...\n"))
                            if metrics:
                                df.write(f"vs_prior (token-entity): P={metrics['precision']} R={metrics['recall']} F1={metrics['f1']}  (tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']})\n")
                            df.write("\nTOK\tPRIOR\tLLM\n")
                            for i, w in enumerate(words):
                                pt = prior_tags_aligned[i] if prior_tags_aligned is not None else "-"
                                df.write(f"{w}\t{pt}\t{llm_tags[i]}\n")
                            df.write("\n")
                        dumped += 1

            # Write files
            out_file = result_dir / f"{json_file.stem}_annotated.tsv"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write("\n".join(tsv_lines))

            file_time = time.time() - file_start
            file_mem = process.memory_info().rss / 1024 ** 2
            file_gpu = get_gpu_memory()
            with open(log_path, "a", encoding="utf-8") as log2:
                log2.write(f"\nFile: {json_file.name}\n")
                log2.write(f"Time: {file_time:.2f} s\n")
                log2.write(f"Memory: {file_mem:.2f} MB\n")
                if file_gpu is not None:
                    log2.write(f"GPU memory: {file_gpu:.2f} MB\n")

            print(f"Annotated {json_file.name} -> {out_file.name}")
            if (args.debug_dump or args.debug_examples > 0) and dumped:
                print(f"  Debug -> {dbg_path} (dumped {dumped} passage(s))")

        print("Done.")


if __name__ == "__main__":
    main()
