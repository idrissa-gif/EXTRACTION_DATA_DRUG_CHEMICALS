import argparse
import json
import difflib
from collections import Counter
from prettytable import PrettyTable

def normalize(text):
    """Normalize text for fairer comparison (lowercase, strip)."""
    return str(text).lower().strip()

def classify_error(truth, pred):
    """
    Classifies the mismatch between a single ground truth entity and a prediction.
    Returns: 'exact', 'partial', 'completely_wrong'
    """
    t_norm = normalize(truth)
    p_norm = normalize(pred)
    
    if t_norm == p_norm:
        return "exact"
    
    # Check for substring overlap (Partial match)
    if t_norm in p_norm or p_norm in t_norm:
        return "partial"
        
    # Check for high similarity (Typo or slight variation)
    similarity = difflib.SequenceMatcher(None, t_norm, p_norm).ratio()
    if similarity > 0.8:
        return "partial"
        
    return "completely_wrong"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", default="results_finetuned.jsonl", help="Path to your high-score results file")
    parser.add_argument("--output_report", default="error_report.txt")
    args = parser.parse_args()

    print(f"Analyzing {args.results_file}...")
    
    # Statistics Containers
    stats = {
        "total_docs": 0,
        "total_gold_entities": 0,
        "total_pred_entities": 0,
        "exact_matches": 0,
        "false_positives": 0,  # Model predicted it, but it wasn't in Truth
        "false_negatives": 0,  # In Truth, but model missed it
    }
    
    fp_examples = [] # Store specific FP strings
    fn_examples = [] # Store specific FN strings
    partial_matches = [] # Store tuples of (Truth, Pred)
    
    with open(args.results_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                # handle keys depending on which script created the file
                ground_truth = item.get("ground_truth", item.get("output", []))
                prediction = item.get("prediction", item.get("prediction_parsed", []))
                
                # Normalize lists for comparison
                gt_counter = Counter([normalize(x) for x in ground_truth])
                pred_counter = Counter([normalize(x) for x in prediction])
                
                stats["total_docs"] += 1
                stats["total_gold_entities"] += sum(gt_counter.values())
                stats["total_pred_entities"] += sum(pred_counter.values())

                # 1. Calculate Intersection (True Positives)
                # We use overlap of counts to handle duplicates correctly
                tp_iter = gt_counter & pred_counter
                tp_count = sum(tp_iter.values())
                stats["exact_matches"] += tp_count
                
                # 2. Calculate False Negatives (Missed)
                fn_iter = gt_counter - pred_counter
                for chemical, count in fn_iter.items():
                    stats["false_negatives"] += count
                    fn_examples.extend([chemical] * count)

                # 3. Calculate False Positives (Hallucinations)
                fp_iter = pred_counter - gt_counter
                for chemical, count in fp_iter.items():
                    stats["false_positives"] += count
                    fp_examples.extend([chemical] * count)
                
                # 4. Advanced: Detect Partial Matches
                # We look at the 'leftover' FPs and FNs to see if they overlap
                leftover_fns = list(fn_iter.elements())
                leftover_fps = list(fp_iter.elements())
                
                for fn in leftover_fns:
                    for fp in leftover_fps:
                        if fn in fp or fp in fn or difflib.SequenceMatcher(None, fn, fp).ratio() > 0.8:
                            partial_matches.append((fn, fp))
                            break # Assume 1-to-1 mapping for simplicity

            except json.JSONDecodeError:
                continue

    # --- GENERATE REPORT ---
    report = []
    report.append("="*40)
    report.append("ERROR ANALYSIS REPORT")
    report.append("="*40)
    
    # 1. High Level Stats
    report.append("\n1. GLOBAL STATISTICS")
    report.append(f"Total Documents:      {stats['total_docs']}")
    report.append(f"Total Gold Entities:  {stats['total_gold_entities']}")
    report.append(f"Total Pred Entities:  {stats['total_pred_entities']}")
    report.append("-" * 20)
    report.append(f"Correct (TP):         {stats['exact_matches']} ({stats['exact_matches']/stats['total_gold_entities']*100:.1f}%)")
    report.append(f"Missed (FN):          {stats['false_negatives']} ({stats['false_negatives']/stats['total_gold_entities']*100:.1f}%)")
    report.append(f"Hallucinated (FP):    {stats['false_positives']}")
    
    # 2. Top Missed Chemicals (FN)
    report.append("\n2. MOST FREQUENTLY MISSED CHEMICALS (False Negatives)")
    report.append("These are hard for the model to find.")
    t = PrettyTable(['Chemical', 'Count'])
    for chem, count in Counter(fn_examples).most_common(10):
        t.add_row([chem, count])
    report.append(str(t))

    # 3. Top Hallucinations (FP)
    report.append("\n3. MOST FREQUENT HALLUCINATIONS (False Positives)")
    report.append("The model thinks these are chemicals, but they aren't labeled as such.")
    t = PrettyTable(['Predicted String', 'Count'])
    for chem, count in Counter(fp_examples).most_common(10):
        t.add_row([chem, count])
    report.append(str(t))
    
    # 4. Boundary Errors
    report.append("\n4. BOUNDARY / PARTIAL ERRORS")
    report.append("The model found the chemical, but the text boundary was slightly off.")
    t = PrettyTable(['Ground Truth', 'Prediction', 'Type'])
    # Show unique boundary errors
    unique_partials = list(set(partial_matches))[:15]
    for truth, pred in unique_partials:
        if truth in pred:
            err_type = "Predicted text too long"
        elif pred in truth:
            err_type = "Predicted text too short"
        else:
            err_type = "Spelling/Format"
        t.add_row([truth, pred, err_type])
    report.append(str(t))

    # Print and Save
    final_report = "\n".join(report)
    print(final_report)
    
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(final_report)
    print(f"\nReport saved to {args.output_report}")

if __name__ == "__main__":
    main()
