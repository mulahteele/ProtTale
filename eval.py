#Eval for the testing data result
import json
import pandas as pd
import pickle
from pathlib import Path
import os
import ast
import statistics as stats
import random
from evals.Ontology_tools.Onto_extraction import process_texts_with_api, process_texts_with_ollama
from evals.Ontology_tools.InfoAccretion import compute_InfoAccretion_distance
from evals.Ontology_tools.wang_similarity import compute_wang_similarity
from evals.Ontology_tools.jaccard_similarity import compute_jaccard_similarity
from evals.Ontology_tools.Onto_extraction import process_texts_with_api, process_texts_with_ollama
from evals.linguistic_evaluation import compute_linguistic_metrics
import argparse
import os
import sys
import pickle
import csv
import random

def load_or_process(file_path, data, data_name, safe_model_type):
    """
    Load data from a cached file if available; otherwise, process and save it.
    """
    try:
        with open(file_path, 'rb') as file:
            print(f"Loading {data_name} GO terms from {file_path} processing by {safe_model_type}...")
            return pickle.load(file)
    except FileNotFoundError:
        print(f"File not found. Processing {data_name} using {safe_model_type}...")
        if safe_model_type.lower().startswith("gpt"):
            result = process_texts_with_api(data, safe_model_type)
            # result = process_texts_with_ontogpt(data, safe_model_type)
        else:
            result = process_texts_with_ollama(data, safe_model_type)
        with open(file_path, 'wb') as file:
            pickle.dump(result, file)
        print(f"Processed {data_name} GO terms saved to {file_path}.")
        return result


# Function to filter and return only existing GO terms from the example list in a list of lists format
def filter_existing_go_terms(go_list, extracted_go_terms):
    extracted_ids = {term[0] for term in extracted_go_terms}
    filtered_terms = [[go_id for go_id in go_set if go_id in extracted_ids] for go_set in go_list]
    return filtered_terms


# Function to read GO terms from the saved CSV file
def read_go_terms_from_csv(file_path):
    go_terms = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            go_terms.append(row)
    return go_terms


# ---------- Utilities ----------
def _is_nested_list(x):
    """Return True if x is a list of lists/tuples/sets (or empty list)."""
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], (list, tuple, set)))

def empty_rate_nested(xs):
    """
    For nested list[list[str]]: proportion of empty inner lists.
    For flat list[str]: 0.0 if non-empty else 1.0.
    """
    if not isinstance(xs, list):
        return float("nan")
    if _is_nested_list(xs):
        n = len(xs) if xs else 1
        return sum(1 for v in xs if not v) / n
    return 0.0 if xs else 1.0

def build_joint_nonempty_mask(pred_list, ref_list):
    """
    Build a boolean mask keeping indices where BOTH pred_list[i] and ref_list[i] are non-empty.
    Assumes nested list[list[str]]. If lengths differ, truncate to the min length.
    """
    if not (_is_nested_list(pred_list) and _is_nested_list(ref_list)):
        # Treat as single 'sample'
        return [bool(pred_list) and bool(ref_list)]
    n = min(len(pred_list), len(ref_list))
    if len(pred_list) != len(ref_list):
        print(f"[WARN] Different lengths: pred={len(pred_list)} ref={len(ref_list)}; truncating to {n}.")
    return [bool(pred_list[i]) and bool(ref_list[i]) for i in range(n)]

def filter_parallel_by_mask(seq_list, mask):
    """
    Apply a boolean mask to each sequence list in seq_list in parallel.
    Each element must be nested list[list[str]] aligned by index.
    Returns the filtered lists in the same order.
    """
    out = []
    for seq in seq_list:
        if not _is_nested_list(seq):
            # Enforce nested form for safety
            raise ValueError("Expected nested list[list[str]] for masking.")
        out.append([v for v, m in zip(seq, mask) if m])
    return out






def compute_ontology_metrics(predictions, references, all_GO, model_type, is_pretrained):
    """
    Compute ontology metrics for given predictions and references.
    
    Args:
        predictions (list of str): List of predicted ontology texts.
        references (list of str): List of reference ontology texts.
    
    Returns:
        dict: A dictionary containing Wang Semantic Similarity and InfoAccretion Distance metrics.
    """
    # Create the 'cached' folder if it doesn't exist
    if not os.path.exists('saved_results'):
        os.makedirs('saved_results')

    # File paths for cached data
    safe_model_type = model_type.split("/")[-1]
    reference_file = os.path.join("saved_results", f"reference_go_terms_{safe_model_type}.pkl")
    ia_path = "evals/Ontology_tools/IA.txt"

    ###_____________________________Read and Process the data_____________________________###

    print("Processing reference...")
    reference_go_terms = load_or_process(reference_file, references, "reference", safe_model_type)

    print(f"Processing predictions using {safe_model_type}...")

    if is_pretrained:
        prediction_file = os.path.join("saved_results", f"prediction_go_terms_{safe_model_type}.pkl")

    else:
        prediction_file = os.path.join("saved_results", f"stage2_prediction_go_terms_{safe_model_type}.pkl")

    go_file = os.path.join("saved_results", f"go_terms_{safe_model_type}.pkl")
    if not os.path.exists(go_file):
        with open(go_file, 'wb') as file:
            pickle.dump(all_GO, file)

    predicted_go_terms = load_or_process(prediction_file, predictions, "predictions", safe_model_type)



    # ---------- Read & namespace filtering you already have ----------
    mf_go_terms_file = "evals/Ontology_tools/molecular_function_go_terms.csv"
    MF_go_terms = read_go_terms_from_csv(mf_go_terms_file)
    print(f"Extracted {len(MF_go_terms)} GO terms under molecular_function namespace.")

    all_GO = filter_existing_go_terms(all_GO, MF_go_terms)
    reference_go_terms = filter_existing_go_terms(reference_go_terms, MF_go_terms)
    predicted_go_terms = filter_existing_go_terms(predicted_go_terms, MF_go_terms)

    # print(f"GO terms from all_GO: {all_GO}")
    # print(f"GO terms from reference_go_terms: {reference_go_terms}")
    # print(f"GO terms from predicted_go_terms: {predicted_go_terms}")



    # ---------- 1) Empty rates BEFORE any filtering ----------
    pred_empty_rate = empty_rate_nested(predicted_go_terms)
    ref_empty_rate  = empty_rate_nested(reference_go_terms)
    print(f"Empty rate (predicted_go_terms): {pred_empty_rate:.3%}")
    print(f"Empty rate (reference_go_terms): {ref_empty_rate:.3%}")

    # (Optional) If you also want all_GO empty rate:
    # all_empty_rate = empty_rate_nested(all_GO)
    # print(f"Empty rate (all_GO): {all_empty_rate:.3%}")

    # ---------- 2) Joint filtering: keep cases where BOTH are non-empty ----------
    mask_joint = build_joint_nonempty_mask(predicted_go_terms, reference_go_terms)

    all_GO_joint, predicted_go_terms_joint, reference_go_terms_joint = filter_parallel_by_mask(
        [all_GO, predicted_go_terms, reference_go_terms], mask_joint
    )

    # Diagnostics
    kept = len(predicted_go_terms_joint)
    total = len(mask_joint)
    print(f"Kept pairs (both non-empty): {kept}/{total} ({(kept/total if total else 0.0):.1%})")

    # ---------- 3) Metrics on jointly filtered sets ----------
    print("Calculating current model result (predicted vs all_GO) with joint filtering ...")
    mean_similarity_uncover_all = compute_wang_similarity(all_GO_joint, predicted_go_terms_joint)
    s_2_uncover_all             = compute_InfoAccretion_distance(all_GO_joint, predicted_go_terms_joint, ia_file=ia_path, k=2)
    jaccard_uncover_similarity  = compute_jaccard_similarity(all_GO_joint, predicted_go_terms_joint)

    print("Calculating reference result (reference vs all_GO) with joint filtering ...")
    mean_similarity_ontogpt_all = compute_wang_similarity(all_GO_joint, reference_go_terms_joint)
    s_2_ontogpt_all             = compute_InfoAccretion_distance(all_GO_joint, reference_go_terms_joint, ia_file=ia_path, k=2)
    jaccard_ontogpt_similarity = compute_jaccard_similarity(all_GO_joint, reference_go_terms_joint)


    # predicted_list = []
    # for i in range(len(all_GO)):
    #     predicted_list.append(compute_wang_similarity([all_GO[i]], [predicted_go_terms[i]]))

    
    # df = pd.DataFrame({
    #     "index": range(len(predicted_list)),
    #     "wang_similarity": predicted_list,
    # })
    # df.to_csv("wang_scores.csv", index=False)
   # ====================== BEGIN: Random Baseline ======================
    # Shuffle the pairing between all_GO_joint and predicted_go_terms_joint
    # while keeping predicted_go_terms_joint order fixed.
    n_shuffles = 10
    rng = random.Random(42)  # fixed seed for reproducibility

    wang_rand_list   = []
    ia2_rand_list    = []
    jacc_rand_list   = []

    n_pairs = len(all_GO_joint)
    idx_base = list(range(n_pairs))

    if n_pairs == 0:
        print("[WARN] No pairs to shuffle for random baseline.")
    else:
        for _ in range(n_shuffles):
            # create a random permutation of indices
            idx_perm = idx_base[:] 
            rng.shuffle(idx_perm)

            # permute all_GO_joint only; keep predictions fixed
            all_GO_shuf = [all_GO_joint[i] for i in idx_perm]

            # compute metrics for the shuffled pairing
            wang_rand   = compute_wang_similarity(all_GO_shuf, predicted_go_terms_joint)
            ia2_rand    = compute_InfoAccretion_distance(all_GO_shuf, predicted_go_terms_joint, ia_file=ia_path, k=2)
            jacc_rand   = compute_jaccard_similarity(all_GO_shuf, predicted_go_terms_joint)

            wang_rand_list.append(wang_rand)
            ia2_rand_list.append(ia2_rand)
            jacc_rand_list.append(jacc_rand)

    # Aggregate random baseline (mean ± std)
    wang_rand_mean = round(stats.mean(wang_rand_list), 4) if wang_rand_list else float("nan")
    wang_rand_std  = round(stats.pstdev(wang_rand_list), 4) if len(wang_rand_list) > 1 else 0.0

    ia2_rand_mean  = round(stats.mean(ia2_rand_list), 4) if ia2_rand_list else float("nan")
    ia2_rand_std   = round(stats.pstdev(ia2_rand_list), 4) if len(ia2_rand_list) > 1 else 0.0

    jacc_rand_mean = round(stats.mean(jacc_rand_list), 4) if jacc_rand_list else float("nan")
    jacc_rand_std  = round(stats.pstdev(jacc_rand_list), 4) if len(jacc_rand_list) > 1 else 0.0

    print(f"[Random baseline] Wang mean±std: {wang_rand_mean}±{wang_rand_std}")
    print(f"[Random baseline] IA@k=2 mean±std: {ia2_rand_mean}±{ia2_rand_std}")
    print(f"[Random baseline] Jaccard mean±std: {jacc_rand_mean}±{jacc_rand_std}")
    # ====================== END: Random Baseline ======================

    # Return average scores
    return {

        "Wang_Similarity_prediction": round(mean_similarity_uncover_all, 4),
        "InfoAccretion_Distance_prediction": round(s_2_uncover_all, 4),
        "Jaccard_Similarity_prediction": round(jaccard_uncover_similarity, 4),

        "Wang_Similarity_reference": round(mean_similarity_ontogpt_all, 4),
        "InfoAccretion_Distance_reference": round(s_2_ontogpt_all, 4),
        "Jaccard_Similarity_reference": round(jaccard_ontogpt_similarity, 4),
        
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt-4o")
    parser.add_argument("--is_pretrained", action="store_true")
    parser.add_argument("--test_json", type=str, default="data/SwissProtV3/test_set.json")
    parser.add_argument("--test_is_jsonl", action="store_true")
    parser.add_argument("--pred_txt", type=str, default="ProtT3-baseline/all_checkpoints/protein_captioning_swiss_dataset/lightning_logs/version_0/dataset0_predictions.txt")
    args = parser.parse_args()

    model_type = args.model_type
    is_pretrained = args.is_pretrained
    print(f"[info] model_type={model_type}, is_pretrained={is_pretrained}")

    # Ground Truth GO
    gt_path = Path(args.test_json)
    go_lists = []

    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)       
            last_col = row[-1]  
            if isinstance(last_col, str) and last_col.startswith('['):
                try:
                    last_col = ast.literal_eval(last_col)
                except Exception:
                    pass
            go_lists.append(last_col)

    print("Ground Truth GO:", len(go_lists), (go_lists[0] if len(go_lists) else "<empty>"))

    # Reference texts
    test_path = Path(args.test_json)

    # If your file is JSONL like:
    # you must set lines=True (or pass --test_is_jsonl True)
    test_df = pd.read_json(test_path, lines=(args.test_is_jsonl or True))  # force True since your sample is JSONL

    if 1 in test_df.columns and "label" not in test_df.columns:
        if 0 in test_df.columns and "sequence" not in test_df.columns:
            test_df = test_df.rename(columns={0: "sequence", 1: "label"})
        else:
            test_df = test_df.rename(columns={1: "label"})

    # if "label" not in test_df.columns:
    #     print(f"[error] 'label' column not found in {test_path}", file=sys.stderr)
    #     sys.exit(1)

    referenced_text = test_df["label"].astype(str).tolist()
    print("Referenced Text:", len(referenced_text), (referenced_text[1] if len(referenced_text) else "<empty>"))

    # Predicted texts
    pred_path = Path(args.pred_txt)
    # with pred_path.open("r", encoding="utf-8") as f:
    #     predicted_text = [
    #         (json.loads(line).get("predictions", "").rstrip("\n"))
    #         for line in f if line.strip()
    #     ]
    df = pd.read_json(pred_path, lines=True)
    df["indices"] = pd.to_numeric(df["indices"], errors="coerce")
    df = df.sort_values(["indices"], na_position="last")
    predicted_text = df["predictions"].fillna("").str.rstrip("\n").tolist()

    print("Predicted Text:", len(predicted_text), (predicted_text[1] if len(predicted_text) else "<empty>"))

    # Truncate to the shortest length if mismatched
    # n_pred, n_ref, n_go = len(predicted_text), len(referenced_text), len(go_lists)
    # if len({n_pred, n_ref, n_go}) != 1:
    #     n = 10
    #     print(f"[warn] Length mismatch: pred={n_pred}, ref={n_ref}, go={n_go}. Truncate to {n}.", file=sys.stderr)
    #     predicted_text  = predicted_text[:n]
    #     referenced_text = referenced_text[:n]
    #     go_lists        = go_lists[:n]

    # Metrics (use your existing functions)
    linguistic_metrics = compute_linguistic_metrics(predicted_text, referenced_text)
    ontology_metrics = compute_ontology_metrics(predicted_text, referenced_text, go_lists, model_type, is_pretrained)

    # Print results
    try:
        print("\n=== Linguistic Metrics ===")
        print(json.dumps(linguistic_metrics, ensure_ascii=False, indent=2))
    except TypeError:
        print("\n=== Linguistic Metrics ===")
        print(linguistic_metrics)

    try:
        print("\n=== Ontology Metrics ===")
        print(json.dumps(ontology_metrics, ensure_ascii=False, indent=2))
    except TypeError:
        print("\n=== Ontology Metrics ===")
        print(ontology_metrics)





