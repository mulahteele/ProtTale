from evaluation.Ontology_tools.InfoAccretion import compute_InfoAccretion_distance
from evaluation.Ontology_tools.wang_similarity import compute_wang_similarity
from evaluation.Ontology_tools.jaccard_similarity import compute_jaccard_similarity
from evaluation.Ontology_tools.Onto_extraction import process_texts_with_api, process_texts_with_ollama
import os
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
    ia_path = "evaluation/Ontology_tools/IA.txt"

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




    # Read GO terms back from the CSV file
    mf_go_terms_file = "evaluation/Ontology_tools/molecular_function_go_terms.csv"
    MF_go_terms = read_go_terms_from_csv(mf_go_terms_file)
    print(f"Extracted {len(MF_go_terms)} GO terms under molecular_function namespace.")

    all_GO = filter_existing_go_terms(all_GO, MF_go_terms)
    print(f"GO terms from the all_GO: {all_GO}")
    
    reference_go_terms = filter_existing_go_terms(reference_go_terms, MF_go_terms)
    print(f"GO terms from the reference_go_terms: {reference_go_terms}")

    predicted_go_terms = filter_existing_go_terms(predicted_go_terms, MF_go_terms)
    print(f"GO terms from the predicted_go_terms: {predicted_go_terms}")



    ###_____________________________Read and Process the data_____________________________###


    print('Calculating the current\'s model result......')
    shuffled_go_terms = reference_go_terms.copy()
    random.shuffle(shuffled_go_terms)

    mean_similarity_shuffled_all = compute_wang_similarity(all_GO, shuffled_go_terms)
    s_2_shuffled_all = compute_InfoAccretion_distance(all_GO, shuffled_go_terms, ia_file=ia_path, k=2)
    jaccard_shuffled_similarity = compute_jaccard_similarity(all_GO, shuffled_go_terms)

    mean_similarity_uncover_all = compute_wang_similarity(all_GO, predicted_go_terms)
    s_2_uncover_all = compute_InfoAccretion_distance(all_GO, predicted_go_terms, ia_file=ia_path, k=2)
    jaccard_uncover_similarity = compute_jaccard_similarity(all_GO, predicted_go_terms)

    mean_similarity_ontogpt_all = compute_wang_similarity(all_GO, reference_go_terms)
    s_2_ontogpt_all = compute_InfoAccretion_distance(all_GO, reference_go_terms, ia_file=ia_path, k=2)
    jaccard_ontogpt_similarity = compute_jaccard_similarity(all_GO, reference_go_terms)


    # Return average scores
    return {

        # "Wang_Similarity_input": round(mean_similarity_input_all, 4),
        # "InfoAccretion_Distance_input": round(s_2_input_all, 4),

        "Wang_Similarity_prediction": round(mean_similarity_uncover_all, 4),
        "InfoAccretion_Distance_prediction": round(s_2_uncover_all, 4),
        "Jaccard_Similarity_prediction": round(jaccard_uncover_similarity, 4),

        "Wang_Similarity_reference": round(mean_similarity_ontogpt_all, 4),
        "InfoAccretion_Distance_reference": round(s_2_ontogpt_all, 4),
        "Jaccard_Similarity_reference": round(jaccard_ontogpt_similarity, 4),

        "Wang_Similarity_shuffled": round(mean_similarity_shuffled_all, 4),
        "InfoAccretion_Distance_shuffled": round(s_2_shuffled_all, 4),
        "Jaccard_Similarity_shuffled": round(jaccard_shuffled_similarity, 4),

        # "Wang_Similarity_baseline_4o": round(mean_similarity_baseline_4o, 4),
        # "InfoAccretion_Distance_baseline_4o": round(s_2_baseline_4o, 4),

        # "Wang_Similarity_baseline_deepseek": round(mean_similarity_baseline_deepseek, 4),
        # "InfoAccretion_Distance_baseline_deepseek": round(s_2_baseline_deepseek, 4),

        # "Wang_Similarity_baseline_llama": round(mean_similarity_baseline_llama, 4),
        # "InfoAccretion_Distance_baseline_llama": round(s_2_baseline_llama, 4),

    }

