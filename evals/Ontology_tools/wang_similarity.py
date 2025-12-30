from goatools.base import get_godag
from goatools.semsim.termwise.wang import SsWang
import numpy as np



def compute_wang_similarity(real_terms_list, predicted_terms_list):
    """
    Compute the semantic similarity between two lists of GO term sets
    using Wang's semantic similarity method.

    Args:
        real_terms_list: List of lists containing real GO term sets
        predicted_terms_list: List of lists containing predicted GO term sets

    Returns:
        The mean semantic similarity score and a list of similarity scores for each pair of real and predicted GO term sets.
    """
    # Load the GO DAG from the OBO file
    godag = get_godag("go-basic.obo", optional_attrs=("relationship",))

    # Initialize results list
    similarity_scores = []

    # Iterate through the paired real and predicted GO term sets
    for real_terms, predicted_terms in zip(real_terms_list, predicted_terms_list):

        real_terms = [go for go in real_terms if go in godag]
        predicted_terms = [go for go in predicted_terms if go in godag]

        if len(predicted_terms) == 0 or len(real_terms) == 0:
            # If either set is empty, similarity is 0
            similarity_scores.append(0)
            continue

        # Combine all GO terms from both sets
        goids = set(real_terms).union(set(predicted_terms))
        # Define the relationships to include
        relationships = {"part_of"}  
        # Initialize Wang's semantic similarity calculator
        wang = SsWang(goids, godag, relationships)

        total_similarity = 0
        count = 0

        # Compute the maximum similarity for each term in the real set against the predicted set
        for go_real in real_terms:
            max_sim = max(wang.get_sim(go_real, go_pred) for go_pred in predicted_terms)
            total_similarity += max_sim
            count += 1
            
        # Compute the maximum similarity for each term in the predicted set against the real set
        for go_pred in predicted_terms:
            max_sim = max(wang.get_sim(go_real, go_pred) for go_real in real_terms)
            total_similarity += max_sim
            count += 1


        # Compute average similarity
        avg_similarity = total_similarity / count if count > 0 else 0
        similarity_scores.append(avg_similarity)

    # Compute mean similarity score
    mean_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    # print('similarity_scores:')
    # print(similarity_scores)

    return mean_similarity

# Example usage
if __name__ == "__main__":
    real_terms_list = [
        ["GO:2001068", "GO:2001071"],
        ["GO:2001068", "GO:2001071"]
    ]
    predicted_terms_list = [
        ["GO:2001072", "GO:2001080"],
        ["GO:2001072", "GO:2001080"]
    ]

    mean_similarity, similarities = compute_wang_similarity(real_terms_list, predicted_terms_list)
    print(f"Mean Semantic Similarity: {mean_similarity}")
    print(f"Semantic Similarities: {similarities}")
