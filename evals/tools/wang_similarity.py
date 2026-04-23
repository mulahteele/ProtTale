from goatools.base import get_godag
from goatools.semsim.termwise.wang import SsWang
import numpy as np



def compute_wang_similarity(real_terms_list, predicted_terms_list, godag=None):
    """
    Compute the semantic similarity between two lists of GO term sets
    using Wang's semantic similarity method.

    Args:
        real_terms_list: List of lists containing real GO term sets
        predicted_terms_list: List of lists containing predicted GO term sets
        godag: Optional pre-loaded GO DAG. If None, loads from evals/tools/go-basic.obo

    Returns:
        The mean semantic similarity score and a list of similarity scores for each pair of real and predicted GO term sets.
    """
    if godag is None:
        godag = get_godag("evals/tools/go-basic.obo", optional_attrs=("relationship",))

    def get_ancestors(go_id):
        if go_id not in godag:
            return set()
        term = godag[go_id]
        return {go_id} | term.get_all_parents()

    def expand_to_ancestors(go_list):
        anc = set()
        for go in go_list:
            anc |= get_ancestors(go)
        return anc

    similarity_scores = []
    for real_terms, predicted_terms in zip(real_terms_list, predicted_terms_list):
        try:
            real_terms = [go for go in real_terms if go in godag]
            predicted_terms = [go for go in predicted_terms if go in godag]

            if len(predicted_terms) == 0 or len(real_terms) == 0:
                similarity_scores.append(0)
                continue

            real_expanded = expand_to_ancestors(real_terms)
            pred_expanded = expand_to_ancestors(predicted_terms)
            goids = real_expanded | pred_expanded
            relationships = {"part_of"}
            wang = SsWang(goids, godag, relationships)

            total_similarity = 0
            count = 0
            for go_real in real_expanded:
                max_sim = max(wang.get_sim(go_real, go_pred) for go_pred in pred_expanded)
                total_similarity += max_sim
                count += 1
            for go_pred in pred_expanded:
                max_sim = max(wang.get_sim(go_real, go_pred) for go_real in real_expanded)
                total_similarity += max_sim
                count += 1

            avg_similarity = total_similarity / count if count > 0 else 0
            similarity_scores.append(avg_similarity)
        except Exception:
            similarity_scores.append(0)

    mean_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    return mean_similarity, similarity_scores

