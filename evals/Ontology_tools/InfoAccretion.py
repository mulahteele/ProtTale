import pandas as pd
import numpy as np
from goatools.obo_parser import GODag

def load_ia(ia_file):
    """
    Load IA values from the IA.txt file into a dictionary.
    :param ia_file: Path to IA.txt.
    :return: Dictionary {GO term: IA value}.
    """
    ia_df = pd.read_csv(ia_file, sep="\t", header=None, names=["term", "ia"])
    ia_dict = dict(zip(ia_df["term"], ia_df["ia"]))
    return ia_dict

def get_ancestors(go_term, go_dag):
    """
    get all ancestors
    """
    if go_term not in go_dag:
        return set()
    ancestors = set(go_dag[go_term].get_all_parents())
    ancestors.add(go_term)  
    return ancestors

def compute_InfoAccretion_distance(real_terms_list, predicted_terms_list, ia_file="IA.txt", k=1):
    """
    Compute the semantic distance (S_k) based on weighted remaining uncertainty (wru) and weighted misinformation (wmi).
    
    Args:
        real_terms_list: List of real GO term sets (T) for all entities.
        predicted_terms_list: List of predicted GO term sets (P) for all entities.
        ia_file: Path to IA.txt containing {GO term: IA value}.
        k: Power coefficient for distance calculation (default: 1).
        
    Returns:
        Weighted remaining uncertainty (wru), weighted misinformation (wmi), and semantic distance (S_k).
    """
    # Load IA values
    ia_dict = load_ia(ia_file)
    go_dag = GODag("go-basic.obo")


    weighted_ru = 0
    weighted_mi = 0
    total_weight = 0
    s_k_list = []

    for real_terms, predicted_terms in zip(real_terms_list, predicted_terms_list):
        
        real_terms = [go for go in real_terms if go in ia_dict]
        predicted_terms = [go for go in predicted_terms if go in ia_dict]

        # if not real_terms:  # Skip if real_terms is empty
        #     continue

        real_terms_set = set()
        for term in real_terms:
            real_terms_set |= get_ancestors(term, go_dag)

        predicted_terms_set = set()
        for term in predicted_terms:
            predicted_terms_set |= get_ancestors(term, go_dag)

        real_terms_set = {go for go in real_terms_set if go in ia_dict}
        predicted_terms_set = {go for go in predicted_terms_set if go in ia_dict}

        # Calculate ru and mi for this entity
        ru_terms = real_terms_set - predicted_terms_set
        mi_terms = predicted_terms_set - real_terms_set

        ru = sum(ia_dict.get(term, 0) for term in ru_terms)
        mi = sum(ia_dict.get(term, 0) for term in mi_terms)

        # Calculate weight for this entity (sum of IA values for real terms)
        # weight = sum(ia_dict.get(term, 0) for term in real_terms_set)

        # weighted_ru += weight * ru
        # weighted_mi += weight * mi
        # total_weight += weight
        s_k_i = (ru**k + mi**k)**(1/k)
        s_k_list.append(s_k_i)

    # Normalize by total weight
    # wru = weighted_ru / total_weight if total_weight > 0 else 0
    # wmi = weighted_mi / total_weight if total_weight > 0 else 0

    # Calculate semantic distance
    s_k = np.mean(s_k_list)
    # print('information accretion score: ')
    # print(s_k_list)

    return s_k

# Example usage
if __name__ == "__main__":
    ia_file = "IA.txt"  # Path to your IA.txt file

    # Example GO term sets for multiple entities
    real_terms_list = [
        ["GO:2001072", "GO:2001080"],
        ["GO:2001068", "GO:2001071"]
    ]
    predicted_terms_list = [
        ["GO:2001072", "GO:2001080"],
        ["GO:2001072", "GO:2001080"]
    ]

    # Compute distances
    wru, wmi, s_k = compute_InfoAccretion_distance(real_terms_list, predicted_terms_list, ia_file="IA.txt", k=2)

    print(f"Weighted Remaining Uncertainty (wru): {wru:.4f}")
    print(f"Weighted Misinformation (wmi): {wmi:.4f}")
    print(f"Semantic Distance (S_k): {s_k:.4f}")