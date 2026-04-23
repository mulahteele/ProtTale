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
        Mean semantic distance (S_k).
    """
    ia_dict = load_ia(ia_file)
    go_dag = GODag("evals/tools/go-basic.obo")
    s_k_list = []

    for real_terms, predicted_terms in zip(real_terms_list, predicted_terms_list):
        
        real_terms = [go for go in real_terms if go in ia_dict]
        predicted_terms = [go for go in predicted_terms if go in ia_dict]

        real_terms_set = set()
        for term in real_terms:
            real_terms_set |= get_ancestors(term, go_dag)

        predicted_terms_set = set()
        for term in predicted_terms:
            predicted_terms_set |= get_ancestors(term, go_dag)

        real_terms_set = {go for go in real_terms_set if go in ia_dict}
        predicted_terms_set = {go for go in predicted_terms_set if go in ia_dict}

        ru_terms = real_terms_set - predicted_terms_set
        mi_terms = predicted_terms_set - real_terms_set
        ru = sum(ia_dict.get(term, 0) for term in ru_terms)
        mi = sum(ia_dict.get(term, 0) for term in mi_terms)
        s_k_i = (ru**k + mi**k)**(1/k)
        s_k_list.append(s_k_i)

    s_k = np.mean(s_k_list)
    return s_k
