from typing import List, Iterable
from goatools.obo_parser import GODag


def compute_jaccard_similarity_per_row(
    real_terms_list: List[Iterable[str]],
    predicted_terms_list: List[Iterable[str]],
    obo_path: str = "go-basic.obo",
    include_relationships: bool = True,
) -> List[float]:
    """
    Per-row Jaccard similarity between ancestor-expanded GO sets (including self).

    Returns:
        One float per (real_terms, pred_terms) pair, same order as input lists.
    """
    optional = ("relationship",) if include_relationships else ()
    godag = GODag(obo_path, optional_attrs=optional)

    def get_ancestors(go_id: str) -> set[str]:
        if go_id not in godag:
            return set()
        term = godag[go_id]
        return {go_id} | term.get_all_parents()

    def expand_to_ancestors(go_list: Iterable[str]) -> set[str]:
        anc = set()
        for go in go_list:
            anc |= get_ancestors(go)
        return anc

    scores: List[float] = []
    for real_terms, pred_terms in zip(real_terms_list, predicted_terms_list):
        real = [go for go in real_terms if go in godag]
        pred = [go for go in pred_terms if go in godag]

        if not real or not pred:
            scores.append(0.0)
            continue

        anc_real = expand_to_ancestors(real)
        anc_pred = expand_to_ancestors(pred)

        union = anc_real | anc_pred
        score = len(anc_real & anc_pred) / len(union) if union else 0.0
        scores.append(score)

    return scores


def compute_jaccard_similarity(
    real_terms_list: List[Iterable[str]],
    predicted_terms_list: List[Iterable[str]],
    obo_path: str = "go-basic.obo",
    include_relationships: bool = True,
) -> float:
    """
    Compute Jaccard similarity between ancestor-expanded GO sets (including self).

    Args:
        real_terms_list: list of iterables of GO IDs (e.g., [["GO:0008150", ...], ...])
        predicted_terms_list: list of iterables of GO IDs
        obo_path: path to go-basic.obo
        include_relationships: whether to load 'relationship' (e.g., part_of)

    Returns:
        Mean Jaccard similarity over pairs (float), mirroring compute_wang_similarity.
    """
    scores = compute_jaccard_similarity_per_row(
        real_terms_list,
        predicted_terms_list,
        obo_path=obo_path,
        include_relationships=include_relationships,
    )
    return sum(scores) / len(scores) if scores else 0.0
