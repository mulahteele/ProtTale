from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

def compute_linguistic_metrics(predictions, references):
    """
    Compute corpus-level BLEU (2,4) and average ROUGE/METEOR for lists of predictions and references.
    Returns scores in percentage (0–100).
    """
    assert len(predictions) == len(references), "Number of predictions and references must match."

    # 1) Tokenize once up front
    refs_tok = [r.split() for r in references]
    hyps_tok = [p.split() for p in predictions]

    # 2) Corpus-level BLEU (single pass over the corpus)
    #    corpus_bleu expects each reference as a list of possible refs -> [[ref_tokens], ...]
    refs_for_bleu = [[rt] for rt in refs_tok]
    bleu2 = 100 * corpus_bleu(refs_for_bleu, hyps_tok, weights=(0.5, 0.5))
    bleu4 = 100 * corpus_bleu(refs_for_bleu, hyps_tok, weights=(0.25, 0.25, 0.25, 0.25))

    # 3) ROUGE (per-example, then average)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)  # first arg: reference, second: prediction
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rL.append(scores["rougeL"].fmeasure)
    rouge1 = 100 * (sum(r1) / len(r1))
    rouge2 = 100 * (sum(r2) / len(r2))
    rougeL = 100 * (sum(rL) / len(rL))

    # 4) METEOR (per-example, then average)
    meteors = [meteor_score([rt], ht) for rt, ht in zip(refs_tok, hyps_tok)]
    meteor = 100 * (sum(meteors) / len(meteors))

    return {
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougeL,
        "METEOR": meteor,
    }
