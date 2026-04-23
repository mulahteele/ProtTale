import torch
import torch.nn.functional as F
import pickle
import csv
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np


def caption_evaluate(predictions, targets, tokenizer, text_trunc_length, verbose=True):
    targets = [t.strip() for t in targets]
    meteor_scores = []
    references = []
    hypotheses = []
    for gt, out in tqdm(zip(targets, predictions), disable=not verbose):
        gt_tokens = tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        ## added for galactica
        gt_tokens = list(filter(('<pad>').__ne__, gt_tokens))
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('<pad>').__ne__, out_tokens))
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))
    bleu2 *= 100
    bleu4 *= 100

    if verbose:
        print('BLEU-2 score:', bleu2)
        print('BLEU-4 score:', bleu4)
    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
    if verbose:
        print('Average Meteor score:', _meteor_score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for gt, out in tqdm(zip(targets, predictions), disable=not verbose):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]) * 100
    if verbose:
        print('ROUGE score:')
        print('rouge1:', rouge_1)
        print('rouge2:', rouge_2)
        print('rougeL:', rouge_l)
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def pad_and_concat(tensor_list, fill_value=0):
    '''
    concat the first dimension and pad the second dimension
    tensor_list: [[B (diff), N_num, *], ...]
    '''
    device = tensor_list[0].device
    dtype=tensor_list[0].dtype
    max_dim1 = max(t.shape[1] for t in tensor_list)
    sum_dim0 = sum(t.shape[0] for t in tensor_list)
    if len(tensor_list[0].shape) == 3:
        out = torch.full((sum_dim0, max_dim1, tensor_list[0].shape[-1]), fill_value=fill_value, device=device, dtype=dtype)
        i = 0
        for t in tensor_list:
            out[i:i+t.shape[0], :t.shape[1]] = t
            i += t.shape[0]
        return out
    elif len(tensor_list[0].shape) == 2:
        out = torch.full((sum_dim0, max_dim1), fill_value=fill_value, device=device, dtype=dtype)
        i = 0
        for t in tensor_list:
            out[i:i+t.shape[0], :t.shape[1]] = t
            i += t.shape[0]
        return out
    raise NotImplementedError()


def hf_enable_gradient_checkpointing(hf_model):
    if hasattr(hf_model, "enable_input_require_grads"):
        hf_model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        hf_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # enable gradient checkpointing for memory efficiency
    hf_model.gradient_checkpointing_enable()
    return hf_model


# ----- stage2 helpers (from blip2_stage2) -----

def _mean_conf(all_confidences):
    """Input: list of list of float (e.g. from result_list confidences). Returns mean rounded to 4 decimals."""
    flat = [float(x) for c in all_confidences for x in (c if isinstance(c, (list, tuple)) else [c])]
    return round(sum(flat) / len(flat), 4) if flat else 0.0


def _json_default(o):
    """Convert non-JSON types for json.dumps: tensor -> scalar/list, else str()."""
    if isinstance(o, torch.Tensor):
        return o.item() if o.dim() == 0 else o.detach().cpu().tolist()
    return str(o)


def load_or_process(file_path, data, data_name, safe_model_type):
    """Load data from a cached file if available; otherwise, process and save it."""
    try:
        with open(file_path, 'rb') as file:
            print(f"Loading {data_name} GO terms from {file_path} processing by {safe_model_type}...")
            return pickle.load(file)
    except FileNotFoundError:
        print(f"File not found. Processing {data_name} using {safe_model_type}...")
        from evals.tools.extraction import process_texts_with_api
        result = process_texts_with_api(data)
        with open(file_path, 'wb') as file:
            pickle.dump(result, file)
        print(f"Processed {data_name} GO terms saved to {file_path}.")
        return result


def filter_existing_go_terms(go_list, extracted_go_terms):
    """Filter and return only existing GO terms from the example list in a list of lists format."""
    extracted_ids = {term[0] for term in extracted_go_terms}
    return [[go_id for go_id in go_set if go_id in extracted_ids] for go_set in go_list]


def load_mf_go_ids_from_tsv(file_path, aspect_value='molecular_function'):
    """Load GO IDs from go_files.tsv (columns: go_id, aspect). Returns set of go_id where aspect == aspect_value."""
    mf_ids = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # header
        for row in reader:
            if len(row) >= 2 and row[1].strip() == aspect_value:
                mf_ids.add(row[0].strip())
    return mf_ids


def filter_go_terms_by_set(go_list, allowed_ids):
    """Filter list of lists of GO IDs to only keep IDs in allowed_ids (e.g. molecular_function only)."""
    return [[go_id for go_id in go_set if go_id in allowed_ids] for go_set in go_list]


def read_go_terms_from_csv(file_path):
    """Read GO terms from the saved CSV file."""
    go_terms = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            go_terms.append(row)
    return go_terms


def _is_nested_list(x):
    """Return True if x is a list of lists/tuples/sets (or empty list)."""
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], (list, tuple, set)))


def empty_rate_nested(xs):
    """For nested list[list[str]]: proportion of empty inner lists. For flat list[str]: 0.0 if non-empty else 1.0."""
    if not isinstance(xs, list):
        return float("nan")
    if _is_nested_list(xs):
        n = len(xs) if xs else 1
        return sum(1 for v in xs if not v) / n
    return 0.0 if xs else 1.0


def build_joint_nonempty_mask(pred_list, ref_list):
    """Build a boolean mask keeping indices where BOTH pred_list[i] and ref_list[i] are non-empty."""
    if not (_is_nested_list(pred_list) and _is_nested_list(ref_list)):
        return [bool(pred_list) and bool(ref_list)]
    n = min(len(pred_list), len(ref_list))
    if len(pred_list) != len(ref_list):
        print(f"[WARN] Different lengths: pred={len(pred_list)} ref={len(ref_list)}; truncating to {n}.")
    return [bool(pred_list[i]) and bool(ref_list[i]) for i in range(n)]


def filter_parallel_by_mask(seq_list, mask):
    """Apply a boolean mask to each sequence list in seq_list in parallel."""
    out = []
    for seq in seq_list:
        if not _is_nested_list(seq):
            raise ValueError("Expected nested list[list[str]] for masking.")
        out.append([v for v, m in zip(seq, mask) if m])
    return out


# ----- stage1 helpers (from blip2_stage1) -----

def l2_normalize(x, eps=1e-8):
    """L2-normalize per row."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def cosine_matrix(Z):
    """Z must be L2-normalized; returns pairwise cosine similarities."""
    return Z @ Z.T


def align_loss(z_s, z_t, beta=1.0):
    """z_t is teacher; gradients must not flow into it."""
    z_t_det = z_t.detach()
    mse = F.mse_loss(z_s, z_t_det)
    cos = 1.0 - (z_s * z_t_det).sum(dim=-1).mean()
    return mse + beta * cos


def struct_consistency(Zs, Zt, alpha=1.0):
    """Match pairwise relation matrices; Zt is teacher (no grad)."""
    Zt_det = Zt.detach()
    Cs = cosine_matrix(Zs)
    Ct = cosine_matrix(Zt_det)
    return ((alpha * Cs - alpha * Ct) ** 2).mean()


@torch.no_grad()
def auc_from_scores(scores: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """AUROC via Mann–Whitney U (rank statistic) with tie-handling."""
    scores = scores.float()
    labels = labels.float()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = pos.numel(), neg.numel()
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(float('nan'), device=scores.device)
    all_scores = torch.cat([pos, neg], dim=0)
    order = torch.argsort(all_scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, all_scores.numel() + 1, device=all_scores.device, dtype=torch.float32)
    uniq, inv, counts = torch.unique(all_scores, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sum_ranks = torch.zeros_like(uniq, dtype=torch.float32)
        sum_ranks.scatter_add_(0, inv, ranks)
        mean_ranks = sum_ranks / counts.float()
        ranks = mean_ranks[inv]
    sum_pos_ranks = ranks[:n_pos].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg + eps)
    return auc