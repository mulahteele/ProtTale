import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blip2qformer import Blip2Qformer
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from tqdm import tqdm
from pathlib import Path
from torchmetrics.functional import auroc
from model.help_funcs import AttrDict, pad_and_concat
from typing import Any, Dict
import contextlib


@contextlib.contextmanager
def freeze_params(module):
    if module is None:
        yield
        return
    prev = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, rg in zip(module.parameters(), prev):
            p.requires_grad_(rg)


def l2_normalize(x, eps=1e-8):
    # L2-normalize per row
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def cosine_matrix(Z):
    # Z must be L2-normalized; returns pairwise cosine similarities
    return Z @ Z.T

def align_loss(z_s, z_t, beta=0.5):
    # z_t is treated as teacher; gradients must not flow into it
    z_t_det = z_t.detach()
    mse = F.mse_loss(z_s, z_t_det)
    cos = 1.0 - (z_s * z_t_det).sum(dim=-1).mean()
    return mse + beta * cos

def struct_consistency(Zs, Zt, alpha=20.0):
    # match the pairwise relation matrices; Zt is teacher (no grad)
    Zt_det = Zt.detach()
    Cs = cosine_matrix(Zs)       # [B, B]
    Ct = cosine_matrix(Zt_det)   # [B, B]
    return ((alpha * Cs - alpha * Ct) ** 2).mean()

@torch.no_grad()
def auc_from_scores(scores: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    AUROC via Mann–Whitney U (rank statistic) with tie-handling.
    scores: continuous scores (higher = more likely label==1)
    labels: 0/1 tensor
    """
    scores = scores.float()
    labels = labels.float()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = pos.numel(), neg.numel()
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(float('nan'), device=scores.device)

    all_scores = torch.cat([pos, neg], dim=0)

    # ranks with average tie-handling
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



class Blip2Stage1(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)
        
        self.args = args
        self.rerank_cand_num = args.rerank_cand_num
        
        # Validate encoder_type and plm_name consistency
        encoder_type = getattr(args, 'encoder_type', 'auto')
        if encoder_type != 'auto':
            if encoder_type == 'esm2' and not args.plm_name.startswith('facebook/esm2'):
                raise ValueError(f"encoder_type='{encoder_type}' but plm_name='{args.plm_name}' does not start with 'facebook/esm2'")
            elif encoder_type == 'esmc' and not args.plm_name.startswith('esmc_'):
                raise ValueError(f"encoder_type='{encoder_type}' but plm_name='{args.plm_name}' does not start with 'esmc_'")
        
        self.blip2qformer = Blip2Qformer(args.ptm, args.lm, args.bert_name, args.plm_name, args.temperature, args.plm_lora_r, args.plm_lora_alpha, args.plm_lora_dropout, args.plm_tune, args.num_query_token, args.cross_attention_freq, args.projection_dim, args.pool_size, args.load_4bit)
        self.save_hyperparameters(args)
        
        self.w_t2t = args.w_t2t
        self.w_align = args.w_align
        self.w_struct = args.w_struct
        self.temperature = args.temperature
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Save ALL parameters (including frozen PLM weights)
        # This makes checkpoints larger but self-contained
        # Original behavior: only save trainable parameters (commented out below)
        pass
        
        # # Original code - only save trainable parameters:
        # to_be_removed = []
        # for key, value in checkpoint['state_dict'].items():
        #     try:
        #         if not self.get_parameter(key).requires_grad:
        #             to_be_removed.append(key)
        #     except AttributeError:
        #         to_be_removed.append(key)
        # for key in to_be_removed:
        #     checkpoint['state_dict'].pop(key)
        

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def on_validation_epoch_start(self):
        # VAL buffers
        self._val_sim_ss = []     # seq-seq cosine scores (VAL)
        self._val_labels = []     # 0/1 labels (VAL)
        self._val_acc = []        # optional: text-branch acc (VAL)

        # TEST buffers
        self._test_sim_ss = []    # seq-seq cosine scores (TEST)
        self._test_labels = []    # 0/1 labels (TEST)
        self._test_acc = []       # optional: text-branch acc (TEST)

        #logits (only collect from TEST stream if desired)
        self.logits = []   


    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        idx==0 (VAL): compute text-text loss/acc and collect seq-seq scores for AUROC on validation data.
        idx==1 (TEST): same but collected into separate buffers; optionally collect logits at epoch 10.
        """
        prot_tokens1, prot_tokens2, text_tokens1, text_tokens2, labels = batch

        # Forward
        text_emb1 = self.blip2qformer.text_forward(text_tokens1)
        text_emb2 = self.blip2qformer.text_forward(text_tokens2)
        seq_emb1, _  = self.blip2qformer.prot_forward(prot_tokens1)
        seq_emb2, _  = self.blip2qformer.prot_forward(prot_tokens2)

        # Normalize for cosine stability
        text_emb1 = F.normalize(text_emb1, dim=-1, eps=1e-8)
        text_emb2 = F.normalize(text_emb2, dim=-1, eps=1e-8)
        seq_emb1  = F.normalize(seq_emb1,  dim=-1, eps=1e-8)
        seq_emb2  = F.normalize(seq_emb2,  dim=-1, eps=1e-8)

        # Text-text contrastive (monitoring / teacher training signal)
        temperature = getattr(self, "temperature", 0.07)
        labels = labels.to(text_emb1.device).float().view(-1)  # [B]
        sim_tt = (text_emb1 * text_emb2).sum(dim=-1)           # cosine in [-1,1]
        logits_tt = sim_tt / temperature
        loss = F.binary_cross_entropy_with_logits(logits_tt, labels)

        # Optional text-branch accuracy
        probs = torch.sigmoid(logits_tt)
        preds = (probs >= 0.5).float()
        acc = (preds == labels).float().mean()

        # Seq-seq scores to evaluate AUROC against 0/1 labels
        sim_ss = (seq_emb1 * seq_emb2).sum(dim=-1)             # cosine in [-1,1]

        if dataloader_idx % 2 == 0:
            # VAL stream
            self._val_sim_ss.append(sim_ss.detach().cpu())
            self._val_labels.append(labels.detach().cpu())
            self._val_acc.append(acc)
        else:
            # TEST stream
            self._test_sim_ss.append(sim_ss.detach().cpu())
            self._test_labels.append(labels.detach().cpu())
            self._test_acc.append(acc)

            if (self.current_epoch + 1) == 10:
                logits_debug = sim_ss / temperature
                self.logits.extend(logits_debug.detach().cpu().tolist())

        return loss


    
    def get_precision(self, precision):
        if precision in {'16', '16-mixed'}:
            return torch.float16
        elif precision in {'bf16', 'bf16-mixed'}:
            return torch.bfloat16
        elif precision in {'32',}:
            return torch.float32
        else:
            raise NotImplementedError
    
    def retrieval_evaluation_and_log(self, match_dataloader, log_prefix="") -> None:
        with self.maybe_autocast(self.get_precision(self.trainer.precision)):
            ## for onto test set
            p2t_acc, t2p_acc, p2t_rec20, t2p_rec20, \
            p2t_rerank_acc, t2p_rerank_acc, p2t_rerank_rec20, t2p_rerank_rec20, \
            prot_feat_total, text_feat_total, prot_embed_total, prot_mask_total, text_total, text_mask_total = \
                eval_retrieval_inbatch_with_rerank(self.blip2qformer, match_dataloader, self.device)

            self.log(f"{log_prefix}inbatch_p2t_acc", p2t_acc, sync_dist=False)
            self.log(f"{log_prefix}inbatch_t2p_acc", t2p_acc, sync_dist=False)
            self.log(f"{log_prefix}inbatch_p2t_rec20", p2t_rec20, sync_dist=False)
            self.log(f"{log_prefix}inbatch_t2p_rec20", t2p_rec20, sync_dist=False)

            self.log(f"{log_prefix}rerank_inbatch_p2t_acc", p2t_rerank_acc, sync_dist=False)
            self.log(f"{log_prefix}rerank_inbatch_t2p_acc", t2p_rerank_acc, sync_dist=False)
            self.log(f"{log_prefix}rerank_inbatch_p2t_rec20", p2t_rerank_rec20, sync_dist=False)
            self.log(f"{log_prefix}rerank_inbatch_t2p_rec20", t2p_rerank_rec20, sync_dist=False)
            
            p2t_acc, p2t_rec20, t2p_acc, t2p_rec20, sim_p2t = \
                eval_retrieval_fullset(prot_feat_total, text_feat_total, self.device)
            self.log(f"{log_prefix}fullset_p2t_acc", p2t_acc, sync_dist=False)
            self.log(f"{log_prefix}fullset_t2p_acc", t2p_acc, sync_dist=False)
            self.log(f"{log_prefix}fullset_p2t_rec20", p2t_rec20, sync_dist=False)
            self.log(f"{log_prefix}fullset_t2p_rec20", t2p_rec20, sync_dist=False)

            p2t_acc, p2t_rec20, t2p_acc, t2p_rec20 = \
                eval_retrieval_fullset_for_rerank(self.blip2qformer, sim_p2t, prot_embed_total, prot_mask_total, text_total, text_mask_total, self.rerank_cand_num, self.device)
            self.log(f"{log_prefix}rerank_fullset_p2t_acc", p2t_acc, sync_dist=False)
            self.log(f"{log_prefix}rerank_fullset_t2p_acc", t2p_acc, sync_dist=False)
            self.log(f"{log_prefix}rerank_fullset_p2t_rec20", p2t_rec20, sync_dist=False)
            self.log(f"{log_prefix}rerank_fullset_t2p_rec20", t2p_rec20, sync_dist=False)



    # ====================== END (per epoch) ======================
    def on_validation_epoch_end(self) -> None:
        # Compute epoch-level AUROC for VAL
        auc_val = torch.tensor(float('nan'))
        if hasattr(self, "_val_sim_ss") and len(self._val_sim_ss) > 0:
            sim_all_val = torch.cat(self._val_sim_ss, dim=0)
            lab_all_val = torch.cat(self._val_labels, dim=0)
            auc_val = auc_from_scores(sim_all_val, lab_all_val)

        # Compute epoch-level AUROC for TEST
        auc_test = torch.tensor(float('nan'))
        if hasattr(self, "_test_sim_ss") and len(self._test_sim_ss) > 0:
            sim_all_test = torch.cat(self._test_sim_ss, dim=0)
            lab_all_test = torch.cat(self._test_labels, dim=0)
            auc_test = auc_from_scores(sim_all_test, lab_all_test)

        # Safe mean accuracies
        acc_val_mean = torch.stack(self._val_acc).mean() if len(self._val_acc) > 0 else torch.tensor(float('nan'))
        acc_test_mean = torch.stack(self._test_acc).mean() if len(self._test_acc) > 0 else torch.tensor(float('nan'))

        # Rank-0 logging + optional dump
        if self.trainer.global_rank == 0:
            # Dump debug logits at epoch 10 (1-indexed), from TEST stream
            if (self.current_epoch + 1) == 10 and hasattr(self, "logits") and len(self.logits) > 0:
                Path("saved_results").mkdir(parents=True, exist_ok=True)
                torch.save({"logits_list": self.logits}, "saved_results/stage1_result_struct")

            # Log metrics per stream
            self.log_dict(
                {
                    "val/seqseq_AUC_VAL": auc_val,
                    "val/seqseq_AUC_TEST": auc_test,
                    "val/text_pair_acc_VAL": acc_val_mean,
                    "val/text_pair_acc_TEST": acc_test_mean,
                },
                prog_bar=True,
                sync_dist=False,
            )

        # Clear per-epoch buffers (all ranks)
        if hasattr(self, "_val_sim_ss"):   self._val_sim_ss.clear()
        if hasattr(self, "_val_labels"):   self._val_labels.clear()
        if hasattr(self, "_val_acc"):      self._val_acc.clear()
        if hasattr(self, "_test_sim_ss"):  self._test_sim_ss.clear()
        if hasattr(self, "_test_labels"):  self._test_labels.clear()
        if hasattr(self, "_test_acc"):     self._test_acc.clear()
        if hasattr(self, "logits") and (self.current_epoch + 1) == 10:
            self.logits.clear()



    def training_step(self, batch, batch_idx):
        # Step the LR scheduler (epoch-aware)
        self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        prot_tokens1, prot_tokens2, text_tokens1, text_tokens2, labels = batch
        batch_size = text_tokens1.input_ids.shape[0]

        # ----- teacher/text branch is trained by t2t loss -----
        text_emb1 = self.blip2qformer.text_forward(text_tokens1)  # [B, D]
        text_emb2 = self.blip2qformer.text_forward(text_tokens2)  # [B, D]

        text_emb1 = F.normalize(text_emb1, dim=-1, eps=1e-8)
        text_emb2 = F.normalize(text_emb2, dim=-1, eps=1e-8)

        temperature = getattr(self, "temperature", 0.07)
        labels = labels.to(text_emb1.device).float().view(-1)

        sim_tt = (text_emb1 * text_emb2).sum(dim=-1)
        logits_tt = sim_tt / temperature
        bce = nn.BCEWithLogitsLoss()
        loss_t2t = bce(logits_tt, labels)

        # ===== freeze Q-Former only for the align/struct path; allow PLM to update =====
        with freeze_params(self.blip2qformer.Qformer):
            seq_emb1, _ = self.blip2qformer.prot_forward(prot_tokens1)  # [B, D]
            # seq_emb2, _ = self.blip2qformer.prot_forward(prot_tokens2)  # [B, D]

            seq_emb1 = F.normalize(seq_emb1, dim=-1, eps=1e-8)
            # seq_emb2 = F.normalize(seq_emb2, dim=-1, eps=1e-8)

            beta_align = getattr(self, "beta_align", 0.5)
            loss_align = align_loss(seq_emb1, text_emb1, beta=beta_align)

            alpha_struct = getattr(self, "alpha_struct", 1.0)
            loss_struct = struct_consistency(seq_emb1, text_emb1, alpha=alpha_struct)

        # Combine losses (exclude s2s from optimization if desired)
        w_t2t    = getattr(self, "w_t2t", 1.0)
        w_align  = getattr(self, "w_align", 1.0)
        w_struct = getattr(self, "w_struct", 0.1)

        loss = w_t2t * loss_t2t + w_align * loss_align + w_struct * loss_struct

        # Logging
        self.log_dict(
            {
                "train/loss_total": loss,
                "train/loss_t2t": loss_t2t,
                "train/loss_align": loss_align,
                "train/loss_struct": loss_struct,
            },
            on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size,
        )
        return loss


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Stage1")
        # train mode
        parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
        parser.add_argument('--save_every_n_epochs', type=int, default=0)
        parser.add_argument('--ptm', action='store_true', help='use graph-text matching or not', default=True)
        parser.add_argument('--lm', action='store_true', help='use language modeling or not', default=True)

        # evaluation
        parser.add_argument('--rerank_cand_num', type=int, default=128)
        
        # plm
        parser.add_argument('--plm_name', type=str, default='facebook/esm2_t30_150M_UR50D')
        parser.add_argument('--plm_tune', type=str, default='freeze')
        parser.add_argument('--load_4bit', action='store_true', default=False)
        parser.add_argument('--pool_size', type=int, default=0)
        
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_name', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
        parser.add_argument('--projection_dim', type=int, default=256)
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)

        parser.add_argument('--w_t2t', type=float, default=1.0, help='Weight for text-text loss')
        parser.add_argument('--w_align', type=float, default=1.0, help='Weight for align loss')
        parser.add_argument('--w_struct', type=float, default=0.5, help='Weight for structural consistency loss')
        

        parser.add_argument('--plm_lora_r', type=int, default=8)
        parser.add_argument('--plm_lora_alpha', type=int, default=8)
        parser.add_argument('--plm_lora_dropout', type=int, default=0.1)

        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.01, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=5e-5, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=3e-6, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=3e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--retrieval_eval_epoch', type=int, default=10)
        
        # Encoder selection (automatically inferred from plm_name but can be explicit)
        parser.add_argument('--encoder_type', type=str, default='auto', 
                            choices=['auto', 'esm2', 'esmc'],
                            help='Protein encoder type: auto (infer from plm_name), esm2 (HuggingFace ESM2), or esmc (official ESM-C package)')
        return parent_parser



@torch.no_grad()
def eval_retrieval_fullset(prot_feat, text_feat, device):    
    N = prot_feat.shape[0]
    B = 8
    text_feat = text_feat.to(device)
    sim_p2t = []
    for i in tqdm(range(0, N, B)):
        l_prot_feat = prot_feat[i:i+B].to(device)
        l_sim_q2t = (l_prot_feat.unsqueeze(1) @ text_feat.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [N, D, 1]; output shape = [B, N, num_qs]
        l_sim_p2t, _ = l_sim_q2t.max(-1) # shape = [B, N]
        sim_p2t.append(l_sim_p2t)
    sim_p2t = torch.cat(sim_p2t, dim=0).cpu() # shape = [N, N]
    
    rank_p2t = []
    for i in range(0, N, B):
        sorted_ids = torch.argsort(sim_p2t[i:i+B].to(device), descending=True)
        rank_p2t.append((sorted_ids == torch.arange(i,i+sorted_ids.shape[0], device=device).reshape(-1, 1)).int().argmax(dim=-1))
    rank_p2t = torch.cat(rank_p2t, dim=0)
    
    rank_t2p = []
    for i in range(0, N, B):
        sorted_ids = torch.argsort(sim_p2t.T[i:i+B].to(device), descending=True)
        rank_t2p.append((sorted_ids == torch.arange(i,i+sorted_ids.shape[0], device=device).reshape(-1, 1)).int().argmax(dim=-1))
    rank_t2p = torch.cat(rank_t2p, dim=0)
    
    p2t_acc = float((rank_p2t == 0).float().mean())
    p2t_rec20 = float((rank_p2t < 20).float().mean())
    t2p_acc = float((rank_t2p == 0).float().mean())
    t2p_rec20 = float((rank_t2p < 20).float().mean())
    p2t_acc = round(p2t_acc * 100, 2)
    p2t_rec20 = round(p2t_rec20 * 100, 2)
    t2p_acc = round(t2p_acc * 100, 2)
    t2p_rec20 = round(t2p_rec20 * 100, 2)
    return p2t_acc, p2t_rec20, t2p_acc, t2p_rec20, sim_p2t


@torch.no_grad()
def eval_retrieval_fullset_for_rerank(model, sim_p2t_total, prot_embed_total, prot_mask_total, text_total, text_mask_total, rerank_cand_num, device):
    N = sim_p2t_total.shape[0]
    B = 4    
    rcn = rerank_cand_num ## re-rank candidate numbers
    
    hit_p2t = []
    for i in tqdm(range(0, N, B), desc='re-ranking p2t'):
        sim = sim_p2t_total[i:i+B].to(device)
        rB = sim.shape[0] # real batch size
        topk_sim, topk_idx = sim.topk(k=rcn, dim=1) # shape = [B, rcn]
        topk_idx = topk_idx.cpu()
        prot_embed = prot_embed_total[i:i+B].to(device).repeat_interleave(rcn, 0) # shape = [B * rcn, num_qs, D]
        prot_mask = prot_mask_total[i:i+B].to(device).repeat_interleave(rcn, 0) # shape = [B * rcn, num_qs, D]
        text = text_total[topk_idx].flatten(0,1).to(device) # shape = [B * rcn, text_len]
        text_mask = text_mask_total[topk_idx].flatten(0,1).to(device) # shape = [B * rcn, text_len]
        ptm_sim = model.compute_ptm(prot_embed, prot_mask, text, text_mask).reshape(rB, rcn) ## fixme, using the linear clf's logits directly, without softmax
        sorted_ids = torch.argsort(topk_sim + ptm_sim, descending=True).cpu() # shape = [B, rcn]
        # sorted_ids = torch.argsort(gtm_sim, descending=True).cpu() # shape = [B, rcn]
        sorted_ids = torch.gather(topk_idx, 1, sorted_ids) # mapping to original ids
        hit_p2t.append((sorted_ids == torch.arange(i,i+rB).reshape(-1, 1)).int())
    
    hit_p2t = torch.cat(hit_p2t, dim=0) # shape = [N, rcn]
    # p2t_acc = float((hit_p2t[:, 0]).float().mean())
    # p2t_rec20 = float((hit_p2t[:, :20]).float().sum() / N)
    # print(p2t_acc, p2t_rec20)

    hit_t2p = []
    sim_t2p_total = sim_p2t_total.T
    for i in tqdm(range(0, N, B), desc='re-ranking t2p'):
        sim = sim_t2p_total[i:i+B].to(device)
        rB = sim.shape[0]
        topk_sim, topk_idx = sim.topk(k=rcn, dim=1)
        topk_idx = topk_idx.cpu()
        text = text_total[i:i+B].to(device).repeat_interleave(rcn, 0)
        text_mask = text_mask_total[i:i+B].to(device).repeat_interleave(rcn, 0)
        prot_embed = prot_embed_total[topk_idx].to(device).flatten(0,1)
        prot_mask = prot_mask_total[topk_idx].to(device).flatten(0,1)
        ptm_sim = model.compute_ptm(prot_embed, prot_mask, text, text_mask).reshape(rB, rcn)
        sorted_ids = torch.argsort(topk_sim + ptm_sim, descending=True).cpu() # shape = [B, rcn]
        sorted_ids = torch.gather(topk_idx, 1, sorted_ids)
        hit_t2p.append((sorted_ids == torch.arange(i,i+sorted_ids.shape[0]).reshape(-1, 1)).int())
    hit_t2p = torch.cat(hit_t2p, dim=0)
    
    p2t_acc = float((hit_p2t[:, 0]).float().mean())
    p2t_rec20 = float((hit_p2t[:, :20]).float().sum() / N)
    t2p_acc = float((hit_t2p[:, 0]).float().mean())
    t2p_rec20 = float((hit_t2p[:, :20]).float().sum() / N)
    p2t_acc = round(p2t_acc * 100, 2)
    p2t_rec20 = round(p2t_rec20 * 100, 2)
    t2p_acc = round(t2p_acc * 100, 2)
    t2p_rec20 = round(t2p_rec20 * 100, 2)
    return p2t_acc, p2t_rec20, t2p_acc, t2p_rec20


@torch.no_grad()
def eval_retrieval_inbatch_with_rerank(model, dataloader, device=None):
    '''
    include rerank
    '''
    assert isinstance(model, Blip2Qformer)
    pad_token_id = model.tokenizer.pad_token_id
    model.eval()
    p2t_acc = 0
    t2p_acc = 0
    p2t_rec20 = 0
    t2p_rec20 = 0
    allcnt = 0
    
    p2t_rerank_acc = 0
    t2p_rerank_acc = 0
    p2t_rerank_rec20 = 0
    t2p_rerank_rec20 = 0

    prot_feat_total = []  
    text_feat_total = []
    
    prot_embed_total = [] 
    prot_mask_total = []
    
    text_total = []
    text_mask_total = []
    
    for batch in tqdm(dataloader):
        prot_batch, text_batch = batch
        prot_batch, text_batch = prot_batch.to(device), text_batch.to(device)
        text_total.append(text_batch.input_ids)
        text_mask_total.append(text_batch.attention_mask)

        
        prot_feats, prot_embeds = model.prot_forward(prot_batch) # shape = [B, num_qs, D]
        text_feats = model.text_forward(text_batch) # shape = [B, D]
        

        sim_q2t = (prot_feats.unsqueeze(1) @ text_feats.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_p2t, _ = sim_q2t.max(-1) # shape = [B, B]

        B = sim_p2t.shape[0]
        sorted_ids = sim_p2t.argsort(descending=True).cpu()
        p2t_rank = (sorted_ids == torch.arange(B).reshape(-1, 1)).int().argmax(dim=-1)
        sorted_ids = sim_p2t.T.argsort(descending=True).cpu()
        t2p_rank = (sorted_ids == torch.arange(B).reshape(-1, 1)).int().argmax(dim=-1)
        
        p2t_acc += float((p2t_rank == 0).sum())
        t2p_acc += float((t2p_rank == 0).sum())
        p2t_rec20 += float((p2t_rank < 20).sum())
        t2p_rec20 += float((t2p_rank < 20).sum())

        allcnt += B

        prot_feat_total.append(prot_feats.cpu())
        text_feat_total.append(text_feats.cpu())
        prot_embed_total.append(prot_embeds.cpu())
        prot_mask_total.append(prot_batch.attention_mask.cpu())

        ## reranking
        prot_embeds = prot_embeds.repeat_interleave(B, 0) # shape = [B * B, prot_len, D]
        prot_mask = prot_batch.attention_mask.repeat_interleave(B, 0) # shape = [B * B, prot_len]
        text_ids = text_batch.input_ids.repeat(B, 1) # shape = [B * B, text_len]
        text_mask = text_batch.attention_mask.repeat(B, 1) # shape = [B * B, text_len]

        ## batched reranking
        batch_size = 64
        ptm_sim = []
        for i in range(0, prot_embeds.shape[0], batch_size):
            ptm_sim_local = model.compute_ptm(prot_embeds[i:i+batch_size], prot_mask[i:i+batch_size], text_ids[i:i+batch_size], text_mask[i:i+batch_size])
            ptm_sim.append(ptm_sim_local)
        ptm_sim = torch.cat(ptm_sim, dim=0).reshape(B, B)

        rerank_sim = sim_p2t + ptm_sim

        ## p2t rerank
        sorted_ids = torch.argsort(rerank_sim, descending=True).cpu() # shape = [B, B]
        hit_p2t = (sorted_ids == torch.arange(B).reshape(-1, 1)).float()
        p2t_rerank_acc += float(hit_p2t[:, 0].sum())
        p2t_rerank_rec20 += float(hit_p2t[:, :20].sum())
        
        ## t2p rerank
        sorted_ids = torch.argsort(rerank_sim.T, descending=True).cpu() # shape = [B, B]
        hit_t2p = (sorted_ids == torch.arange(B).reshape(-1, 1)).float()
        t2p_rerank_acc += float(hit_t2p[:, 0].sum())
        t2p_rerank_rec20 += float(hit_t2p[:, :20].sum())

    prot_feat_total = torch.cat(prot_feat_total, dim=0)
    text_feat_total = torch.cat(text_feat_total, dim=0)
    prot_embed_total = pad_and_concat(prot_embed_total)
    prot_mask_total = pad_and_concat(prot_mask_total)
    text_total = pad_and_concat(text_total, fill_value=pad_token_id)
    text_mask_total = pad_and_concat(text_mask_total)
    # # text_total = torch.cat(text_total, dim=0)
    # text_mask_total = torch.cat(text_mask_total, dim=0)

    p2t_acc = round(p2t_acc/allcnt * 100, 2)
    t2p_acc = round(t2p_acc/allcnt * 100, 2)
    p2t_rec20 = round(p2t_rec20 / allcnt * 100, 2)
    t2p_rec20 = round(t2p_rec20 / allcnt * 100, 2)

    p2t_rerank_acc = round(p2t_rerank_acc / allcnt * 100, 2)
    t2p_rerank_acc = round(t2p_rerank_acc / allcnt * 100, 2)
    p2t_rerank_rec20 = round(p2t_rerank_rec20 / allcnt * 100, 2)
    t2p_rerank_rec20 = round(t2p_rerank_rec20 / allcnt * 100, 2)
    return p2t_acc, t2p_acc, p2t_rec20, t2p_rec20, \
        p2t_rerank_acc, t2p_rerank_acc, p2t_rerank_rec20, t2p_rerank_rec20, \
        prot_feat_total, text_feat_total, prot_embed_total, prot_mask_total, text_total, text_mask_total


