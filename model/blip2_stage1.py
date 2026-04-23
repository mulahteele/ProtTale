import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blip2qformer import Blip2Qformer
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from tqdm import tqdm
from torchmetrics.functional import auroc
from model.help_funcs import AttrDict, pad_and_concat, l2_normalize, cosine_matrix, align_loss, struct_consistency, auc_from_scores
from typing import Any, Dict


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
        
        self.w_pairwise_training = args.w_pairwise_training
        self.w_pointwise_align = args.w_pointwise_align
        self.w_pairwise_align = args.w_pairwise_align
        self.temperature = args.temperature
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass
        

    def maybe_autocast(self, dtype=torch.bfloat16):
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
        self.student_val_seq_sim = []
        self.student_val_labels = []
        self.student_test_seq_sim = []
        self.student_test_labels = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        idx==0 (VAL): collect student seq-seq similarity scores and labels for validation set.
        idx==1 (TEST): same for test set. Only student (protein) branch is evaluated.
        """
        prot_tokens1, prot_tokens2, text_tokens1, text_tokens2, labels = batch

        seq_emb1, _ = self.blip2qformer.prot_forward(prot_tokens1)
        seq_emb2, _ = self.blip2qformer.prot_forward(prot_tokens2)
        seq_emb1 = F.normalize(seq_emb1, dim=-1, eps=1e-8)
        seq_emb2 = F.normalize(seq_emb2, dim=-1, eps=1e-8)

        labels = labels.to(seq_emb1.device).float().view(-1)
        sim_ss = (seq_emb1 * seq_emb2).sum(dim=-1)

        if dataloader_idx % 2 == 0:
            self.student_val_seq_sim.append(sim_ss.detach().cpu())
            self.student_val_labels.append(labels.detach().cpu())
        else:
            self.student_test_seq_sim.append(sim_ss.detach().cpu())
            self.student_test_labels.append(labels.detach().cpu())


    def on_validation_epoch_end(self) -> None:
        auc_val = torch.tensor(float('nan'))
        if hasattr(self, "student_val_seq_sim") and len(self.student_val_seq_sim) > 0:
            sim_all_val = torch.cat(self.student_val_seq_sim, dim=0)
            lab_all_val = torch.cat(self.student_val_labels, dim=0)
            auc_val = auc_from_scores(sim_all_val, lab_all_val)

        auc_test = torch.tensor(float('nan'))
        if hasattr(self, "student_test_seq_sim") and len(self.student_test_seq_sim) > 0:
            sim_all_test = torch.cat(self.student_test_seq_sim, dim=0)
            lab_all_test = torch.cat(self.student_test_labels, dim=0)
            auc_test = auc_from_scores(sim_all_test, lab_all_test)

        if self.trainer.global_rank == 0:
            self.log_dict(
                {
                    "val/seqseq_AUC_VAL": auc_val,
                    "val/seqseq_AUC_TEST": auc_test,
                },
                prog_bar=True,
                sync_dist=False,
            )

        if hasattr(self, "student_val_seq_sim"):   self.student_val_seq_sim.clear()
        if hasattr(self, "student_val_labels"):   self.student_val_labels.clear()
        if hasattr(self, "student_test_seq_sim"): self.student_test_seq_sim.clear()
        if hasattr(self, "student_test_labels"):  self.student_test_labels.clear()



    def training_step(self, batch, batch_idx):
        self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        prot_tokens1, prot_tokens2, text_tokens1, text_tokens2, labels = batch
        batch_size = text_tokens1.input_ids.shape[0]

        text_emb1 = self.blip2qformer.text_forward(text_tokens1)
        text_emb2 = self.blip2qformer.text_forward(text_tokens2)

        text_emb1 = F.normalize(text_emb1, dim=-1, eps=1e-8)
        text_emb2 = F.normalize(text_emb2, dim=-1, eps=1e-8)

        temperature = getattr(self, "temperature", 0.07)
        labels = labels.to(text_emb1.device).float().view(-1)

        sim_tt = (text_emb1 * text_emb2).sum(dim=-1)
        logits_tt = sim_tt / temperature
        bce = nn.BCEWithLogitsLoss()
        loss_pairwise_training = bce(logits_tt, labels)

        with freeze_params(self.blip2qformer.Qformer):
            seq_emb1, _ = self.blip2qformer.prot_forward(prot_tokens1)
            seq_emb1 = F.normalize(seq_emb1, dim=-1, eps=1e-8)

            beta_align = getattr(self, "beta_align", 0.5)
            loss_pointwise_align = align_loss(seq_emb1, text_emb1, beta=beta_align)

            alpha_struct = getattr(self, "alpha_struct", 1.0)
            loss_pairwise_align = struct_consistency(seq_emb1, text_emb1, alpha=alpha_struct)

        w_pt = getattr(self, "w_pairwise_training", 1.0)
        w_pa = getattr(self, "w_pointwise_align", 1.0)
        w_pwa = getattr(self, "w_pairwise_align", 0.5)

        loss = w_pt * loss_pairwise_training + w_pa * loss_pointwise_align + w_pwa * loss_pairwise_align

        self.log_dict(
            {
                "train/loss_total": loss,
                "train/loss_pairwise_training": loss_pairwise_training,
                "train/loss_pointwise_align": loss_pointwise_align,
                "train/loss_pairwise_align": loss_pairwise_align,
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
        parser.add_argument('--plm_name', type=str, default='facebook/esm2_t30_150M_UR50D')
        parser.add_argument('--plm_tune', type=str, default='freeze')
        parser.add_argument('--load_4bit', action='store_true', default=False)
        parser.add_argument('--pool_size', type=int, default=0)
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_name', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
        parser.add_argument('--projection_dim', type=int, default=256)
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)

        parser.add_argument('--w_pairwise_training', type=float, default=1.0, help='Weight for pairwise training (text-text) loss')
        parser.add_argument('--w_pointwise_align', type=float, default=1.0, help='Weight for pointwise alignment loss (protein-text)')
        parser.add_argument('--w_pairwise_align', type=float, default=0.5, help='Weight for pairwise alignment (structural consistency) loss')
        

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
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--retrieval_eval_epoch', type=int, default=10)
        parser.add_argument('--encoder_type', type=str, default='auto', 
                            choices=['auto', 'esm2', 'esmc'],
                            help='Protein encoder type: auto (infer from plm_name), esm2 (HuggingFace ESM2), or esmc (official ESM-C package)')
        return parent_parser

