"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from opendelta import LoraModel
from opendelta.delta_models.lora import LoraConfig as DeltaLoraConfig

from lavis.models.blip2_models.blip2 import (
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput
from model.blip2 import Blip2Base
from model.help_funcs import pad_and_concat
from model.dist_funs import pl_concat_all_gather



class Blip2Qformer(Blip2Base):
    """BLIP2 first-stage model with Q-former and protein encoder."""
    def __init__(
        self,
        ptm,
        lm,
        bert_name,
        plm_name,
        temperature,
        plm_lora_r,
        plm_lora_alpha,
        plm_lora_dropout,
        plm_tune='freeze',
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        pool_size=0,
        load_4bit=False
    ):
        super().__init__()
        self.ptm = ptm
        self.lm = lm
        self.pool_size = pool_size
        self.plm_lora_r = plm_lora_r
        self.plm_lora_alpha = plm_lora_alpha
        self.plm_lora_dropout = plm_lora_dropout
        
        self.plm_tokenizer, self.plm, self.ln_layer = self.init_protein_encoder(plm_name, load_4bit)
        self.plm_tune = plm_tune
        if plm_tune == 'freeze':
            for name, param in self.plm.named_parameters():
                param.requires_grad = False
            self.plm = self.plm.eval()
            self.plm.train = disabled_train
            logging.info("freeze plm")
        elif plm_tune == 'lora':
            plm_cfg = DeltaLoraConfig(self.plm_lora_r, 
                                    self.plm_lora_alpha, 
                                    self.plm_lora_dropout,
                                    modified_modules=["attn.layernorm_qkv.1","ffn.1"])
            self.plm_delta = LoraModel.from_config(plm_cfg, self.plm)
            self.plm_delta.freeze_module(set_state_dict=False)
            self.plm_delta.log()
        elif plm_tune == 'full':
            for name, param in self.plm.named_parameters():
                param.requires_grad = True

        print("LoRA trainable:", sum(p.numel() for p in self.plm_delta.trainable_parameters()))
        print("All trainable:", sum(p.numel() for p in self.plm.parameters() if p.requires_grad))

        print('plm_tune',plm_tune)

        self.tokenizer, self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.plm.num_features, cross_attention_freq)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.prot_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.ptm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temperature = temperature



    def prot_forward(self, prot_batch):
        """Forward protein through encoder + Q-Former. Returns (prot_feats [B,D], prot_embeds [B,L,D])."""
        if isinstance(prot_batch, dict) and "input_ids" in prot_batch:
            pass
        else:
            prot_batch = next((x for x in prot_batch if isinstance(x, dict) and "input_ids" in x), None)
        if prot_batch is None:
            raise ValueError("prot_forward: prot_batch must be a dict with 'input_ids' or an iterable containing one")
        device = next(self.plm.parameters()).device

        def _pad_list_ids(ids_list, Lfix, pad_id=1):
            padded, mask = [], []
            for ids in ids_list:
                ids = ids[:Lfix]
                need = Lfix - len(ids)
                if need > 0: ids = ids + [pad_id] * need
                padded.append(ids)
                mask.append([1] * (Lfix - need) + [0] * need)
            return (torch.tensor(padded, dtype=torch.long, device=device),
                    torch.tensor(mask,   dtype=torch.long, device=device))

        raw_ids = prot_batch.get("input_ids")
        raw_mask = prot_batch.get("attention_mask", None)
        Lfix = int(getattr(self, "prot_max_len", 0) or 0) or None
        pad_id = getattr(getattr(self, "plm_tokenizer", None), "pad_token_id", 1)

        if torch.is_tensor(raw_ids):
            input_ids = raw_ids.to(device)
            attention_mask = raw_mask.to(device) if torch.is_tensor(raw_mask) else None
        else:
            # raw_ids is list of lists (e.g. from tokenizer fallback or other collater)
            if raw_ids and not isinstance(raw_ids[0], (list, tuple)):
                raw_ids = [raw_ids]
            if Lfix is None:
                Lfix = max(len(ids) if hasattr(ids, "__len__") else 0 for ids in raw_ids) or 1
            input_ids, attention_mask = _pad_list_ids(raw_ids, Lfix, pad_id)

        if attention_mask is None:
            attention_mask = (input_ids != pad_id).long()

        out = self.plm(input_ids=input_ids, attention_mask=attention_mask)

        prot_embeds = out.last_hidden_state

        if getattr(self, "plm_tune", "freeze") == "freeze":
            prot_embeds = prot_embeds.detach()

        prot_embeds = self.ln_layer(prot_embeds)

        B = prot_embeds.size(0)
        query_tokens = self.query_tokens.expand(B, -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=prot_embeds,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )

        prot_feats = self.prot_proj(query_output.last_hidden_state).mean(dim=1)
        prot_feats = F.normalize(prot_feats, dim=-1, p=2)
        return prot_feats, prot_embeds

    def text_forward(self, text_batch):
        text_output = self.Qformer.bert(**text_batch, return_dict=True)
        text_feats = self.text_proj(text_output.last_hidden_state).mean(dim=1)
        text_feats = F.normalize(text_feats, dim=-1, p=2)
        return text_feats