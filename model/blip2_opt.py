"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Mapping
from torch.cuda.amp import autocast as autocast
# from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from lavis.models.blip2_models.blip2 import disabled_train
from model.blip2 import Blip2Base
from transformers import AutoTokenizer
from transformers import OPTForCausalLM
from opendelta import LoraModel
from torch.cuda.amp import autocast
from opendelta.delta_models.lora import LoraConfig as DeltaLoraConfig
from transformers import BertTokenizer, BitsAndBytesConfig
from transformers.tokenization_utils_base import BatchEncoding
from model.help_funcs import hf_enable_gradient_checkpointing

opt_model_list = [
    "facebook/galactica-125m",
    "facebook/galactica-1.3b",
    "facebook/galactica-6.7b",
    "facebook/galactica-30b",
]

def get_gpu_memory(device=0):
    # t = torch.cuda.get_device_properties(device).total_memory
    # r = torch.cuda.memory_reserved(device)
    # a = torch.cuda.memory_allocated(device)
    # f = r-a  # free inside reserved
    free, total = torch.cuda.mem_get_info(device)
    free = free / (1024 ** 3)
    total = total / (1024 ** 3)
    return free, total-free, total

def mask_by_len(input, lens, fill_value=0):
    '''
    input: shape = [N, D]
    lens: shape = [N]
    '''
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input


class Blip2OPT(Blip2Base):
    """
    BLIP2 model for protein function prediction with Q-former.
    Supports both generation (function text) and classification (GO terms).
    """
    def __init__(
        self,
        bert_name,
        num_query_token=32,
        cross_attention_freq=2,
        plm_model="facebook/esm2_t30_150M_UR50D",
        plm_tune='freeze',
        llm_name="facebook/galactica-1.3b",
        llm_tune='freeze',
        peft_dir='',
        args=None,
    ):
        super().__init__()
        self.args = args
        self.head = args.head
        self.id2go = args.id2go
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_inference_len = args.max_inference_len
        self.min_inference_len = args.min_inference_len
        self.enbale_gradient_checkpointing = args.enbale_gradient_checkpointing
        self.plm_lora_r = args.plm_lora_r
        self.plm_lora_alpha = args.plm_lora_alpha
        self.plm_lora_dropout = args.plm_lora_dropout

        # Detect encoder type based on model name
        self.plm_model = plm_model
        self.is_esmc = str(plm_model).startswith('esmc_')
        
        self.plm_tokenizer, self.plm, self.ln_layer = self.init_protein_encoder(plm_model)
        self.plm_tune = plm_tune
        if plm_tune == 'freeze':
            for name, param in self.plm.named_parameters():
                param.requires_grad = False
            self.plm = self.plm.eval()
            self.plm.train = disabled_train
            logging.info("freeze plm encoder")
        elif plm_tune == 'lora':
            plm_cfg = DeltaLoraConfig(self.plm_lora_r, 
                                    self.plm_lora_alpha, 
                                    self.plm_lora_dropout,
                                    modified_modules=["attn.layernorm_qkv.1","ffn.1"])
            self.plm_delta = LoraModel.from_config(plm_cfg, self.plm)
            self.plm_delta.freeze_module(set_state_dict=False)
            self.plm_delta.log()
        else:
            raise NotImplementedError()
        
        self.num_query_token = num_query_token
        _, self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.plm.num_features, cross_attention_freq)
        ### remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        ############################attention!!
        for p in self.ln_layer.parameters():
            p.requires_grad = False
        for p in self.plm.parameters():
            p.requires_grad = False     
        ############################attention!!

        ## initialize opt model
        self.llm_model, self.llm_tokenizer = self.load_llm(llm_name, load_4bit=False, enable_gradient_checkpointing=self.enbale_gradient_checkpointing)

        if llm_tune == 'freeze':
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        elif llm_tune == 'full':
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = True
        elif llm_tune == 'lora':
            lora_config = DeltaLoraConfig(args.lora_r, 
                                          args.lora_alpha, 
                                          args.lora_dropout,
                                          )
            self.lm_delta = LoraModel.from_config(lora_config, self.llm_model)
            self.lm_delta.freeze_module(set_state_dict=False)
            self.lm_delta.log()
        elif llm_tune == 'mid_lora':
            lora_config = DeltaLoraConfig(args.lora_r, args.lora_alpha, args.lora_dropout, modified_modules=["q_proj", "v_proj", 'k_proj', "out_proj", "fc1", "fc2"])
            self.delta = LoraModel.from_config(lora_config, self.llm_model)
            self.delta.freeze_module(set_state_dict=False)
            self.delta.log()
        elif llm_tune == 'peft_lora':
            config = PeftLoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                # target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm_model = get_peft_model(self.llm_model, config)
            for name, module in self.llm_model.named_modules():
                if isinstance(module, LoraLayer):
                    if True:
                        module = module.to(torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if True and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)
        else:
            raise NotImplementedError()

        ## fixme: this is different from the original BLIP2
        self.eos_token_id = self.llm_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]
        self.opt_proj = nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size)

        # Enhanced reliability head with multi-feature fusion
        # Input: [weighted_mean, max_pool, protein_features]
        # Note: prot_feats is already projected to llm_hidden_size via opt_proj
        # Dimension: llm_hidden * 3 (weighted_mean + max_pool + projected_prot_feats)
        reliability_input_dim = self.llm_model.config.hidden_size * 3
        self.reliability_head = nn.Sequential(
            nn.Linear(reliability_input_dim, self.llm_model.config.hidden_size),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.LayerNorm(self.llm_model.config.hidden_size),
            nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size//4),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(self.llm_model.config.hidden_size//4, 1),
        )
        self.reliability_head = self.reliability_head.to(torch.float32)

        if self.head == 'classification':
            #self.opt_proj = nn.Linear(self.Qformer.config.hidden_size, len(args.go2id))
            hidden = self.Qformer.config.hidden_size
            self.opt_proj = nn.Sequential(
                # block 1
                nn.Linear(hidden, hidden),      # keep feature dim
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.1),

                # block 2
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.1),

                # final projection to classes
                nn.Linear(hidden, len(args.go2id)),
            )


    def load_llm(self, llm_model, load_4bit=False, enable_gradient_checkpointing=True):
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, padding_side='right')
        llm_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        if load_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
            ## Automatic device selection based on CUDA_VISIBLE_DEVICES
            import os
            visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
            device_ids = [int(d) for d in visible_devices.split(',') if d.strip()]
            # Use first available device, or second if first is heavily used
            outputs = get_gpu_memory(device_ids[0] if device_ids else 0)
            used_memory = outputs[1]
            if used_memory > 1 and len(device_ids) > 1:
                device_map = {"": device_ids[1]}
            else:
                device_map = {"": device_ids[0] if device_ids else 0}
            llm_model = OPTForCausalLM.from_pretrained(
                llm_model, 
                quantization_config=quant_config,
                load_in_4bit=True,
                load_in_8bit=False,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
            )
            llm_model.resize_token_embeddings(len(llm_tokenizer)) ## this will cause bug when 
            llm_model = prepare_model_for_kbit_training(llm_model, use_gradient_checkpointing=True)
        else:
            llm_model = OPTForCausalLM.from_pretrained(llm_model, torch_dtype=torch.bfloat16)
            llm_model.resize_token_embeddings(len(llm_tokenizer)) ## this will cause bug when 
            if enable_gradient_checkpointing:
                llm_model = hf_enable_gradient_checkpointing(llm_model)
        return llm_model, llm_tokenizer


    def _normalize_batch(self, batch):
        """Normalize batch to dict format."""
        if isinstance(batch, (Mapping, BatchEncoding)):
            return batch
        if isinstance(batch, (list, tuple)):
            for x in batch:
                if isinstance(x, (Mapping, BatchEncoding)) and "input_ids" in x:
                    return x
        raise TypeError(f"Batch must be a mapping with input_ids; got {type(batch)}")

    def prot_encode(self, prot_batch):
        """Encode protein sequence to embeddings."""
        prot_batch = self._normalize_batch(prot_batch)

        device = next(self.plm.parameters()).device
        raw_ids = prot_batch.get("input_ids")
        raw_mask = prot_batch.get("attention_mask", None)
        pad_id = getattr(self.plm_tokenizer, "pad_token_id", 1)

        # Convert to tensor and move to device
        if torch.is_tensor(raw_ids):
            input_ids = raw_ids.to(device)
            attention_mask = raw_mask.to(device) if torch.is_tensor(raw_mask) else None
        elif isinstance(raw_ids, np.ndarray):
            input_ids = torch.as_tensor(raw_ids, dtype=torch.long, device=device)
            attention_mask = torch.as_tensor(raw_mask, dtype=torch.long, device=device) if raw_mask is not None else None
        elif isinstance(raw_ids, list):
            # Handle list input (ESM-C path)
            ids_list = [raw_ids] if (not raw_ids or not isinstance(raw_ids[0], list)) else raw_ids
            Lfix = getattr(self, "prot_max_len", None) or max(len(x) for x in ids_list)
            
            padded, masks = [], []
            for ids in ids_list:
                curr_len = len(ids)
                ids = ids[:Lfix] + [pad_id] * max(0, Lfix - curr_len)
                padded.append(ids)
                masks.append([1] * min(curr_len, Lfix) + [0] * max(0, Lfix - curr_len))
            
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
        else:
            raise TypeError(f"Unsupported input_ids type: {type(raw_ids)}")

        if attention_mask is None:
            attention_mask = (input_ids != pad_id).long()

        # Get protein embeddings
        out = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        prot_embeds = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        
        if self.plm_tune == "freeze":
            prot_embeds = prot_embeds.detach()
        
        # Match dtype with ln_layer for inference compatibility
        ln_dtype = next(self.ln_layer.parameters()).dtype
        if prot_embeds.dtype != ln_dtype:
            prot_embeds = prot_embeds.to(ln_dtype)
        
        return self.ln_layer(prot_embeds), attention_mask

    def prot_qformer(self, prot_embeds, attention_mask):
        """Apply Q-former to protein embeddings."""
        B = prot_embeds.size(0)
        query_tokens = self.query_tokens.expand(B, -1, -1)
        
        q_out = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=prot_embeds,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )
        
        prot_tokens = self.opt_proj(q_out.last_hidden_state)
        prot_feats = F.normalize(prot_tokens.mean(dim=1), dim=-1, p=2)
        return prot_tokens, prot_feats

    def _extract_reliability_features(self, last_hidden, attention_mask, prot_feats, protein_token_count):
        """
        Extract enhanced features for reliability prediction using multi-pooling strategy.
        
        Args:
            last_hidden: Hidden states from LLM [B, seq_len, hidden_size]
            attention_mask: Attention mask for valid tokens [B, seq_len]
            prot_feats: Normalized protein features (already projected to llm_hidden_size) [B, hidden_size]
            protein_token_count: Number of protein tokens (Q) to skip
            
        Returns:
            Combined feature tensor [B, hidden_size*3] = [weighted_mean, max_pool, prot_feats]
        """
        last_hidden_fp32 = last_hidden.to(torch.float32)
        B, seq_len, hidden_size = last_hidden_fp32.shape
        
        # Skip protein tokens, only use text tokens for reliability features
        text_hidden = last_hidden_fp32[:, protein_token_count:, :]  # [B, text_len, H]
        text_mask = attention_mask[:, protein_token_count:]  # [B, text_len]
        
        # Handle edge case: no text tokens (only protein tokens)
        if text_hidden.shape[1] == 0:
            # Use protein features only, duplicate to match expected dimension
            prot_feat_fp32 = prot_feats.to(torch.float32).detach()
            # Return: [prot_feat, prot_feat, prot_feat] to maintain dimension
            return torch.cat([prot_feat_fp32, prot_feat_fp32, prot_feat_fp32], dim=-1).detach()
        
        # 1. Weighted mean pooling (masked average over valid tokens)
        text_mask_expanded = text_mask.unsqueeze(-1).to(torch.float32)  # [B, text_len, 1]
        hidden_masked = text_hidden * text_mask_expanded
        seq_lengths = text_mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
        weighted_mean = hidden_masked.sum(dim=1) / seq_lengths  # [B, H]
        
        # 2. Max pooling over valid tokens
        # Mask out padding with very negative values before max pooling
        masked_for_max = text_hidden.clone()
        masked_for_max[text_mask_expanded.expand_as(text_hidden) == 0] = -1e9
        max_pool = masked_for_max.max(dim=1)[0]  # [B, H]
        
        # 3. Include protein features (already normalized, projected to llm_hidden_size, and detached)
        prot_feat_fp32 = prot_feats.to(torch.float32).detach()  # [B, H]
        
        # 4. Concatenate all features
        combined_features = torch.cat([
            weighted_mean,
            max_pool,
            prot_feat_fp32
        ], dim=-1)  # [B, H*3]
        
        return combined_features.detach()

    def _focal_loss_with_logits(self, pred, target, alpha=0.25, gamma=2.0):
        """
        Focal Loss for binary classification to handle class imbalance.
        
        Args:
            pred: Predictions (logits) [B]
            target: Ground truth [B], values in [0, 1]
            alpha: Weight for positive class (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
            
        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)  # Probability of correct class
        focal_weight = (1 - pt) ** gamma
        
        # Apply alpha weighting
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * focal_weight * bce_loss
        
        return loss.mean()

    def forward(self, batch, return_pred=False):
        """Forward pass for training."""
        if self.head == 'generation':
            prot_batch, text_batch, prompt_batch, r_tensor = batch
            samples = {'prot_batch': prot_batch, 'prompt_batch': prompt_batch, 'reliability': r_tensor}
            
            # Encode protein
            prot_embeds, prot_attn = self.prot_encode(prot_batch)
            prot_tokens, prot_feats = self.prot_qformer(prot_embeds, prot_attn)
            device = prot_embeds.device
            
            # Process text batch
            text_batch = self._normalize_batch(text_batch)

            input_ids = text_batch["input_ids"].to(device)
            attention_mask = text_batch["attention_mask"].to(device)
            token_type_ids = text_batch.get("token_type_ids")
            
            # Prepare targets
            targets = input_ids.masked_fill(input_ids == self.llm_tokenizer.pad_token_id, -100)
            if token_type_ids is not None:
                targets = targets.masked_fill(token_type_ids.to(device) == 0, -100)

            # Prepare inputs for LLM
            B, Q = prot_tokens.shape[:2]
            prot_mask = torch.ones((B, Q), dtype=attention_mask.dtype, device=device)
            prot_empty_targets = torch.full((B, Q), -100, dtype=torch.long, device=device)
            
            # Get text embeddings and match dtype with prot_tokens
            text_embeds = self.llm_model.get_input_embeddings()(input_ids)
            if prot_tokens.dtype != text_embeds.dtype:
                prot_tokens = prot_tokens.to(text_embeds.dtype)
            
            inputs_embeds = torch.cat([prot_tokens, text_embeds], dim=1)
            full_attn = torch.cat([prot_mask, attention_mask], dim=1)
            targets = torch.cat([prot_empty_targets, targets], dim=1)
            
            # Forward through LLM
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attn,
                return_dict=True,
                labels=targets,
                output_hidden_states=True
            )
            
            # Compute reliability score with enhanced feature extraction
            last_hidden = outputs.hidden_states[-1]
            last_hidden = last_hidden[0] if isinstance(last_hidden, (tuple, list)) else last_hidden
            
            # Extract enhanced reliability features
            text_feat = self._extract_reliability_features(
                last_hidden=last_hidden,
                attention_mask=full_attn,
                prot_feats=prot_feats,
                protein_token_count=Q
            )
            
            with autocast(enabled=False):
                head_param = next(self.reliability_head.parameters())
                text_feat = text_feat.to(device=head_param.device, dtype=head_param.dtype)
                r_pred = self.reliability_head(text_feat).squeeze(-1)
                r_target = r_tensor.to(device=r_pred.device, dtype=r_pred.dtype).view(-1)
                
                # Use focal loss if enabled, otherwise standard BCE
                use_focal = getattr(self.args, 'use_focal_loss', False)
                if use_focal:
                    focal_alpha = getattr(self.args, 'focal_alpha', 0.25)
                    focal_gamma = getattr(self.args, 'focal_gamma', 2.0)
                    r_loss = self._focal_loss_with_logits(r_pred, r_target, focal_alpha, focal_gamma)
                else:
                    loss_fn = torch.nn.BCEWithLogitsLoss()
                    r_loss = loss_fn(r_pred, r_target)

            if return_pred:
                with torch.no_grad():
                    pred_texts, r_pred, conf, _ = self.generate(
                        samples,
                        do_sample=self.do_sample,
                        num_beams=self.num_beams,
                        max_length=self.max_inference_len,
                        min_length=self.min_inference_len
                    )
                return outputs.loss, r_loss, pred_texts
            return outputs.loss, r_loss
        
        # Classification head
        prot_batch, go_class = batch
        prot_embeds, prot_attn = self.prot_encode(prot_batch)
        prot_tokens, _ = self.prot_qformer(prot_embeds, prot_attn)
        logits = self.opt_proj(prot_tokens.mean(dim=1))
        return F.binary_cross_entropy_with_logits(logits, go_class)

    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=False,
        num_beams=3,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.3,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        prot_batch = samples['prot_batch']
        prompt_batch = samples['prompt_batch']
        
        # Encode protein sequence
        prot_embeds, prot_attn = self.prot_encode(prot_batch)
        prot_tokens, prot_feats = self.prot_qformer(prot_embeds, prot_attn)
        device = prot_embeds.device

        # Prepare prompt embeddings
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_batch.input_ids)
        
        # Match dtype between prot_tokens and prompt_embeds for compatibility
        if prot_tokens.dtype != prompt_embeds.dtype:
            prot_tokens = prot_tokens.to(prompt_embeds.dtype)
        
        inputs_embeds = torch.cat((prot_tokens, prompt_embeds), dim=1)
        prot_mask = torch.ones(prot_tokens.shape[:2], dtype=prompt_batch.attention_mask.dtype, device=prompt_embeds.device)
        attention_mask = torch.cat([prot_mask, prompt_batch.attention_mask], dim=1)

        gen_out = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            # pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True, 
            output_hidden_states=True
        )

        texts = self.llm_tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)
        texts = [t.strip() for t in texts]

        # Compute reliability score with enhanced feature extraction
        # Note: gen_out.hidden_states from generate() has different structure than forward()
        # We need to run a forward pass with the generated sequences to get proper hidden states
        
        B_gen = prot_tokens.shape[0]
        Q_gen = prot_tokens.shape[1]
        
        # When using inputs_embeds, gen_out.sequences contains only the generated token IDs
        # (not including the input embeddings part)
        generated_token_ids = gen_out.sequences
        
        # Only compute reliability if we have generated tokens
        if generated_token_ids.shape[1] > 0 and (generated_token_ids != self.llm_tokenizer.pad_token_id).any():
            # Forward pass with generated sequence to get proper hidden states
            generated_embeds = self.llm_model.get_input_embeddings()(generated_token_ids)
            
            # Concatenate: prot_tokens + prompt_embeds + generated_embeds
            full_embeds = torch.cat([prot_tokens, prompt_embeds, generated_embeds], dim=1)
            
            # Create proper attention mask
            gen_mask = torch.ones((B_gen, generated_token_ids.shape[1]), 
                                 dtype=attention_mask.dtype, device=device)
            full_attention_mask = torch.cat([attention_mask, gen_mask], dim=1)
            
            # Forward pass to get hidden states
            with torch.no_grad():
                outputs = self.llm_model(
                    inputs_embeds=full_embeds,
                    attention_mask=full_attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                last_hidden = outputs.hidden_states[-1]
            
            # Extract enhanced reliability features
            text_feat = self._extract_reliability_features(
                last_hidden=last_hidden,
                attention_mask=full_attention_mask,
                prot_feats=prot_feats,
                protein_token_count=Q_gen
            )
            
            with autocast(enabled=False):
                head_param = next(self.reliability_head.parameters())
                text_feat = text_feat.to(device=head_param.device, dtype=head_param.dtype)
                r_pred = torch.sigmoid(self.reliability_head(text_feat).squeeze(-1))
        else:
            # No generated tokens, return default reliability score
            r_pred = torch.ones(B_gen, device=device) * 0.5
        
        # Compute confidence from transition scores
        ts = self.llm_model.compute_transition_scores(
            gen_out.sequences, gen_out.scores,
            beam_indices=getattr(gen_out, "beam_indices", None),
            normalize_logits=True
        )
        conf = torch.exp(ts.to(torch.float32).mean(dim=1)).clamp(1e-9, 1.0).tolist()
        
        # Return embeddings for analysis
        emb_out = {
            "plm_mean_fp16": prot_embeds.to(torch.float32).mean(dim=1).detach().to(torch.float16).cpu(),
            "qformer_feats_fp16": prot_feats.detach().to(torch.float16).cpu()
        }
        return texts, r_pred, conf, emb_out

    @torch.no_grad()
    def inferred_class(
        self,
        samples,
        threshold: float = 0.5,
        min_k: int = 1,
        aggregation: str = "mean",   # "mean" | "min"
    ):
        """
        Returns:
            preds_go: List[List[str]]   # predicted GO ids per sample
            confidences: List[float]    # one scalar confidence per sample in [0,1]
        """
        prot_batch = samples['prot_batch']
        prot_embeds, prot_attn = self.prot_encode(prot_batch)
        prot_tokens, _ = self.prot_qformer(prot_embeds, prot_attn)
        
        # Get probabilities
        probs = torch.sigmoid(prot_tokens.mean(dim=1))
        B, C = probs.shape
        
        # Apply threshold and ensure top-k
        mask = probs >= threshold
        K = max(1, int(min_k))
        _, topk_idx = torch.topk(probs, k=K, dim=1)
        for i in range(B):
            mask[i, topk_idx[i]] = True
        
        # Build predictions and confidences
        preds_go = [] if self.id2go else None
        confidences = []
        
        for i in range(B):
            idx = torch.nonzero(mask[i], as_tuple=False).view(-1)
            idx = idx if idx.numel() > 0 else topk_idx[i]
            sel_probs = probs[i, idx]
            
            conf = sel_probs.min() if aggregation == "min" else sel_probs.mean()
            confidences.append(float(conf.item()))
            
            if preds_go is not None:
                preds_go.append([self.id2go[j.item()] for j in idx])
        
        return preds_go, confidences
