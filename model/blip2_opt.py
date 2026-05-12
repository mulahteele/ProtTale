"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Mapping
from torch.cuda.amp import autocast as autocast
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
    free, total = torch.cuda.mem_get_info(device)
    free = free / (1024 ** 3)
    total = total / (1024 ** 3)
    return free, total-free, total

# Reliability head: 4-class classification mapping
# Class indices: {-1: 0, 0: 1, 0.5: 2, 1: 3}
RELIABILITY_CLASSES = [-1.0, 0.0, 0.5, 1.0]
RELIABILITY_VAL_TO_IDX = {v: i for i, v in enumerate(RELIABILITY_CLASSES)}
RELIABILITY_NUM_CLASSES = len(RELIABILITY_CLASSES)

def _r_value_to_class(r_tensor):
    """Convert reliability float values to class indices."""
    class_indices = torch.zeros_like(r_tensor, dtype=torch.long)
    for val, idx in RELIABILITY_VAL_TO_IDX.items():
        class_indices[torch.isclose(r_tensor, torch.tensor(val, device=r_tensor.device, dtype=r_tensor.dtype), atol=1e-4)] = idx
    return class_indices

def _class_to_r_value(class_indices):
    """Convert class indices back to reliability float values."""
    classes = torch.tensor(RELIABILITY_CLASSES, dtype=torch.float32, device=class_indices.device)
    return classes[class_indices]


def _r_value_to_binary_class(r_tensor):
    """Binary mapping: 1 if r ~= 1.0 (positive class), else 0."""
    pos = torch.isclose(r_tensor, torch.tensor(1.0, device=r_tensor.device, dtype=r_tensor.dtype), atol=1e-4)
    return pos.long()


def _scan_r_counts(json_path):
    """Return (raw_counts_4class, total). raw_counts indexed by RELIABILITY_VAL_TO_IDX."""
    import json as _json
    counts = torch.zeros(RELIABILITY_NUM_CLASSES, dtype=torch.float32)
    if not json_path or not os.path.isfile(json_path):
        return counts, 0
    total = 0
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = _json.loads(line)
            r = float(row[2])
            total += 1
            for cls_val, idx in RELIABILITY_VAL_TO_IDX.items():
                if abs(r - cls_val) < 1e-4:
                    counts[idx] += 1
                    break
    return counts, total


def _inverse_freq_from_counts(counts):
    """Inverse-frequency weights w_k = N / (K_present * n_k), mean weight over
    present classes = 1. Empty classes get weight 0."""
    if counts.sum() == 0:
        return None
    nonzero = counts > 0
    weights = torch.zeros_like(counts)
    weights[nonzero] = counts.sum() / (nonzero.sum().float() * counts[nonzero])
    return weights


def _compute_inverse_freq_class_weights(json_path):
    """4-class inverse-frequency weights from JSON-lines file."""
    counts, total = _scan_r_counts(json_path)
    if total == 0:
        return None
    return _inverse_freq_from_counts(counts)


def _compute_binary_inverse_freq_class_weights(json_path):
    """Binary inverse-frequency weights. Index 0 = negative (r != 1), index 1 = positive (r == 1)."""
    counts, total = _scan_r_counts(json_path)
    if total == 0:
        return None
    pos = counts[RELIABILITY_VAL_TO_IDX[1.0]]
    neg = counts.sum() - pos
    binary_counts = torch.tensor([neg, pos], dtype=torch.float32)
    return _inverse_freq_from_counts(binary_counts)


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
    BLIP2 model for protein function prediction with Q-former (generation only).
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
        self.id2go = getattr(args, 'id2go', {})
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
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        for p in self.ln_layer.parameters():
            p.requires_grad = False
        for p in self.plm.parameters():
            p.requires_grad = False

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

        self.eos_token_id = self.llm_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]
        self.opt_proj = nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size)

        # Reliability head can be either 4-class ({-1, 0, 0.5, 1}) or binary (positive=r==1, else negative).
        self.reliability_binary = bool(getattr(args, 'reliability_binary', False))
        head_out_dim = 2 if self.reliability_binary else RELIABILITY_NUM_CLASSES
        reliability_input_dim = self.Qformer.config.hidden_size + self.plm.num_features
        self.reliability_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(reliability_input_dim, head_out_dim),
        )
        self.reliability_head = self.reliability_head.to(torch.float32)

        # Class weights for reliability cross-entropy. Computed from the reliability
        # finetune JSON when train_reliability_head_only is set; uniform otherwise.
        cw = None
        if getattr(args, 'train_reliability_head_only', False):
            json_path = getattr(args, 'reliability_finetune_data', '')
            if self.reliability_binary:
                cw = _compute_binary_inverse_freq_class_weights(json_path)
            else:
                cw = _compute_inverse_freq_class_weights(json_path)
            if cw is not None:
                mode_tag = 'binary' if self.reliability_binary else '4-class'
                logging.info(f"reliability {mode_tag} inverse-freq class weights: {cw.tolist()}")
        if cw is None:
            cw = torch.ones(head_out_dim, dtype=torch.float32)
        self.register_buffer('reliability_class_weights', cw)

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
            import os
            visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
            device_ids = [int(d) for d in visible_devices.split(',') if d.strip()]
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
            llm_model.resize_token_embeddings(len(llm_tokenizer))
            llm_model = prepare_model_for_kbit_training(llm_model, use_gradient_checkpointing=True)
        else:
            llm_model = OPTForCausalLM.from_pretrained(llm_model, torch_dtype=torch.bfloat16)
            llm_model.resize_token_embeddings(len(llm_tokenizer))
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

        if torch.is_tensor(raw_ids):
            input_ids = raw_ids.to(device)
            attention_mask = raw_mask.to(device) if torch.is_tensor(raw_mask) else None
        elif isinstance(raw_ids, np.ndarray):
            input_ids = torch.as_tensor(raw_ids, dtype=torch.long, device=device)
            attention_mask = torch.as_tensor(raw_mask, dtype=torch.long, device=device) if raw_mask is not None else None
        elif isinstance(raw_ids, list):
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

        out = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        prot_embeds = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        
        if self.plm_tune == "freeze":
            prot_embeds = prot_embeds.detach()

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

        qformer_raw = q_out.last_hidden_state  # [B, num_query, qformer_hidden]
        prot_tokens = self.opt_proj(qformer_raw)
        prot_feats = F.normalize(prot_tokens.mean(dim=1), dim=-1, p=2)
        # Mean-pooled Q-Former features before projection (for reliability head)
        qformer_feats = qformer_raw.mean(dim=1).detach()
        # Mean-pooled PLM embeddings (masked) for reliability head
        plm_pooled = (prot_embeds * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        plm_pooled = plm_pooled.detach()
        return prot_tokens, prot_feats, qformer_feats, plm_pooled

    def _extract_reliability_features(self, last_hidden, attention_mask, prot_feats, protein_token_count):
        """
        Extract features for reliability prediction: masked mean of LLM last hidden layer.

        Args:
            last_hidden: Hidden states from LLM [B, seq_len, hidden_size]
            attention_mask: Attention mask for valid tokens [B, seq_len]
            prot_feats: (unused, kept for API compatibility)
            protein_token_count: (unused, kept for API compatibility)

        Returns:
            Mean-pooled feature tensor [B, hidden_size]
        """
        last_hidden_fp32 = last_hidden.to(torch.float32)
        mask_expanded = attention_mask.unsqueeze(-1).to(torch.float32)
        hidden_masked = last_hidden_fp32 * mask_expanded
        seq_lengths = mask_expanded.sum(dim=1).clamp(min=1)
        mean_pool = hidden_masked.sum(dim=1) / seq_lengths
        return mean_pool.detach()

    def forward(self, batch, return_pred=False):
        """Forward pass for training (generation + reliability classification)."""
        prot_batch, text_batch, prompt_batch, r_tensor = batch
        samples = {'prot_batch': prot_batch, 'prompt_batch': prompt_batch, 'reliability': r_tensor}

        prot_embeds, prot_attn = self.prot_encode(prot_batch)
        prot_tokens, prot_feats, qformer_feats, plm_pooled = self.prot_qformer(prot_embeds, prot_attn)
        device = prot_embeds.device

        text_batch = self._normalize_batch(text_batch)

        input_ids = text_batch["input_ids"].to(device)
        attention_mask = text_batch["attention_mask"].to(device)
        token_type_ids = text_batch.get("token_type_ids")

        targets = input_ids.masked_fill(input_ids == self.llm_tokenizer.pad_token_id, -100)
        if token_type_ids is not None:
            targets = targets.masked_fill(token_type_ids.to(device) == 0, -100)

        B, Q = prot_tokens.shape[:2]
        prot_mask = torch.ones((B, Q), dtype=attention_mask.dtype, device=device)
        prot_empty_targets = torch.full((B, Q), -100, dtype=torch.long, device=device)

        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        if prot_tokens.dtype != text_embeds.dtype:
            prot_tokens = prot_tokens.to(text_embeds.dtype)

        inputs_embeds = torch.cat([prot_tokens, text_embeds], dim=1)
        full_attn = torch.cat([prot_mask, attention_mask], dim=1)
        targets = torch.cat([prot_empty_targets, targets], dim=1)

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attn,
            return_dict=True,
            labels=targets,
            output_hidden_states=True
        )

        last_hidden = outputs.hidden_states[-1]
        last_hidden = last_hidden[0] if isinstance(last_hidden, (tuple, list)) else last_hidden

        with autocast(enabled=False):
            head_param = next(self.reliability_head.parameters())
            qf = qformer_feats.to(device=head_param.device, dtype=head_param.dtype)
            pf = plm_pooled.to(device=head_param.device, dtype=head_param.dtype)
            r_input = torch.cat([qf, pf], dim=-1)
            r_logits = self.reliability_head(r_input)  # [B, NUM_CLASSES] or [B, 2]
            r_target = r_tensor.to(device=r_logits.device, dtype=r_logits.dtype).view(-1)
            if self.reliability_binary:
                r_class_target = _r_value_to_binary_class(r_target).to(r_logits.device)
            else:
                r_class_target = _r_value_to_class(r_target).to(r_logits.device)
            cw = self.reliability_class_weights.to(device=r_logits.device, dtype=r_logits.dtype)
            r_loss = nn.functional.cross_entropy(r_logits, r_class_target, weight=cw)

        if return_pred:
            with torch.no_grad():
                pred_texts, r_pred, conf, _, _prob = self.generate(
                    samples,
                    do_sample=self.do_sample,
                    num_beams=self.num_beams,
                    max_length=self.max_inference_len,
                    min_length=self.min_inference_len
                )
            return outputs.loss, r_loss, pred_texts, r_pred
        return outputs.loss, r_loss

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
        prot_tokens, prot_feats, qformer_feats, plm_pooled = self.prot_qformer(prot_embeds, prot_attn)
        device = prot_embeds.device

        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_batch.input_ids)

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

        B_gen = prot_tokens.shape[0]
        Q_gen = prot_tokens.shape[1]
        generated_token_ids = gen_out.sequences

        if generated_token_ids.shape[1] > 0 and (generated_token_ids != self.llm_tokenizer.pad_token_id).any():
            generated_embeds = self.llm_model.get_input_embeddings()(generated_token_ids)
            full_embeds = torch.cat([prot_tokens, prompt_embeds, generated_embeds], dim=1)
            gen_mask = torch.ones((B_gen, generated_token_ids.shape[1]),
                                 dtype=attention_mask.dtype, device=device)
            full_attention_mask = torch.cat([attention_mask, gen_mask], dim=1)

            with torch.no_grad():
                outputs = self.llm_model(
                    inputs_embeds=full_embeds,
                    attention_mask=full_attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                last_hidden = outputs.hidden_states[-1]

            with autocast(enabled=False):
                head_param = next(self.reliability_head.parameters())
                qf = qformer_feats.to(device=head_param.device, dtype=head_param.dtype)
                pf = plm_pooled.to(device=head_param.device, dtype=head_param.dtype)
                r_input = torch.cat([qf, pf], dim=-1)
                r_logits = self.reliability_head(r_input)  # [B, K]
                r_probs = torch.softmax(r_logits, dim=-1)  # [B, K]
                pred_idx = r_logits.argmax(dim=-1)
                if self.reliability_binary:
                    r_pred = pred_idx.to(torch.float32)  # 0.0 (neg) or 1.0 (pos)
                else:
                    r_pred = _class_to_r_value(pred_idx)
            # LLM last layer embedding: mean over sequence (protein + prompt + generated)
            llm_last_mean = last_hidden.to(torch.float32).mean(dim=1).detach().to(torch.float16).cpu()
        else:
            r_pred = torch.ones(B_gen, device=device) * (0.0 if self.reliability_binary else 0.5)
            r_probs = torch.zeros(B_gen, 2 if self.reliability_binary else RELIABILITY_NUM_CLASSES, device=device)
            # LLM last layer embedding: forward on input only (protein + prompt), then mean over sequence
            with torch.no_grad():
                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                last_hidden = outputs.hidden_states[-1]
            llm_last_mean = last_hidden.to(torch.float32).mean(dim=1).detach().to(torch.float16).cpu()

        ts = self.llm_model.compute_transition_scores(
            gen_out.sequences, gen_out.scores,
            beam_indices=getattr(gen_out, "beam_indices", None),
            normalize_logits=True
        )
        conf = torch.exp(ts.to(torch.float32).mean(dim=1)).clamp(1e-9, 1.0).tolist()

        emb_out = {
            "plm_mean_fp16": prot_embeds.to(torch.float32).mean(dim=1).detach().to(torch.float16).cpu(),
            "qformer_feats_fp16": prot_feats.detach().to(torch.float16).cpu(),
            "llm_last_fp16": llm_last_mean
        }

        return texts, r_pred, conf, emb_out, r_probs
