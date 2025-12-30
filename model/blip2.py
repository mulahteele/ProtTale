"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
import torch.nn as nn

from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from transformers import BertTokenizer, BitsAndBytesConfig
from transformers import EsmTokenizer, EsmModel
try:
    from esm.models.esmc import ESMC
except Exception:
    # some older installs expose it under esm.models.esmc.esmc
    try:
        from esm.models import esmc as _esmc_mod
        ESMC = _esmc_mod.ESMC
    except Exception as e:
        raise ImportError(
            "Cannot import ESMC. Make sure `pip install esm` succeeded "
            "and esm>=2.x is installed. Original error: %r" % e
        )

from esm.sdk.api import LogitsConfig

def get_gpu_memory(device=0):
    # t = torch.cuda.get_device_properties(device).total_memory
    # r = torch.cuda.memory_reserved(device)
    # a = torch.cuda.memory_allocated(device)
    # f = r-a  # free inside reserved
    free, total = torch.cuda.mem_get_info(device)
    free = free / (1024 ** 3)
    total = total / (1024 ** 3)
    return free, total-free, total


class Blip2Base(BaseModel):
    # @classmethod
    # def init_tokenizer(cls):
    #     tokenizer = BertTokenizer.from_pretrained('./bert_pretrained/')
    #     tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    #     return tokenizer

    @classmethod
    def init_Qformer(cls, model_name, num_query_token, plm_width, cross_attention_freq=2):
        assert model_name == 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        print("bert load microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        
        encoder_config = BertConfig.from_pretrained(model_name)
        encoder_config.encoder_width = plm_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        
        Qformer = BertLMHeadModel.from_pretrained(model_name, config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        tokenizer = BertTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer, Qformer, query_tokens
    

    def init_protein_encoder(self, plm_name, load_4bit=False, device="cuda"):
        """
        Create a protein encoder + tokenizer + LayerNorm.
        
        Supported Encoders:
          1. ESM2 (HuggingFace transformers): plm_name starts with 'facebook/esm2'
             - Uses EsmTokenizer and EsmModel from transformers
             - Examples: 'facebook/esm2_t30_150M_UR50D', 'facebook/esm2_t33_650M_UR50D'
             
          2. ESM-C (official ESM package): plm_name starts with 'esmc_'
             - Uses ESMC from esm.models.esmc
             - Examples: 'esmc_300m', 'esmc_600m'
        
        Args:
            plm_name (str): Model name/identifier
            load_4bit (bool): Whether to use 4-bit quantization (ESM2 only)
            device (str): Target device
            
        Returns:
            tuple: (plm_tokenizer, plm_module, ln_layer)
                - plm_tokenizer: Tokenizer function or object
                - plm_module: The encoder model
                - ln_layer: LayerNorm layer for encoder output
        """
        # ---------- Case A: ESM-2 (HF transformers) ----------
        if str(plm_name).startswith("facebook/esm2"):
            plm_tokenizer = EsmTokenizer.from_pretrained(plm_name)

            if not load_4bit:
                plm = EsmModel.from_pretrained(
                    plm_name,
                    add_pooling_layer=False,
                    torch_dtype=torch.bfloat16,
                ).to(device)
            else:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    load_in_8bit=False,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                # Automatic device selection for 4-bit quantization
                # Use CUDA_VISIBLE_DEVICES or default to first available device
                import os
                visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
                device_id = int(visible_devices.split(',')[0])
                device_map = {"": device_id}

                plm = EsmModel.from_pretrained(
                    plm_name,
                    add_pooling_layer=False,
                    quantization_config=quant_config,
                    load_in_4bit=True,
                    load_in_8bit=False,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16,
                )

            plm.num_features = plm.config.hidden_size
            ln_layer = nn.LayerNorm(plm.num_features)
            return plm_tokenizer, plm, ln_layer

        # ---------- Case B: ESM-C (official esm package) ----------
        elif str(plm_name).startswith("esmc_"):
            esmc = ESMC.from_pretrained(plm_name).to(device)
            esmc.eval()

            # tokenizer shim: return python lists (no tensors here)
            def esmc_tokenizer(batch_seqs, *args, **kwargs):
                """
                ESM-C tokenizer returns python lists; we intentionally avoid tensors here.
                Collate function will handle truncation/padding/tensorization.
                """
                if isinstance(batch_seqs, str):
                    batch_seqs = [batch_seqs]
                toks = esmc.tokenizer(batch_seqs)  # no return_tensors
                # unify to HF-like mapping keys
                return {"input_ids": toks["input_ids"]}

            class ESMCWrapper(nn.Module):
                """Expose HF-like forward that returns .last_hidden_state [B, L, D]."""
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    # probe hidden size (fallback if missing)
                    dim = getattr(getattr(model, "config", None), "hidden_size", None)
                    if dim is None:
                        with torch.no_grad():
                            probe = model.tokenizer(["M"])["input_ids"]
                            # make tensor on device for probing embeddings shape
                            probe_ids = torch.tensor(probe, dtype=torch.long, device=device)
                            if probe_ids.dim() == 1:
                                probe_ids = probe_ids.unsqueeze(0)
                            emb = self._forward_embeddings(probe_ids)
                            dim = emb.shape[-1]
                    self.num_features = dim

                def _forward_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
                    """
                    Use logits(..., return_embeddings=True) if available; otherwise fallback to __call__.
                    Returns per-residue embeddings [B, L, D].
                    """
                    try:
                        out = self.model.logits(
                            input_ids,
                            LogitsConfig(sequence=True, return_embeddings=True),
                        )
                        emb = out.embeddings
                    except Exception:
                        out = self.model(input_ids)
                        emb = out.embeddings
                    if emb.dim() == 2:
                        emb = emb.unsqueeze(0)
                    return emb

                def forward(self, input_ids, attention_mask=None, **kwargs):
                    emb = self._forward_embeddings(input_ids)  # [B, L, D]
                    class _Out:
                        pass
                    ret = _Out()
                    ret.last_hidden_state = emb
                    return ret

            plm = ESMCWrapper(esmc).to(device)
            ln_layer = nn.LayerNorm(plm.num_features)
            return esmc_tokenizer, plm, ln_layer

        else:
            raise ValueError(f"Unknown PLM name: {plm_name}")


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""

#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)

