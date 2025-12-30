# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from data_provider.stage1_dm import SwissProtDataset, OntoProteinDataset


class Stage2Collater(object):
    def __init__(self, tokenizer, prot_tokenizer, text_max_len, prot_max_len, head):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.text_max_len = text_max_len
        self.prot_max_len = prot_max_len
        self.head = head
        
    def _tokenize_proteins(self, seqs):
        """
        Tokenize protein sequences for both ESM2 and ESM-C encoders.
        
        Compatible Encoders:
          - ESM2 (HuggingFace): Uses EsmTokenizer with standard HF interface
          - ESM-C (official): Uses ESMC.tokenizer, returns lists, manually padded
          
        Returns:
            dict: Tokenized output with 'input_ids' and 'attention_mask' tensors
        """
        try:
            toks = self.prot_tokenizer(
                seqs,
                truncation=True,
                padding='max_length',
                max_length=self.prot_max_len,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
            )
            return toks
        except Exception:
            pass
        toks = self.prot_tokenizer(seqs)   # ESM-C path: returns lists
        ids_list = toks["input_ids"]
        Lfix = self.prot_max_len
        pad_id = getattr(self.prot_tokenizer, "pad_token_id", 1)
        padded, mask = [], []
        for ids in ids_list:
            ids = ids[:Lfix]
            need = Lfix - len(ids)
            if need > 0:
                ids = ids + [pad_id] * need
            padded.append(ids)
            mask.append([1] * (Lfix - need) + [0] * need)
        input_ids = torch.tensor(padded, dtype=torch.long)
        attention_mask = torch.tensor(mask, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __call__(self, batch):
        prot_seqs, prompt_seqs, text_seqs, r_tensor, go, go_class, indices = zip(*batch)

        prot_tokens = self._tokenize_proteins(prot_seqs)

        go_class = torch.stack(go_class, dim=0).to(torch.float32)
        indices = torch.tensor(indices, dtype=torch.long)

        r_tensor = torch.tensor(r_tensor, dtype=torch.float32)
        
        # text side
        original_side = getattr(self.tokenizer, "padding_side", "right")  # save
        try:
            self.tokenizer.padding_side = 'left'
            prompt_tokens = self.tokenizer(
                prompt_seqs,
                truncation=True,
                padding='longest',
                add_special_tokens=True,
                max_length=self.text_max_len,
                return_tensors='pt',
                return_attention_mask=True,
                return_token_type_ids=False
            )
            # Calculate max length after concatenation (ensuring not exceeding model limit)
            max_prompt_len = int(prompt_tokens.attention_mask.sum(dim=1).max())
            model_max = getattr(self.tokenizer, "model_max_length", 4096)
            pair_max_len = min(self.text_max_len + max_prompt_len, model_max)

            input_pair = [[p, t] for p, t in zip(prompt_seqs, text_seqs)]
            input_tokens = self.tokenizer(
                input_pair,
                truncation=True,
                padding='max_length',
                add_special_tokens=True,
                max_length=pair_max_len,
                return_tensors='pt',
                return_attention_mask=True,
                return_token_type_ids=True  # Set to False if your text backbone doesn't need token_type_ids
            )
        finally:
            self.tokenizer.padding_side = original_side  # restore

        if self.head == 'generation':
            return prot_tokens, input_tokens, prompt_tokens, r_tensor, indices
        else:
            return prot_tokens, go_class

# prot_tokens, prompt_tokens, r_tensor, target_dict

class InferenceCollater(object):
    def __init__(self, tokenizer, prot_tokenizer, text_max_len, prot_max_len, head):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.text_max_len = text_max_len
        self.prot_max_len = prot_max_len
        self.head = head

    def _tokenize_proteins(self, seqs):
        """
        Tokenize protein sequences for both ESM2 and ESM-C encoders.
        
        Compatible Encoders:
          - ESM2 (HuggingFace): Uses EsmTokenizer with standard HF interface
          - ESM-C (official): Uses ESMC.tokenizer, returns lists, manually padded
          
        Returns:
            dict: Tokenized output with 'input_ids' and 'attention_mask' tensors
        """
        try:
            toks = self.prot_tokenizer(
                seqs,
                truncation=True,
                padding='max_length',
                max_length=self.prot_max_len,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
            )
            return toks
        except Exception:
            pass
        toks = self.prot_tokenizer(seqs)   # ESM-C lists
        ids_list = toks["input_ids"]
        Lfix = self.prot_max_len
        pad_id = getattr(self.prot_tokenizer, "pad_token_id", 1)
        padded, mask = [], []
        for ids in ids_list:
            ids = ids[:Lfix]
            need = Lfix - len(ids)
            if need > 0:
                ids = ids + [pad_id] * need
            padded.append(ids)
            mask.append([1] * (Lfix - need) + [0] * need)
        input_ids = torch.tensor(padded, dtype=torch.long)
        attention_mask = torch.tensor(mask, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __call__(self, batch):
        prot_seqs, prompt_seqs, text_seqs, r_tensor, go, go_class, indices = zip(*batch)

        original_side = getattr(self.tokenizer, "padding_side", "right")
        try:
            self.tokenizer.padding_side = 'right'
            prompt_tokens = self.tokenizer(
                prompt_seqs,
                truncation=True,
                padding='longest',
                add_special_tokens=True,
                max_length=self.text_max_len,
                return_tensors='pt',
                return_attention_mask=True,
                return_token_type_ids=False
            )
        finally:
            self.tokenizer.padding_side = original_side

        prot_tokens = self._tokenize_proteins(prot_seqs)
        go_class = torch.stack(go_class, dim=0).to(torch.float32)

        r_tensor = torch.tensor(r_tensor, dtype=torch.float32)
        
        if self.head == 'generation':
            target_dict = {'indices': indices, 'prot_seqs': prot_seqs, 'targets': text_seqs}
            return prot_tokens, prompt_tokens, r_tensor, target_dict
        else:
            target_dict = {'prot_seqs': prot_seqs, 'targets': text_seqs, 'indices': indices, 'Ground Truth GO': go}
            return prot_tokens, go_class, target_dict


class Stage2DM(LightningDataModule):
    def __init__(
        self,
        root: str = 'data/',
        args=None,
    ):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = args.num_workers
        self.text_max_len = args.text_max_len
        self.prot_max_len = args.prot_max_len
        self.head = args.head
        self.go2id = None
        # self.prompt = args.prompt
        
        if root.find('SwissProtV3') >= 0:
            self.train_dataset = SwissProtDataset(root+'/train_set.json', prompt='Swiss-Prot description: ', return_prompt=True, classification_dict = self.go2id)
            self.go2id = self.train_dataset.go2id

            self.val_dataset = SwissProtDataset(root+'/valid_set.json', prompt='Swiss-Prot description: ', return_prompt=True, classification_dict = self.go2id)
            self.test_dataset = SwissProtDataset(root+'/test_set.json', prompt='Swiss-Prot description: ', return_prompt=True, classification_dict = self.go2id)
        else:
            raise NotImplementedError

        self.tokenizer = None
        self.prot_tokenizer = None
    
    def init_tokenizer(self, tokenizer, prot_tokenizer):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=Stage2Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, self.head),
        )
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=Stage2Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, self.head),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=InferenceCollater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, self.head),
        )
        return [val_loader, test_loader]
    

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str, default='data/SwissProtV3')
        parser.add_argument('--text_max_len', type=int, default=256)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        parser.add_argument('--prompt', type=str, default='The protein has the following properties: ')
        parser.add_argument('--filter_side_qa', action='store_true', default=False)
        return parent_parser


    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str, default='data/SwissProtV3')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        # parser.add_argument('--prompt', type=str, default='The protein has the following properties: ')
        return parent_parser


