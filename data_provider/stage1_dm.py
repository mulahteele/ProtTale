# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
import json
import torch
from torch.utils.data import DataLoader, Dataset
import random
import ast
from pathlib import Path


def rand_seq_crop(seq, max_len):
    if len(seq) <= max_len:
        return seq
    rand_pos = random.randint(0, len(seq)-1-max_len)
    return seq[rand_pos:rand_pos+max_len]

class Stage1Collater(object):
    def __init__(self, tokenizer, prot_tokenizer, text_max_len, prot_max_len, prot_aug='None'):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.text_max_len = text_max_len
        self.prot_max_len = prot_max_len
        self.prot_aug = prot_aug

    def _tokenize_proteins(self, seqs):
        """Tokenize protein sequences for ESM2 or ESM-C. Returns dict with input_ids and attention_mask [B, Lfix]."""
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

        toks = self.prot_tokenizer(seqs)
        ids_list = toks["input_ids"]
        Lfix = self.prot_max_len
        pad_id = getattr(self.prot_tokenizer, "pad_token_id", 1)

        padded = []
        mask = []
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
        prot_seqs1, prot_seqs2, text_seqs1, text_seqs2, labels, _ = zip(*batch)

        if self.prot_aug == 'rand_crop':
            prot_seqs1 = [rand_seq_crop(seq, self.prot_max_len - 2) for seq in prot_seqs1]
            prot_seqs2 = [rand_seq_crop(seq, self.prot_max_len - 2) for seq in prot_seqs2]

        text_tokens1 = self.tokenizer(
            text_seqs1,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.text_max_len,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False
        )
        text_tokens2 = self.tokenizer(
            text_seqs2,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.text_max_len,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False
        )

        prot_tokens1 = self._tokenize_proteins(prot_seqs1)
        prot_tokens2 = self._tokenize_proteins(prot_seqs2)

        labels = torch.tensor([float(str(l).strip()) for l in labels], dtype=torch.float32)
        return prot_tokens1, prot_tokens2, text_tokens1, text_tokens2, labels

class Stage1DM(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        self.text_max_len = args.text_max_len
        self.prot_max_len = args.prot_max_len
        if root.find('SwissProt') >= 0:
            self.train_dataset = SwissProtDataset_stage1(root+'/train_set.json')
            self.val_dataset = SwissProtDataset_stage1(root+'/valid_set.json')
            self.test_dataset = SwissProtDataset_stage1(root+'/test_set.json')
            self.val_dataset_match = SwissProtDataset_stage1(root+'/valid_set.json').shuffle()
            self.test_dataset_match = SwissProtDataset_stage1(root+'/test_set.json').shuffle()    
        else:
            raise NotImplementedError

        self.tokenizer = None
        self.prot_tokenizer = None
        self.prot_aug = args.prot_aug

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
            drop_last=True,
            collate_fn=Stage1Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, self.prot_aug)
        )
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=Stage1Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, self.prot_aug)
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=Stage1Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, self.prot_aug)
        )
        return [val_loader, test_loader]

    def match_dataloader(self):
        val_match_loader = DataLoader(self.val_dataset_match, 
                                      batch_size=self.match_batch_size,
                                      shuffle=False,
                                      num_workers=self.num_workers, 
                                      pin_memory=False, 
                                      drop_last=True, 
                                      collate_fn=Stage1Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, self.prot_aug))
        test_match_loader = DataLoader(self.test_dataset_match, 
                                       batch_size=self.match_batch_size,
                                       shuffle=False,
                                       num_workers=self.num_workers, 
                                       pin_memory=False, 
                                       drop_last=False, 
                                       collate_fn=Stage1Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, self.prot_aug))
        return val_match_loader, test_match_loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--match_batch_size', type=int, default=64)
        parser.add_argument('--root', type=str, default='data/SwissProtV3_stage1')
        parser.add_argument('--text_max_len', type=int, default=256)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        parser.add_argument('--prot_aug', type=str, default='None')
        return parent_parser


class SwissProtDataset_stage1(Dataset):
    def __init__(self, data_path, prompt='Swiss-Prot description: ', return_prompt=False):
        super(SwissProtDataset_stage1, self).__init__()
        self.data_path = data_path
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            self.data_list = [json.loads(line) for line in lines]
        self.data_list = [(p1, p2, t1.strip() + '\n', t2.strip() + '\n', float(l)) for p1, p2, t1, t2, l in self.data_list]

        self.prompt = prompt
        self.return_prompt = return_prompt

    def shuffle(self):
        random.shuffle(self.data_list)
        return self

    def len(self,):
        return len(self)
    
    def get(self, idx):
        return self.__getitem__(idx)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        prot_seq1, prot_seq2, text_seq1, text_seq2, label = self.data_list[index]
        if self.return_prompt:
            return prot_seq, self.prompt, text_seq, index
        return prot_seq1, prot_seq2, text_seq1, text_seq2, label, index









