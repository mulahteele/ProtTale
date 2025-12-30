# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
import json
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
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
        """
        Tokenize protein sequences for both ESM2 and ESM-C encoders.
        
        Returns tensors with fixed shapes:
          - input_ids:      [B, Lfix]
          - attention_mask: [B, Lfix]
          
        Compatible Encoders:
          - ESM2 (HuggingFace): Uses EsmTokenizer with standard HF interface
          - ESM-C (official): Uses ESMC.tokenizer, returns lists, manually padded
          
        Args:
            seqs (list[str]): List of protein sequences
            
        Returns:
            dict: Tokenized output with 'input_ids' and 'attention_mask' tensors
        """
        # Try HF-style first
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
            pass  # fall back to ESM-C

        # ESM-C path: tokenizer returns python lists; do manual pad/trunc
        toks = self.prot_tokenizer(seqs)   # no return_tensors here
        ids_list = toks["input_ids"]       # List[List[int]]
        Lfix = self.prot_max_len
        pad_id = getattr(self.prot_tokenizer, "pad_token_id", 1)

        padded = []
        mask = []
        for ids in ids_list:
            ids = ids[:Lfix]  # right truncation
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

        # Optional augmentation: keep if you used it before
        if self.prot_aug == 'rand_crop':
            # If you're unsure about #special tokens, change -2 to -4 or skip cropping.
            prot_seqs1 = [rand_seq_crop(seq, self.prot_max_len - 2) for seq in prot_seqs1]
            prot_seqs2 = [rand_seq_crop(seq, self.prot_max_len - 2) for seq in prot_seqs2]

        # Text tokenization (HF tokenizer)
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

        # Protein tokenization (HF-ESM2 or ESM-C)
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
            # persistent_workers=True,
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
            # persistent_workers=True,
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
                                    #   persistent_workers=True,
                                      collate_fn=Stage1Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, self.prot_aug))
        test_match_loader = DataLoader(self.test_dataset_match, 
                                       batch_size=self.match_batch_size,
                                       shuffle=False,
                                       num_workers=self.num_workers, 
                                       pin_memory=False, 
                                       drop_last=False, 
                                    #    persistent_workers=True,
                                       collate_fn=Stage1Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, self.prot_aug))
        return val_match_loader, test_match_loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--match_batch_size', type=int, default=64)
        parser.add_argument('--root', type=str, default='data/SwissProtV3_stage1') ##change root for stage1
        parser.add_argument('--text_max_len', type=int, default=256)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        parser.add_argument('--prot_aug', type=str, default='None')
        return parent_parser
    
















class Stage1MixDM(LightningDataModule):
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
        assert args.mix_dataset
        train_dataset1 = SwissProtDataset(root+'/SwissProtV3/train_set.json')
        train_dataset2 = OntoProteinDataset(root+'/OntoProteinDatasetV2/train.txt')
        self.train_dataset = ConcatDataset([train_dataset1, train_dataset2], )
        self.swiss_val_dataset = SwissProtDataset(root+'/SwissProtV3/valid_set.json')
        self.onto_val_dataset = OntoProteinDataset(root+'/OntoProteinDatasetV2/valid.txt')
        self.swiss_test_dataset = SwissProtDataset(root+'/SwissProtV3/test_set.json')
        self.onto_test_dataset = OntoProteinDataset(root+'/OntoProteinDatasetV2/test.txt')
        
        self.swiss_val_dataset_match = SwissProtDataset(root+'/SwissProtV3/valid_set.json').shuffle()
        self.onto_val_dataset_match = OntoProteinDataset(root+'/OntoProteinDatasetV2/valid.txt').shuffle()
        self.swiss_test_dataset_match = SwissProtDataset(root+'/SwissProtV3/test_set.json').shuffle()
        self.onto_test_dataset_match = OntoProteinDataset(root+'/OntoProteinDatasetV2/test.txt').shuffle()
        
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
        loader1 = DataLoader(
            self.swiss_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=Stage1Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, self.prot_aug)
        )
        loader2 = DataLoader(
            self.onto_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=Stage1Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, self.prot_aug)
        )
        return [loader1, loader2]

    def swiss_match_dataloader(self):
        val_match_loader = DataLoader(self.swiss_val_dataset_match,
                                      batch_size=self.match_batch_size,
                                      shuffle=False,
                                      num_workers=self.num_workers, 
                                      pin_memory=False, 
                                      drop_last=True, 
                                      collate_fn=Stage1Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, self.prot_aug))
        test_match_loader = DataLoader(self.swiss_test_dataset_match, 
                                       batch_size=self.match_batch_size,
                                       shuffle=False,
                                       num_workers=self.num_workers, 
                                       pin_memory=False, 
                                       drop_last=False, 
                                       collate_fn=Stage1Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, self.prot_aug))
        return val_match_loader, test_match_loader
    
    def onto_match_dataloader(self):
        val_match_loader = DataLoader(self.onto_val_dataset_match,
                                      batch_size=self.match_batch_size,
                                      shuffle=False,
                                      num_workers=self.num_workers, 
                                      pin_memory=False, 
                                      drop_last=False, 
                                      collate_fn=Stage1Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, self.prot_aug))
        test_match_loader = DataLoader(self.onto_test_dataset_match, 
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
        parser.add_argument('--root', type=str, default='data/SwissProtV3')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        parser.add_argument('--prot_aug', type=str, default='None')
        return parent_parser




class SwissProtDataset_stage1(Dataset):
    def __init__(self, data_path, prompt='Swiss-Prot description: ', return_prompt=False):
        super(SwissProtDataset_stage1, self).__init__()
        self.data_path = data_path

        ## load data
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            self.data_list = [json.loads(line) for line in lines]
        # self.data_list = self.data_list[:100]
        ## preprocessing
        self.data_list = [(p1, p2, t1.strip() + '\n', t2.strip() + '\n',  float(l)) for p1, p2, t1, t2, l in self.data_list]###！！！

        self.text2id = {}
        # for prot_seq, text_seq in self.data_list:
        #     if text_seq not in self.text2id:
        #         self.text2id[text_seq] = len(self.text2id)
        
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












class SwissProtDataset(Dataset):
    def __init__(self, data_path, prompt='Swiss-Prot description: ', return_prompt=False, classification_dict=None):
        super(SwissProtDataset, self).__init__()
        self.data_path = data_path

        ## load data
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            self.data_list = [json.loads(line) for line in lines]

        
        ## preprocessing
        self.data_list = [(p, t.strip() + '\n', float(r), ast.literal_eval(g)) for p, t, r, g in self.data_list]###！！！

        # self.data_list = self.data_list[:100]

        self.text2id = {}
        for prot_seq, text_seq, r, go in self.data_list:
            if text_seq not in self.text2id:
                self.text2id[text_seq] = len(self.text2id)
        
        self.prompt = prompt
        self.return_prompt = return_prompt

        self.go2id = classification_dict

        if not classification_dict:
            ##collect all go terms
            self.go_all = {go for _, _, _, g in self.data_list for go in g}
            UNK_GO = "<UNK_GO>"
            go_terms = sorted(self.go_all)

            # put UNK at index 0; avoid duplication if it already appears
            vocab = [UNK_GO] + [go for go in go_terms if go != UNK_GO]
            # mappings
            self.go2id = {go: i for i, go in enumerate(vocab)}
            self.id2go = {i: go for go, i in self.go2id.items()}

            # optional: helper to encode a list of GO terms -> ids (with UNK fallback)
            def encode_go_list(go_list):
                # go_list is a list like ['GO:0003725', 'GO:0015180', ...]
                return [self.go2id.get(go, self.go2id[UNK_GO]) for go in go_list]


            print('number of the class: ', len(self.go2id))



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
        prot_seq, text_seq, r, go_list = self.data_list[index]
        go_class = [self.go2id.get(go, self.go2id["<UNK_GO>"]) for go in go_list]
        C = len(self.go2id)
        idxs = torch.tensor(go_class, dtype=torch.long)
        one_hot = torch.zeros(C, dtype=torch.float32)
        valid = (idxs >= 0) & (idxs < C)
        if valid.any():
            one_hot.index_fill_(0, idxs[valid], 1.0)

        r_tensor = torch.tensor(r, dtype=torch.float32)
        r_tensor = torch.where(r_tensor == -1.0, torch.tensor(0.0, dtype=torch.float32), r_tensor)

        if self.return_prompt:
            return prot_seq, self.prompt, text_seq, r_tensor, go_list, one_hot, index
        return prot_seq, text_seq, r_tensor, go_list, one_hot, index


class PDBAbstractDataset(Dataset):
    def __init__(self, root_path, subset, prompt='ABSTRACT: ', return_prompt=False):
        super(PDBAbstractDataset, self).__init__()
        self.data_path = Path(root_path) / subset
        self.abstract_path = Path(root_path) / 'abstract.json'
        
        ## load dataset
        with open(self.abstract_path, 'r') as f:
            abstract_data = json.load(f)
            abstract_data_dict = {line['pdb_id']: line['caption'] for line in abstract_data}
        
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
            pdb2seq = [line.strip().split('\t') for line in lines]
        
        ## process dataset
        data_list = []
        for pdb_id, seq in pdb2seq:
            abstract = abstract_data_dict[pdb_id]
            abstract = abstract.replace('\n', ' ').strip() + '\n'
            data_list.append((seq, abstract))
        self.data_list = data_list
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
        seq, abstract = self.data_list[index]
        if self.return_prompt:
            return seq, self.prompt, abstract
        return seq, abstract
    

class OntoProteinDataset(Dataset):
    def __init__(self, data_path, prompt='Gene Ontology description: ', return_prompt=False):
        super(OntoProteinDataset, self).__init__()
        self.data_path = data_path
        
        ## load data
        with open(data_path, 'r') as f:
            lines = f.readlines()
            self.data_list = [line.strip().split('\t') for line in lines]

        ## preprocessing
        ## fixme: I have disabled the signal word for this dataset. However, it was used in previous experiments.
        if True:
            self.data_list = [(p, t.strip() + '\n') for p, t in self.data_list]
        else:
            self.data_list = [(p, "KG: " + t.strip() + '\n') for p, t in self.data_list]
        self.prompt = prompt
        self.return_prompt = return_prompt

    def shuffle(self):
        random.shuffle(self.data_list)
        return self

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        prot_seq, text_seq = self.data_list[index]
        if self.return_prompt:
            return prot_seq, self.prompt, text_seq
        return prot_seq, text_seq


if __name__ == '__main__':
    import numpy as np
    ## get statistics for swiss prot dataset
    if False:
        swiss_train = SwissProtDataset('../data/SwissProtV3/train_set.json')
        swiss_valid = SwissProtDataset('../data/SwissProtV3/valid_set.json')
        swiss_test = SwissProtDataset('../data/SwissProtV3/test_set.json')
        print(len(swiss_train), len(swiss_valid), len(swiss_test))
        
        ## get amino acid statistics
        aa_lens = np.asarray([len(seq) for seq, _ in swiss_train.data_list])
        print('Train dataset mean: ', np.mean(aa_lens), 'min: ', aa_lens.min(), 'max: ', aa_lens.max())
        aa_lens = np.asarray([len(seq) for seq, _ in swiss_valid.data_list])
        print('Valid dataset mean: ', np.mean(aa_lens), 'min: ', aa_lens.min(), 'max: ', aa_lens.max())
        aa_lens = np.asarray([len(seq) for seq, _ in swiss_test.data_list])
        print('Test dataset mean: ', np.mean(aa_lens), 'min: ', aa_lens.min(), 'max: ', aa_lens.max())

        ## get text statistics
        text_lens = np.asarray([len(seq.split()) for _, seq in swiss_train.data_list])
        print('Train dataset mean: ', np.mean(text_lens), 'min: ', text_lens.min(), 'max: ', text_lens.max())
        text_lens = np.asarray([len(seq.split()) for _, seq in swiss_valid.data_list])
        print('Valid dataset mean: ', np.mean(text_lens), 'min: ', text_lens.min(), 'max: ', text_lens.max())
        text_lens = np.asarray([len(seq.split()) for _, seq in swiss_test.data_list])
        print('Test dataset mean: ', np.mean(text_lens), 'min: ', text_lens.min(), 'max: ', text_lens.max())
        print('---------------------------')

    ## get statistics for onto protein dataset
    onto_train = OntoProteinDataset('../data/OntoProteinDatasetV2/train.txt')
    onto_valid = OntoProteinDataset('../data/OntoProteinDatasetV2/valid.txt')
    onto_test = OntoProteinDataset('../data/OntoProteinDatasetV2/test.txt')
    print(len(onto_train), len(onto_valid), len(onto_test))

    ## get amino acid statistics
    aa_lens = np.asarray([len(seq) for seq, _ in onto_train.data_list])
    print('Train dataset mean: ', np.mean(aa_lens), 'min: ', aa_lens.min(), 'max: ', aa_lens.max())
    aa_lens = np.asarray([len(seq) for seq, _ in onto_valid.data_list])
    print('Valid dataset mean: ', np.mean(aa_lens), 'min: ', aa_lens.min(), 'max: ', aa_lens.max())
    aa_lens = np.asarray([len(seq) for seq, _ in onto_test.data_list])
    print('Test dataset mean: ', np.mean(aa_lens), 'min: ', aa_lens.min(), 'max: ', aa_lens.max())

    ## get text statistics
    text_lens = np.asarray([len(seq.split()) for _, seq in onto_train.data_list])
    print('Train dataset mean: ', np.mean(text_lens), 'min: ', text_lens.min(), 'max: ', text_lens.max())
    text_lens = np.asarray([len(seq.split()) for _, seq in onto_valid.data_list])
    print('Valid dataset mean: ', np.mean(text_lens), 'min: ', text_lens.min(), 'max: ', text_lens.max())
    text_lens = np.asarray([len(seq.split()) for _, seq in onto_test.data_list])
    print('Test dataset mean: ', np.mean(text_lens), 'min: ', text_lens.min(), 'max: ', text_lens.max())
    print('---------------------------')
