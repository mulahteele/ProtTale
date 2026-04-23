# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import ast
import json
import os
import random
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler, Subset


class SwissProtDataset(Dataset):
    """Stage2 dataset. Supports 6-column format: [prot, text, r, go_list, ec_list, cluster_id]."""
    def __init__(self, data_path, prompt='Swiss-Prot description: ', return_prompt=False):
        super(SwissProtDataset, self).__init__()
        self.data_path = data_path

        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            raw = [json.loads(line) for line in lines]

        self.data_list = []
        self.cluster_ids = []
        for row in raw:
            p, t, r, g = row[0], row[1], row[2], row[3]
            go_list = ast.literal_eval(g) if isinstance(g, str) else g
            cluster_id = row[5] if len(row) > 5 else None
            self.data_list.append((p, t.strip() + '\n', float(r), go_list))
            self.cluster_ids.append(cluster_id)

        self.text2id = {}
        for prot_seq, text_seq, r, go in self.data_list:
            if text_seq not in self.text2id:
                self.text2id[text_seq] = len(self.text2id)

        self.prompt = prompt
        self.return_prompt = return_prompt

    def shuffle(self):
        random.shuffle(self.data_list)
        return self

    def len(self):
        return len(self)

    def get(self, idx):
        return self.__getitem__(idx)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        prot_seq, text_seq, r, go_list = self.data_list[index]
        r_tensor = torch.tensor(r, dtype=torch.float32)
        r_tensor = torch.where(r_tensor == -1.0, torch.tensor(0.0, dtype=torch.float32), r_tensor)

        if self.return_prompt:
            return prot_seq, self.prompt, text_seq, r_tensor, go_list, index
        return prot_seq, text_seq, r_tensor, go_list, index

    def get_indices_with_at_least_k_go(self, k=2, sample_size=2000, seed=42):
        """Indices of samples with >= k GO terms, then random sample up to sample_size. If sample_size<=0, return all such indices."""
        rng = random.Random(seed)
        indices = [i for i, (_, _, _, go_list) in enumerate(self.data_list) if len(go_list) >= k]
        if sample_size <= 0 or len(indices) <= sample_size:
            return indices
        return rng.sample(indices, sample_size)


class ReliabilityFinetuneDataset(Dataset):
    """Dataset from a JSON-lines file with r values for 4-class classification (-1, 0, 0.5, 1).
    Same row format as train_set."""
    def __init__(self, data_path, prompt='Swiss-Prot description: ', return_prompt=True):
        super().__init__()
        self.data_path = data_path
        with open(data_path, 'r', encoding='utf-8') as f:
            raw = [json.loads(line.strip()) for line in f if line.strip()]
        self.data_list = []
        for row in raw:
            p, t, r, g = row[0], row[1], row[2], row[3]
            go_list = ast.literal_eval(g) if isinstance(g, str) else g
            r_val = float(r)
            self.data_list.append((p, t.strip() + '\n', r_val, go_list))

        self.prompt = prompt
        self.return_prompt = return_prompt

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        prot_seq, text_seq, r, go_list = self.data_list[index]
        r_tensor = torch.tensor(r, dtype=torch.float32)
        if self.return_prompt:
            return prot_seq, self.prompt, text_seq, r_tensor, go_list, index
        return prot_seq, text_seq, r_tensor, go_list, index


class ClusterLimitSampler(Sampler):
    """Each epoch samples at most max_per_cluster indices per cluster (random if cluster has more). When seed is set, sampling is reproducible across runs (epoch 0 uses seed+0, epoch 1 uses seed+1, ...).
    Effect: the **total set of training samples** (which indices are used) can differ every epoch; within an epoch that set is fixed and split into mini-batches. So it is epoch-level sampling, not mini-batch-level."""
    def __init__(self, dataset: SwissProtDataset, max_per_cluster: int = 5, seed=None):
        self.dataset = dataset
        self.max_per_cluster = max_per_cluster
        self.seed = seed
        self._call_count = 0
        self.has_cluster = getattr(dataset, 'cluster_ids', None) is not None and all(
            c is not None for c in dataset.cluster_ids
        )

    def __iter__(self):
        if self.seed is not None:
            epoch_seed = self.seed + self._call_count
            self._call_count += 1
            rng = random.Random(epoch_seed)
            gen = torch.Generator().manual_seed(epoch_seed)
        else:
            rng = random.Random()
            gen = None
        if not self.has_cluster:
            perm = torch.randperm(len(self.dataset), generator=gen).tolist() if gen is not None else torch.randperm(len(self.dataset)).tolist()
            return iter(perm)
        cluster_to_indices = {}
        for idx, c in enumerate(self.dataset.cluster_ids):
            cluster_to_indices.setdefault(c, []).append(idx)
        out = []
        for c, indices in cluster_to_indices.items():
            k = min(self.max_per_cluster, len(indices))
            out.extend(rng.sample(indices, k))
        rng.shuffle(out)
        return iter(out)

    def __len__(self):
        if not self.has_cluster:
            return len(self.dataset)
        cluster_to_indices = {}
        for idx, c in enumerate(self.dataset.cluster_ids):
            cluster_to_indices.setdefault(c, []).append(idx)
        return sum(min(self.max_per_cluster, len(indices)) for indices in cluster_to_indices.values())







class Stage2Collater(object):
    def __init__(self, tokenizer, prot_tokenizer, text_max_len, prot_max_len):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.text_max_len = text_max_len
        self.prot_max_len = prot_max_len
        
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
        prot_seqs, prompt_seqs, text_seqs, r_tensor, go, indices = zip(*batch)

        prot_tokens = self._tokenize_proteins(prot_seqs)
        indices = torch.tensor(indices, dtype=torch.long)
        r_tensor = torch.tensor(r_tensor, dtype=torch.float32)

        original_side = getattr(self.tokenizer, "padding_side", "right")
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
            self.tokenizer.padding_side = original_side

        return prot_tokens, input_tokens, prompt_tokens, r_tensor, indices


class InferenceCollater(object):
    def __init__(self, tokenizer, prot_tokenizer, text_max_len, prot_max_len):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.text_max_len = text_max_len
        self.prot_max_len = prot_max_len

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
        prot_seqs, prompt_seqs, text_seqs, r_tensor, go, indices = zip(*batch)

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
        r_tensor = torch.tensor(r_tensor, dtype=torch.float32)
        target_dict = {'indices': indices, 'prot_seqs': prot_seqs, 'targets': text_seqs}
        return prot_tokens, prompt_tokens, r_tensor, target_dict


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

        train_path = getattr(args, 'train_set_path', '') or os.path.join(root, 'train_set.json')
        valid_path = getattr(args, 'valid_set_path', '') or os.path.join(root, 'valid_set.json')
        test_path = getattr(args, 'test_set_path', '') or os.path.join(root, 'test_set.json')
        self.train_set_path = train_path
        self.valid_set_path = valid_path
        self.test_set_path = test_path
        self.train_dataset = SwissProtDataset(train_path, prompt='Swiss-Prot description: ', return_prompt=True)
        self.val_dataset = SwissProtDataset(valid_path, prompt='Swiss-Prot description: ', return_prompt=True)
        self.test_dataset = SwissProtDataset(test_path, prompt='Swiss-Prot description: ', return_prompt=True)

        self.tokenizer = None
        self.prot_tokenizer = None
        self.max_per_cluster = getattr(args, 'max_per_cluster', 5)
        self.seed = getattr(args, 'seed', None)
        self.reliability_finetune_data = getattr(args, 'reliability_finetune_data', '')
        self.reliability_finetune_valid_data = getattr(args, 'reliability_finetune_valid_data', '')
        self.train_reliability_head_only = getattr(args, 'train_reliability_head_only', False)

    def init_tokenizer(self, tokenizer, prot_tokenizer):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer

    def get_inference_training_indices(self, min_go=2, sample_size=2000, seed=42):
        """Indices for inference-on-training: train samples with >= min_go GO terms, random sample up to sample_size."""
        return self.train_dataset.get_indices_with_at_least_k_go(k=min_go, sample_size=sample_size, seed=seed)

    def get_inference_training_dataloader(self, min_go=2, sample_size=2000, seed=42, batch_size=None):
        """DataLoader over subset of train (>= min_go GO, up to sample_size) for running inference."""
        indices = self.get_inference_training_indices(min_go=min_go, sample_size=sample_size, seed=seed)
        subset = Subset(self.train_dataset, indices)
        bs = batch_size if batch_size is not None else self.inference_batch_size
        return DataLoader(
            subset,
            batch_size=bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=InferenceCollater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
        )

    def get_validation_inference_dataloader(self, batch_size=None):
        """DataLoader over full validation set for inference (same format as inference training)."""
        bs = batch_size if batch_size is not None else self.inference_batch_size
        return DataLoader(
            self.val_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=InferenceCollater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
        )

    def train_dataloader(self):
        if self.train_reliability_head_only and self.reliability_finetune_data and os.path.isfile(self.reliability_finetune_data):
            train_dataset = ReliabilityFinetuneDataset(
                self.reliability_finetune_data,
                prompt='Swiss-Prot description: ',
                return_prompt=True,
            )
            return DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False,
                persistent_workers=False,
                collate_fn=Stage2Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
            )
        # Normal training: each epoch sample up to max_per_cluster indices per cluster_id (then shuffle); if no cluster_id, full shuffle. seed for reproducibility.
        sampler = ClusterLimitSampler(self.train_dataset, self.max_per_cluster, seed=self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=Stage2Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
        )

    def val_dataloader(self):
        # Validation and testing: no cluster sampling, use full datasets.
        # Step 4 (reliability head only): use updated validation set (Wang r) when provided.
        if self.train_reliability_head_only and self.reliability_finetune_valid_data and os.path.isfile(self.reliability_finetune_valid_data):
            val_dataset = ReliabilityFinetuneDataset(
                self.reliability_finetune_valid_data,
                prompt='Swiss-Prot description: ',
                return_prompt=True,
            )
        else:
            val_dataset = self.val_dataset
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=Stage2Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=InferenceCollater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
        )
        if self.train_reliability_head_only and self.reliability_finetune_data and os.path.isfile(self.reliability_finetune_data):
            train_dataset_full = ReliabilityFinetuneDataset(
                self.reliability_finetune_data,
                prompt='Swiss-Prot description: ',
                return_prompt=True,
            )
            train_val_loader = DataLoader(
                train_dataset_full,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False,
                persistent_workers=False,
                collate_fn=Stage2Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
            )
            return [val_loader, test_loader, train_val_loader]
        return [val_loader, test_loader]
    

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str, default='data/SwissProtV3')
        parser.add_argument('--text_max_len', type=int, default=256)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        parser.add_argument('--max_per_cluster', type=int, default=5, help='Max samples per cluster per epoch for train (stage2 with cluster column)')
        parser.add_argument('--prompt', type=str, default='The protein has the following properties: ')
        parser.add_argument('--filter_side_qa', action='store_true', default=False)
        return parent_parser


    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str, default='data/SwissProtV3', help='Base data dir; used when train/valid/test_set_path are not set')
        parser.add_argument('--train_set_path', type=str, default='', help='Stage2 train JSON-lines; default root/train_set.json')
        parser.add_argument('--valid_set_path', type=str, default='', help='Stage2 valid JSON-lines; default root/valid_set.json')
        parser.add_argument('--test_set_path', type=str, default='', help='Stage2 test JSON-lines; default root/test_set.json')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        parser.add_argument('--max_per_cluster', type=int, default=5, help='Max samples per cluster per epoch for train (stage2 with cluster column)')
        parser.add_argument('--reliability_num_bins', type=int, default=10, help='Number of bins for inverse frequency weighting in reliability head training')
        return parent_parser


