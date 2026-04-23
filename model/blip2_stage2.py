import os
import torch
from tqdm import tqdm
from model.blip2_opt import Blip2OPT
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
import ast
import pickle
from evals.tools.InfoAccretion import compute_InfoAccretion_distance
from evals.tools.wang_similarity import compute_wang_similarity
from evals.tools.jaccard_similarity import compute_jaccard_similarity
from evals.tools.extraction import process_texts_with_api
import numpy as np
import torch.distributed as dist
from typing import Any, Dict
from model.help_funcs import (
    caption_evaluate,
    AttrDict,
    _mean_conf,
    _json_default,
    load_or_process,
    load_mf_go_ids_from_tsv,
    filter_go_terms_by_set,
    build_joint_nonempty_mask,
    filter_parallel_by_mask,
)


def _batch_to_device(x, device):
    """Move batch (dict/tensor/list) to device; skip non-tensor values (e.g. lists of ints from ESM-C)."""
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _batch_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_batch_to_device(v, device) if torch.is_tensor(v) else v for v in x)
    return x


class Blip2Stage2(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass

    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        self.caption_eval_epoch = args.caption_eval_epoch
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_inference_len = args.max_inference_len
        self.min_inference_len = args.min_inference_len
        self.llm_tune = args.llm_tune
        # GO term extraction on test set (dataloader_idx 1)
        self.report_go_wang_on_test = getattr(args, 'report_go_wang_on_test', False)
        self.ia_path = getattr(args, 'ia_path', 'evals/tools/IA.txt')
        self.test_set_path = getattr(args, 'test_set_path', '') or os.path.join(getattr(args, 'root', 'data/SwissProtV3'), 'test_set.json')
        self.valid_set_path = getattr(args, 'valid_set_path', '') or os.path.join(getattr(args, 'root', 'data/SwissProtV3'), 'valid_set.json')
        self.go_files_tsv_path = getattr(args, 'go_files_tsv_path', 'evals/tools/go_files.tsv')

        # On last epoch: extract GO from val predictions and compute Wang vs valid_set.json (default off)
        self.report_go_wang_on_val = getattr(args, 'report_go_wang_on_val', False)

        # Prediction collection and saving parameters
        self.save_predictions = getattr(args, 'save_predictions', False)

        self.inference_on_training_data = getattr(args, 'inference_on_training_data', False)
        self.train_reliability_head_only = getattr(args, 'train_reliability_head_only', False)

        # Validate encoder_type and plm_model consistency
        encoder_type = getattr(args, 'encoder_type', 'auto')
        if encoder_type != 'auto':
            if encoder_type == 'esm2' and not args.plm_model.startswith('facebook/esm2'):
                raise ValueError(f"encoder_type='{encoder_type}' but plm_model='{args.plm_model}' does not start with 'facebook/esm2'")
            elif encoder_type == 'esmc' and not args.plm_model.startswith('esmc_'):
                raise ValueError(f"encoder_type='{encoder_type}' but plm_model='{args.plm_model}' does not start with 'esmc_'")
        if args.llm_name.find('galactica') >= 0:
            self.blip2 = Blip2OPT(args.bert_name,
                                  args.num_query_token, 
                                  args.cross_attention_freq, 
                                  args.plm_model,
                                  args.plm_tune,
                                  args.llm_name,
                                  args.llm_tune, 
                                  args.peft_dir,  
                                  args)
        else:
            raise NotImplementedError()
        # Default training: freeze ESM (base + LoRA), reliability_head, ln_layer, Qformer; train LLM (decoder) only
        if not self.inference_on_training_data and not self.train_reliability_head_only:
            for name, param in self.blip2.named_parameters():
                if 'reliability_head' in name or 'plm' in name:
                    param.requires_grad = False
        self.save_hyperparameters(args)

    def load_from_stage1_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['state_dict']
        state_dict = {k.split('blip2qformer.')[1]:v for k, v in state_dict.items()}
        self.blip2.load_state_dict(state_dict, strict=False)
        return self

    def freeze_for_reliability_finetune(self):
        """Freeze all parameters except reliability_head. Call after loading checkpoint for train_reliability_head_only."""
        for name, param in self.named_parameters():
            param.requires_grad = 'reliability_head' in name

    @torch.no_grad()
    def run_inference_on_training_subset(
        self,
        dm,
        output_path,
        min_go=2,
        sample_size=2000,
        seed=42,
        reliability_label_zero=False,
    ):
        """
        Run inference on training subset (>= min_go GO terms, up to sample_size).
        If reliability_label_zero: write r=0 for all rows (no GO extraction).
        Else: extract GO from predictions, compute Wang similarity, replace r by that score.
        Save rows to output_path.
        """
        dataloader = dm.get_inference_training_dataloader(min_go=min_go, sample_size=sample_size, seed=seed)
        self.eval()
        device = next(self.parameters()).device
        if device.type == 'cpu' and torch.cuda.is_available():
            self.to('cuda')
            device = next(self.parameters()).device

        idx_to_pred = []
        n_batches = len(dataloader)
        print(f"Inference on training subset: {n_batches} batches (sample_size={sample_size})", flush=True)
        for batch in tqdm(dataloader, total=n_batches, desc="train_inference", unit="batch"):
            prot_tokens, prompt_tokens, r_tensor, target_dict = batch
            prot_tokens = _batch_to_device(prot_tokens, device)
            if hasattr(prompt_tokens, 'to'):
                prompt_tokens = prompt_tokens.to(device)
            else:
                prompt_tokens = type(prompt_tokens)({k: v.to(device) for k, v in prompt_tokens.items()})
            r_tensor = r_tensor.to(device)
            samples = {'prot_batch': prot_tokens, 'prompt_batch': prompt_tokens, 'reliability': r_tensor}
            pred_texts, _, _, _, _ = self.blip2.generate(
                samples,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_inference_len,
                min_length=self.min_inference_len,
            )
            indices = target_dict['indices']
            if hasattr(indices, 'tolist'):
                indices = indices.tolist()
            for i, idx in enumerate(indices):
                idx_to_pred.append((idx, pred_texts[i]))

        idx_to_pred.sort(key=lambda x: x[0])
        sorted_indices = [x[0] for x in idx_to_pred]
        pred_texts_ordered = [x[1] for x in idx_to_pred]

        with open(dm.train_dataset.data_path, 'r', encoding='utf-8') as f:
            train_lines = [line.strip() for line in f if line.strip()]
        train_rows = [json.loads(line) for line in train_lines]

        if reliability_label_zero:
            per_scores = [0.0] * len(sorted_indices)
        else:
            gt_go_list = []
            for idx in sorted_indices:
                row = train_rows[idx]
                g = row[3]
                go_list = ast.literal_eval(g) if isinstance(g, str) else g
                gt_go_list.append(go_list)
            print(f"Extracting GO terms from {len(pred_texts_ordered)} predictions (API calls)...", flush=True)
            predicted_go_terms = process_texts_with_api(pred_texts_ordered)
            _, per_scores = compute_wang_similarity(gt_go_list, predicted_go_terms)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, idx in enumerate(sorted_indices):
                row = list(train_rows[idx])
                row[1] = pred_texts_ordered[i]  # predicted text
                row[2] = per_scores[i]
                f.write(json.dumps(row, ensure_ascii=True) + '\n')
        r_desc = "r=0" if reliability_label_zero else "r (wang score)"
        print(f"Saved {len(sorted_indices)} rows with {r_desc} to {output_path}", flush=True)
        return output_path

    @torch.no_grad()
    def run_inference_on_validation_set(self, dm, output_path, reliability_label_zero=False):
        """
        Run inference on full validation set.
        If reliability_label_zero: write r=0 for all rows (no GO extraction).
        Else: extract GO from predictions, compute Wang similarity, replace r by that score.
        Save rows to output_path (same format as train).
        """
        dataloader = dm.get_validation_inference_dataloader()
        valid_path = getattr(dm, 'valid_set_path', None)
        if not valid_path or not os.path.exists(valid_path):
            raise FileNotFoundError(f"Validation set path not found: {valid_path}")
        self.eval()
        device = next(self.parameters()).device
        if device.type == 'cpu' and torch.cuda.is_available():
            self.to('cuda')
            device = next(self.parameters()).device

        idx_to_pred = []
        n_batches = len(dataloader)
        print(f"Inference on validation set: {n_batches} batches", flush=True)
        for batch in tqdm(dataloader, total=n_batches, desc="val_inference", unit="batch"):
            prot_tokens, prompt_tokens, r_tensor, target_dict = batch
            prot_tokens = _batch_to_device(prot_tokens, device)
            if hasattr(prompt_tokens, 'to'):
                prompt_tokens = prompt_tokens.to(device)
            else:
                prompt_tokens = type(prompt_tokens)({k: v.to(device) for k, v in prompt_tokens.items()})
            r_tensor = r_tensor.to(device)
            samples = {'prot_batch': prot_tokens, 'prompt_batch': prompt_tokens, 'reliability': r_tensor}
            pred_texts, _, _, _, _ = self.blip2.generate(
                samples,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_inference_len,
                min_length=self.min_inference_len,
            )
            indices = target_dict['indices']
            if hasattr(indices, 'tolist'):
                indices = indices.tolist()
            for i, idx in enumerate(indices):
                idx_to_pred.append((idx, pred_texts[i]))

        idx_to_pred.sort(key=lambda x: x[0])
        sorted_indices = [x[0] for x in idx_to_pred]
        pred_texts_ordered = [x[1] for x in idx_to_pred]

        with open(valid_path, 'r', encoding='utf-8') as f:
            valid_lines = [line.strip() for line in f if line.strip()]
        valid_rows = [json.loads(line) for line in valid_lines]

        if reliability_label_zero:
            per_scores = [0.0] * len(sorted_indices)
        else:
            gt_go_list = []
            for idx in sorted_indices:
                row = valid_rows[idx]
                g = row[3]
                go_list = ast.literal_eval(g) if isinstance(g, str) else g
                gt_go_list.append(go_list)
            print(f"Extracting GO terms from {len(pred_texts_ordered)} predictions (API calls)...", flush=True)
            predicted_go_terms = process_texts_with_api(pred_texts_ordered)
            _, per_scores = compute_wang_similarity(gt_go_list, predicted_go_terms)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, idx in enumerate(sorted_indices):
                row = list(valid_rows[idx])
                row[1] = pred_texts_ordered[i]  # predicted text
                row[2] = per_scores[i]
                f.write(json.dumps(row, ensure_ascii=True) + '\n')
        r_desc = "r=0" if reliability_label_zero else "r (wang score)"
        print(f"Saved {len(sorted_indices)} validation rows with {r_desc} to {output_path}", flush=True)
        return output_path

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)

        if self.train_reliability_head_only:
            reliability_params = [p for n, p in self.named_parameters() if 'reliability_head' in n and p.requires_grad]
            reliability_lr = self.args.reliability_lr if self.args.reliability_lr is not None else self.args.init_lr
            optimizer = optim.AdamW(reliability_params, lr=reliability_lr, weight_decay=self.args.weight_decay)
        else:
            main_params = []
            reliability_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if 'reliability_head' in name:
                    reliability_params.append(param)
                else:
                    main_params.append(param)
            reliability_lr = self.args.reliability_lr if self.args.reliability_lr is not None else self.args.init_lr
            optimizer = optim.AdamW([
                {'params': main_params, 'lr': self.args.init_lr, 'weight_decay': self.args.weight_decay},
                {'params': reliability_params, 'lr': reliability_lr, 'weight_decay': self.args.weight_decay}
            ])
        
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def save_predictions(self, predictions, targets, q_types=None, log_prefix=''):
        assert len(predictions) == len(targets)
        if log_prefix:
            name = f'{log_prefix}_predictions.txt'
        else:
            name = 'predictions.txt'
        with open(os.path.join(self.logger.log_dir, name), 'w', encoding='utf8') as f:
            if q_types is not None:
                for p, t, q in zip(predictions, targets, q_types):
                    line = {'prediction': p, 'target': t, 'q_type': q}
                    f.write(json.dumps(line, ensure_ascii=True) + '\n')
            else:
                for p, t in zip(predictions, targets):
                    line = {'prediction': p, 'target': t}
                    f.write(json.dumps(line, ensure_ascii=True) + '\n')

    def on_validation_epoch_start(self) -> None:
        self.saved_dict_list = []
        self.prediction_list0 = []
        self.target_list0 = []
        self.prediction_list1 = []
        self.target_list1 = []
        self._saved_even_list = []
        self.val_saved_list_for_go = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if (dataloader_idx % 2) == 0:
            text_batch = batch[1]
            batch_size = text_batch.input_ids.shape[0]
            blip_batch = batch[:4]
            loss, r_loss, pred_texts, r_pred = self.blip2(blip_batch, return_pred=True)
            idx_list = batch[4].detach().cpu().tolist()
            if not isinstance(idx_list, list):
                idx_list = [idx_list]
            saved_dict = {'indices': idx_list, 'predictions': pred_texts}
            if self.train_reliability_head_only:
                r_gt = batch[3]
                r_pred_list = r_pred.cpu().tolist() if torch.is_tensor(r_pred) else list(r_pred)
                r_gt_list = r_gt.cpu().tolist() if torch.is_tensor(r_gt) else list(r_gt)
                if not isinstance(r_pred_list, list):
                    r_pred_list = [r_pred_list]
                if not isinstance(r_gt_list, list):
                    r_gt_list = [r_gt_list]
                saved_dict['r_pred'] = r_pred_list
                saved_dict['r_gt'] = r_gt_list
                saved_dict['dataloader_idx'] = [dataloader_idx] * len(r_pred_list)
            self.val_saved_list_for_go.append(saved_dict)

            self.log(f"dataloader{dataloader_idx}/val_loss", loss,
                    on_step=False, on_epoch=True, prog_bar=True,
                    sync_dist=True, batch_size=batch_size)
            self.log(f"dataloader{dataloader_idx}/reliability_loss", r_loss,
                    on_step=False, on_epoch=True, prog_bar=False,
                    sync_dist=True, batch_size=batch_size)
            if self.train_reliability_head_only:
                self.log("val/reliability_loss", r_loss,
                         on_step=False, on_epoch=True, prog_bar=False,
                         sync_dist=True, batch_size=batch_size)

        elif (dataloader_idx % 2) == 1:
            # Test set: collect predictions for BLEU/ROUGE and (if --report_go_wang_on_test) GO extraction + Wang
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return
            prot_batch, prompt_batch, r_tensor, target_dict = batch
            samples = {'prot_batch': prot_batch, 'prompt_batch': prompt_batch, 'reliability': r_tensor}
            predictions, r_pred, avg_conf, emb_out, r_prob_class1 = self.blip2.generate(
                samples,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_inference_len,
                min_length=self.min_inference_len
            )
            target_dict['predictions'] = predictions
            target_dict['confidences'] = [round(float(x), 4) for x in avg_conf]
            target_dict['reliability'] = [round(float(x), 4) for x in r_pred]
            target_dict['reliability_prob_class1'] = [round(float(x), 4) for x in r_prob_class1]
            B = len(predictions)
            target_dict['plm_mean_fp16'] = [emb_out['plm_mean_fp16'][i].clone() for i in range(B)]
            target_dict['qformer_feats_fp16'] = [emb_out['qformer_feats_fp16'][i].clone() for i in range(B)]
            target_dict['llm_last_fp16'] = [emb_out['llm_last_fp16'][i].clone() for i in range(B)]
            self.saved_dict_list.append(target_dict)

    def gather_dict_results(self, dict_list):
        if not dict_list:
            return []
        list_of_dict_list = [None for _ in range(self.trainer.world_size)]
        dist.all_gather_object(list_of_dict_list, dict_list)
        dict_list = [i for ii in list_of_dict_list for i in ii]
        keys = dict_list[0].keys()
        gathered_dict = {}

        def _flatten_field(v):
            """Expand batch field to a list (handles scalar tolist() / single int / str)."""
            if isinstance(v, (list, tuple)):
                return list(v)
            return [v]

        for key in keys:
            gathered_dict[key] = [x for d in dict_list for x in _flatten_field(d[key])]
        dict_list = []

        for i in range(len(gathered_dict['predictions'])):
            d = {k:gathered_dict[k][i] for k in keys}
            dict_list.append(d)
        return dict_list

    def load_ground_truth_go_from_test_set(self, result_list):
        """
        Load ground truth GO terms from test set based on indices.
        
        Args:
            result_list: List of dicts with 'indices' field
            
        Returns:
            dict: Mapping from index to GO terms list
        """
        if not self.test_set_path or not os.path.exists(self.test_set_path):
            print(f"[Warning] Test set path not provided or not found: {self.test_set_path}")
            return {}
        
        # Load all GO terms from test set
        go_dict = {}
        with open(self.test_set_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if len(row) >= 4:
                        last_col = row[3]  # GO terms column
                        if isinstance(last_col, str) and last_col.startswith('['):
                            try:
                                go_terms = ast.literal_eval(last_col)
                            except Exception:
                                go_terms = []
                        elif isinstance(last_col, list):
                            go_terms = last_col
                        else:
                            go_terms = []
                        go_dict[idx] = go_terms
                except Exception as e:
                    print(f"[Warning] Error parsing line {idx}: {e}")
                    go_dict[idx] = []
        
        return go_dict

    def load_ground_truth_text_from_valid_set(self):
        """Load target text from valid_set.json. Returns dict index -> text (row[1])."""
        if not self.valid_set_path or not os.path.exists(self.valid_set_path):
            return {}
        text_dict = {}
        with open(self.valid_set_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if len(row) >= 2:
                        text_dict[idx] = str(row[1]).strip()
                    else:
                        text_dict[idx] = ''
                except Exception:
                    text_dict[idx] = ''
        return text_dict

    def load_ground_truth_go_from_valid_set(self):
        """Load GO terms from valid_set.json (same format as test set). Returns dict index -> list of GO terms."""
        if not self.valid_set_path or not os.path.exists(self.valid_set_path):
            return {}
        go_dict = {}
        with open(self.valid_set_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if len(row) >= 4:
                        last_col = row[3]
                        if isinstance(last_col, str) and last_col.startswith('['):
                            try:
                                go_terms = ast.literal_eval(last_col)
                            except Exception:
                                go_terms = []
                        elif isinstance(last_col, list):
                            go_terms = last_col
                        else:
                            go_terms = []
                        go_dict[idx] = go_terms
                except Exception:
                    go_dict[idx] = []
        return go_dict

    def save_results(self, dict_list, log_prefix=""):
        if log_prefix:
            name = f'{log_prefix}_predictions.txt'
        else:
            name = 'predictions.txt'

        with open(os.path.join(self.logger.log_dir, name), 'w', encoding='utf8') as f:
            for d in dict_list:
                f.write(json.dumps(d, ensure_ascii=True,default=_json_default) + '\n')



    def on_validation_epoch_end(self):
        if getattr(self.trainer, "sanity_checking", False):
            return

        # Validation set BLEU/ROUGE: compute every epoch (val_go_gathered is always collected)
        val_go_gathered = self.gather_dict_results(self.val_saved_list_for_go) if self.val_saved_list_for_go else []
        if val_go_gathered:
            val_sorted = sorted(val_go_gathered, key=lambda d: d['indices'] if isinstance(d['indices'], (int, float)) else d['indices'][0])
            val_indices = [d['indices'] for d in val_sorted]
            val_predictions = [d['predictions'] for d in val_sorted]
            text_dict_val = self.load_ground_truth_text_from_valid_set()
            val_targets = [text_dict_val.get(i, '') for i in val_indices]
            if val_targets and any(t for t in val_targets):
                bleu2_val, bleu4_val, rouge_1_val, rouge_2_val, rouge_l_val, meteor_val = caption_evaluate(
                    val_predictions, val_targets, self.blip2.llm_tokenizer, self.max_inference_len,
                    verbose=(self.global_rank == 0))
                acc_val = evaluate_exact_match(val_predictions, val_targets)
                self.log("val/acc", acc_val, sync_dist=False)
                self.log("val/bleu2", bleu2_val, sync_dist=False)
                self.log("val/bleu4", bleu4_val, sync_dist=False)
                self.log("val/rouge_1", rouge_1_val, sync_dist=False)
                self.log("val/rouge_2", rouge_2_val, sync_dist=False)
                self.log("val/rouge_l", rouge_l_val, sync_dist=False)
                self.log("val/meteor_score", meteor_val, sync_dist=False)
                if self.global_rank == 0:
                    print(f'[Validation set] BLEU-2: {bleu2_val:.2f} BLEU-4: {bleu4_val:.2f} ROUGE-L: {rouge_l_val:.2f}')
            # Reliability head training: report Pearson/Spearman correlation every epoch (val and train)
            if self.train_reliability_head_only and val_go_gathered and 'r_pred' in val_go_gathered[0]:
                def _compute_accuracy(gathered, dl_idx, log_prefix, display_name):
                    subset = [d for d in gathered if d.get('dataloader_idx', 0) == dl_idx]
                    if not subset:
                        return
                    all_r_pred, all_r_gt = [], []
                    for d in subset:
                        rp, rg = d['r_pred'], d['r_gt']
                        if isinstance(rp, (list, tuple)):
                            all_r_pred.extend(float(x) for x in rp)
                            all_r_gt.extend(float(x) for x in rg)
                        else:
                            all_r_pred.append(float(rp))
                            all_r_gt.append(float(rg))
                    if not all_r_pred:
                        return
                    pred_arr = np.array(all_r_pred, dtype=np.float64)
                    gt_arr = np.array(all_r_gt, dtype=np.float64)
                    acc = float(np.isclose(pred_arr, gt_arr, atol=1e-4).mean())
                    self.log(f"{log_prefix}/reliability_accuracy", acc, sync_dist=False)
                    gt_is_class1 = np.isclose(gt_arr, 1.0, atol=1e-4)
                    pred_is_class1 = np.isclose(pred_arr, 1.0, atol=1e-4)
                    tp = int((gt_is_class1 & pred_is_class1).sum())
                    fp = int((~gt_is_class1 & pred_is_class1).sum())
                    fn = int((gt_is_class1 & ~pred_is_class1).sum())
                    class1_n = int(gt_is_class1.sum())
                    class1_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    class1_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    class1_f1 = (2 * class1_precision * class1_recall / (class1_precision + class1_recall)) if (class1_precision + class1_recall) > 0 else 0.0
                    self.log(f"{log_prefix}/reliability_class1_accuracy", class1_recall, sync_dist=False)
                    self.log(f"{log_prefix}/reliability_class1_precision", class1_precision, sync_dist=False)
                    self.log(f"{log_prefix}/reliability_class1_f1", class1_f1, sync_dist=False)
                    if self.global_rank == 0:
                        print(f'[{display_name}] Reliability accuracy: {acc:.4f} (n={len(pred_arr)}), class-1 recall: {class1_recall:.4f}, precision: {class1_precision:.4f}, F1: {class1_f1:.4f} (n_class1={class1_n})')

                # Validation set (dataloader_idx 0)
                _compute_accuracy(val_go_gathered, 0, "val", "Validation set")
                # Training set (dataloader_idx 2)
                _compute_accuracy(val_go_gathered, 2, "train", "Training set")
        self.val_saved_list_for_go = []

        if (self.current_epoch+1) % self.caption_eval_epoch != 0:
            return

        if self.save_predictions and hasattr(self, '_saved_even_list') and self._saved_even_list:
            even_list = self.gather_dict_results(self._saved_even_list)
            self._saved_even_list = []
            if self.global_rank == 0:
                out_even = os.path.join(self.logger.log_dir, f"val_epoch_end_{self.current_epoch+1}.json")
                with open(out_even, "w", encoding="utf-8") as f:
                    for d in even_list:
                        f.write(json.dumps({
                            'indices': d.get('indices'),
                            'predictions': d.get('predictions'),
                        }, ensure_ascii=True) + "\n")



        # result_list is from test set only (saved_dict_list filled in validation_step when dataloader_idx==1)
        result_list = self.gather_dict_results(self.saved_dict_list)
        self.saved_dict_list = []

        last_epoch = (self.current_epoch + 1) == self.trainer.max_epochs
        # val_go_gathered already gathered at top of this function (every epoch)

        if self.global_rank == 0:
            # Test/dataset eval: only on last epoch
            if last_epoch:
                print('Store the result.')
                result_list_sorted = sorted(result_list, key=lambda x: x.get('indices', float('inf')))
                result_list = result_list_sorted

                run_name = getattr(self.args, 'filename', 'run')
                out_dir = os.path.join("saved_results", run_name)
                if result_list and 'plm_mean_fp16' in result_list[0]:
                    os.makedirs(out_dir, exist_ok=True)
                    plm_stack = torch.stack([d['plm_mean_fp16'] for d in result_list])
                    qformer_stack = torch.stack([d['qformer_feats_fp16'] for d in result_list])
                    torch.save(plm_stack, os.path.join(out_dir, f"plm_mean_fp16_epoch{self.current_epoch+1}.pt"))
                    torch.save(qformer_stack, os.path.join(out_dir, f"qformer_feats_fp16_epoch{self.current_epoch+1}.pt"))
                    if 'llm_last_fp16' in result_list[0]:
                        llm_last_stack = torch.stack([d['llm_last_fp16'] for d in result_list])
                        torch.save(llm_last_stack, os.path.join(out_dir, f"llm_last_fp16_epoch{self.current_epoch+1}.pt"))
                    saved_names = ['plm_mean_fp16', 'qformer_feats_fp16'] + (['llm_last_fp16'] if 'llm_last_fp16' in result_list[0] else [])
                    print(f'[Test eval] Saved {", ".join(saved_names)} to {out_dir}')

                # print(f'result_list sample: {result_list[0] if result_list else "empty"}')

                all_predictions = [i['predictions'] for i in result_list]
                all_targets = [i['targets'] for i in result_list]
                all_confidences = [i['confidences'] for i in result_list]
                all_reliability = [i['reliability'] for i in result_list]
                all_indices = [i.get('indices', idx) for idx, i in enumerate(result_list)]
                ground_truth_go_dict = self.load_ground_truth_go_from_test_set(result_list)
                all_ground_truth_go = [ground_truth_go_dict.get(idx) for idx in all_indices]
                for idx, (result_idx, gt_go) in enumerate(zip(all_indices, all_ground_truth_go)):
                    result_list[idx]['gt_go'] = gt_go

                if self.report_go_wang_on_test:
                    print("Starting GO term extraction from test set predictions and references...")
                    cache_key = "go_extraction"
                    os.makedirs("saved_results", exist_ok=True)
                    run_name = getattr(self.args, 'filename', 'run')
                    prediction_file = os.path.join("saved_results", f"go_terms_from_predictions_{run_name}_epoch{self.current_epoch+1}.pkl")
                    reference_file = os.path.join("saved_results", f"go_terms_from_references_epoch{self.current_epoch+1}.pkl")
                    # Predictions: always extract (do not reuse cache), different models give different results
                    extracted_go_terms = process_texts_with_api(all_predictions)
                    # References: reuse cache from saved_results when available
                    reference_go_terms = load_or_process(reference_file, all_targets, "reference", cache_key)
                    # Filter all GO terms to molecular_function only using evals/tools/go_files.tsv
                    mf_go_ids = set()
                    if os.path.exists(self.go_files_tsv_path):
                        mf_go_ids = load_mf_go_ids_from_tsv(self.go_files_tsv_path, 'molecular_function')
                        print(f"Filtering GO terms to molecular_function only: {len(mf_go_ids)} MF terms from {self.go_files_tsv_path}")
                    else:
                        print(f"[Warning] go_files.tsv not found at {self.go_files_tsv_path}, skipping MF filter")
                    gt_go_raw = all_ground_truth_go
                    ref_go_raw = reference_go_terms
                    pred_go_raw = extracted_go_terms
                    if mf_go_ids:
                        gt_go_raw = filter_go_terms_by_set(gt_go_raw, mf_go_ids)
                        ref_go_raw = filter_go_terms_by_set(ref_go_raw, mf_go_ids)
                        pred_go_raw = filter_go_terms_by_set(pred_go_raw, mf_go_ids)
                    assert len(gt_go_raw) == len(pred_go_raw) == len(ref_go_raw) == len(result_list), (
                        f"Length mismatch: gt={len(gt_go_raw)} pred={len(pred_go_raw)} ref={len(ref_go_raw)} result_list={len(result_list)}"
                    )
                    for idx in range(len(result_list)):
                        result_list[idx]['gt_go'] = gt_go_raw[idx]
                        result_list[idx]['go_terms_from_predictions'] = pred_go_raw[idx]
                        result_list[idx]['go_terms_from_references'] = ref_go_raw[idx]
                    with open(prediction_file, 'wb') as f:
                        pickle.dump(pred_go_raw, f)
                    if all_ground_truth_go:
                        print("Computing ontology metrics (ground truth vs reference GO, ground truth vs prediction GO, MF only)...")
                        try:
                            assert len(gt_go_raw) == len(ref_go_raw) == len(pred_go_raw), (
                                f"GO list length mismatch: gt={len(gt_go_raw)} ref={len(ref_go_raw)} pred={len(pred_go_raw)}"
                            )
                            gt_go = gt_go_raw
                            ref_go = ref_go_raw
                            pred_go = pred_go_raw
                            ref_wang_similarity, _ = compute_wang_similarity(gt_go, ref_go)
                            ref_ia_distance = compute_InfoAccretion_distance(gt_go, ref_go, ia_file=self.ia_path, k=2)
                            ref_jaccard_similarity = compute_jaccard_similarity(gt_go, ref_go)
                            pred_wang_similarity, _ = compute_wang_similarity(gt_go, pred_go)
                            pred_ia_distance = compute_InfoAccretion_distance(gt_go, pred_go, ia_file=self.ia_path, k=2)
                            pred_jaccard_similarity = compute_jaccard_similarity(gt_go, pred_go)
                            self.log("dataset/go_wang_similarity_reference", ref_wang_similarity, sync_dist=False)
                            self.log("dataset/go_ia_distance_reference", ref_ia_distance, sync_dist=False)
                            self.log("dataset/go_jaccard_similarity_reference", ref_jaccard_similarity, sync_dist=False)
                            self.log("dataset/go_wang_similarity_prediction", pred_wang_similarity, sync_dist=False)
                            self.log("dataset/go_ia_distance_prediction", pred_ia_distance, sync_dist=False)
                            self.log("dataset/go_jaccard_similarity_prediction", pred_jaccard_similarity, sync_dist=False)
                            print(f'Reference vs GT: Wang {ref_wang_similarity:.4f} IA {ref_ia_distance:.4f} Jaccard {ref_jaccard_similarity:.4f}')
                            print(f'Prediction vs GT: Wang {pred_wang_similarity:.4f} IA {pred_ia_distance:.4f} Jaccard {pred_jaccard_similarity:.4f}')
                        except Exception as e:
                            print(f"[Warning] Failed to compute ontology metrics: {e}")

                self.save_results(result_list, 'dataset')
                log_prefix = 'dataset'
                mean_confidences = _mean_conf(all_confidences)
                # Note: BLEU/ROUGE/Wang above are on the *test* set (dataloader_idx 1), not validation set.
                mean_reliability = _mean_conf(all_reliability)
                print('[Inference training subset (dataset)] BLEU/ROUGE/Meteor:')
                bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                    caption_evaluate(all_predictions, all_targets, self.blip2.llm_tokenizer, self.max_inference_len)
                acc = evaluate_exact_match(all_predictions, all_targets)
                self.log(f"{log_prefix}/acc", acc, sync_dist=False)
                self.log(f"{log_prefix}/bleu2", bleu2, sync_dist=False)
                self.log(f"{log_prefix}/bleu4", bleu4, sync_dist=False)
                self.log(f"{log_prefix}/rouge_1", rouge_1, sync_dist=False)
                self.log(f"{log_prefix}/rouge_2", rouge_2, sync_dist=False)
                self.log(f"{log_prefix}/rouge_l", rouge_l, sync_dist=False)
                self.log(f"{log_prefix}/meteor_score", meteor_score, sync_dist=False)
                self.log(f"{log_prefix}/avg_confidences", mean_confidences, sync_dist=False)
                self.log(f"{log_prefix}/avg_reliability", mean_reliability, sync_dist=False)
                # print('avg_confidences', mean_confidences)
                # print('mean_reliability', mean_reliability)

            # Validation set BLEU/ROUGE already computed every epoch at top of this function

            # Last-epoch only (and only when --report_go_wang_on_val): validation set GO extraction + Wang vs valid_set.json
            if self.report_go_wang_on_val and last_epoch and val_go_gathered:
                val_sorted = sorted(val_go_gathered, key=lambda d: d['indices'] if isinstance(d['indices'], (int, float)) else d['indices'][0])
                val_indices = [d['indices'] for d in val_sorted]
                val_predictions = [d['predictions'] for d in val_sorted]
                go_dict_val = self.load_ground_truth_go_from_valid_set()
                gt_go_list = [go_dict_val.get(i, []) for i in val_indices]
                try:
                    pred_go_list = process_texts_with_api(val_predictions)
                    val_wang_mean, _ = compute_wang_similarity(gt_go_list, pred_go_list)
                    self.log("val/go_wang_similarity", val_wang_mean, sync_dist=False)
                    print(f'[Validation set] Last epoch GO Wang similarity (pred vs valid_set.json): {val_wang_mean:.4f}')
                except Exception as e:
                    print(f'[Warning] Validation set GO Wang failed: {e}')

        self.val_saved_list_for_go = []

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = batch[1].input_ids.size(0)
        blip_batch = batch[:4] if self.train_reliability_head_only else batch[:-1]
        loss, r_loss = self.blip2(blip_batch, return_pred=False)

        self.log("loss", loss, sync_dist=True, batch_size=batch_size)
        self.log("reliability_loss", r_loss, batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        # Either full-dataset generation loss or subset reliability loss
        if self.train_reliability_head_only:
            return r_loss
        return loss


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ProtBlip2")
        # train mode
        parser.add_argument('--save_every_n_epochs', type=int, default=0)

        # Bert
        parser.add_argument('--bert_name', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        # OPT
        parser.add_argument('--llm_name', type=str, default="facebook/galactica-1.3b")
        parser.add_argument('--num_beams', type=int, default=3)
        parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--max_inference_len', type=int, default=256)
        parser.add_argument('--min_inference_len', type=int, default=1)
        parser.add_argument('--llm_tune', type=str, default='freeze')
        parser.add_argument('--peft_config', type=str, default='')
        parser.add_argument('--peft_dir', type=str, default='')

        ## plm model
        parser.add_argument('--plm_model', type=str, default='facebook/esm2_t30_150M_UR50D')
        parser.add_argument('--plm_tune', type=str, default='freeze')

        parser.add_argument('--plm_lora_r', type=int, default=8)
        parser.add_argument('--plm_lora_alpha', type=int, default=8)
        parser.add_argument('--plm_lora_dropout', type=int, default=0.1)

        ## lora config
        parser.add_argument('--lora_r', type=int, default=16)
        parser.add_argument('--lora_alpha', type=int, default=16)
        parser.add_argument('--lora_dropout', type=int, default=0.1)
        parser.add_argument('--enbale_gradient_checkpointing', action='store_true', default=False)

        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler')

        # Reliability head specific optimization parameters
        parser.add_argument('--reliability_lr', type=float, default=1e-4, help='learning rate for reliability head (if None, uses init_lr)')
        parser.add_argument('--stage1_path', type=str, default='')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=10)
        parser.add_argument('--save_predictions', action='store_true', default=False,
                            help='Save training and validation predictions to JSON files')
        
        # Encoder selection (automatically inferred from plm_model but can be explicit)
        parser.add_argument('--encoder_type', type=str, default='auto', choices=['auto', 'esm2', 'esmc'], help='Protein encoder type: auto (infer from plm_model), esm2 (HuggingFace ESM2), or esmc (official ESM-C package)')
        # GO term extraction parameters
        parser.add_argument('--report_go_wang_on_test', action='store_true', default=False, help='On test set (dataloader_idx 1): extract GO from predictions, compute Wang/IA/Jaccard, store extracted_go_terms per row')
        parser.add_argument('--ia_path', type=str, default='evals/tools/IA.txt', help='Path to Information Accretion (IA) file')
        parser.add_argument('--report_go_wang_on_val', action='store_true', default=False, help='On last epoch only: extract GO from val predictions (process_texts_with_api) and compute Wang vs valid_set.json')
        parser.add_argument('--go_files_tsv_path', type=str, default='evals/tools/go_files.tsv', help='Path to go_files.tsv (go_id, aspect) to filter GO terms to molecular_function only')

        return parent_parser



def evaluate_exact_match(predictions, targets):
    acc = 0
    for prediction, target in zip(predictions, targets):
        if prediction.strip() == target.strip():
            acc += 1
    acc = round(acc / len(predictions) * 100, 2)
    return acc