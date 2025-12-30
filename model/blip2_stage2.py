import os
import torch
from model.blip2_opt import Blip2OPT
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
import ast
import pickle
import csv
from evals.Ontology_tools.InfoAccretion import compute_InfoAccretion_distance
from evals.Ontology_tools.wang_similarity import compute_wang_similarity
from evals.Ontology_tools.jaccard_similarity import compute_jaccard_similarity
from evals.Ontology_tools.Onto_extraction import process_texts_with_api, process_texts_with_ollama
import torch.distributed as dist
# from peft import LoraConfig, TaskType
from typing import Any, Dict
from model.help_funcs import caption_evaluate, AttrDict


def _mean_conf(all_confidences):
    import torch, numpy as np
    vals = []
    for c in all_confidences:
        if isinstance(c, torch.Tensor):
            vals.append(c.detach().float().cpu().reshape(-1))
        elif isinstance(c, (list, tuple, np.ndarray)):
            vals.append(torch.as_tensor(c, dtype=torch.float32).reshape(-1))
        elif isinstance(c, (int, float)):
            vals.append(torch.tensor([c], dtype=torch.float32))
        else:
            # unrecognized -> skip
            continue
    m = torch.cat(vals, dim=0).mean().item()
    return round(float(m), 4) 

def _json_default(o):
    """Convert non-JSON types to JSON-serializable Python types."""
    # Tensors -> Python scalars or lists
    if isinstance(o, torch.Tensor):
        return o.item() if o.dim() == 0 else o.detach().cpu().tolist()
    # Numpy scalars/arrays -> Python scalars/lists
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)):    return bool(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    # Sets -> lists
    if isinstance(o, set):            return list(o)
    # Bytes -> str (utf-8) or hex fallback
    if isinstance(o, (bytes, bytearray)):
        try:
            return o.decode("utf-8")
        except Exception:
            return o.hex()
    # Fallback: stringify anything else
    return str(o)


def load_or_process(file_path, data, data_name, safe_model_type):
    """
    Load data from a cached file if available; otherwise, process and save it.
    """
    try:
        with open(file_path, 'rb') as file:
            print(f"Loading {data_name} GO terms from {file_path} processing by {safe_model_type}...")
            return pickle.load(file)
    except FileNotFoundError:
        print(f"File not found. Processing {data_name} using {safe_model_type}...")
        # if safe_model_type.lower().startswith("gpt"):
        result = process_texts_with_api(data, safe_model_type)
        # else:
        #     result = process_texts_with_ollama(data, safe_model_type)
        with open(file_path, 'wb') as file:
            pickle.dump(result, file)
        print(f"Processed {data_name} GO terms saved to {file_path}.")
        return result


def filter_existing_go_terms(go_list, extracted_go_terms):
    """
    Filter and return only existing GO terms from the example list in a list of lists format.
    """
    extracted_ids = {term[0] for term in extracted_go_terms}
    filtered_terms = [[go_id for go_id in go_set if go_id in extracted_ids] for go_set in go_list]
    return filtered_terms


def read_go_terms_from_csv(file_path):
    """
    Read GO terms from the saved CSV file.
    """
    go_terms = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            go_terms.append(row)
    return go_terms


def _is_nested_list(x):
    """Return True if x is a list of lists/tuples/sets (or empty list)."""
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], (list, tuple, set)))


def empty_rate_nested(xs):
    """
    For nested list[list[str]]: proportion of empty inner lists.
    For flat list[str]: 0.0 if non-empty else 1.0.
    """
    if not isinstance(xs, list):
        return float("nan")
    if _is_nested_list(xs):
        n = len(xs) if xs else 1
        return sum(1 for v in xs if not v) / n
    return 0.0 if xs else 1.0


def build_joint_nonempty_mask(pred_list, ref_list):
    """
    Build a boolean mask keeping indices where BOTH pred_list[i] and ref_list[i] are non-empty.
    Assumes nested list[list[str]]. If lengths differ, truncate to the min length.
    """
    if not (_is_nested_list(pred_list) and _is_nested_list(ref_list)):
        # Treat as single 'sample'
        return [bool(pred_list) and bool(ref_list)]
    n = min(len(pred_list), len(ref_list))
    if len(pred_list) != len(ref_list):
        print(f"[WARN] Different lengths: pred={len(pred_list)} ref={len(ref_list)}; truncating to {n}.")
    return [bool(pred_list[i]) and bool(ref_list[i]) for i in range(n)]


def filter_parallel_by_mask(seq_list, mask):
    """
    Apply a boolean mask to each sequence list in seq_list in parallel.
    Each element must be nested list[list[str]] aligned by index.
    Returns the filtered lists in the same order.
    """
    out = []
    for seq in seq_list:
        if not _is_nested_list(seq):
            # Enforce nested form for safety
            raise ValueError("Expected nested list[list[str]] for masking.")
        out.append([v for v, m in zip(seq, mask) if m])
    return out




class Blip2Stage2(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Save ALL parameters (including frozen PLM and LLM base weights)
        # This makes checkpoints larger (~14GB) but self-contained
        # Original behavior: only save trainable parameters (commented out below)
        pass
        
        # # Original code - only save trainable parameters (~557MB):
        # to_be_removed = []
        # for key, value in checkpoint['state_dict'].items():
        #     try:
        #         if not self.get_parameter(key).requires_grad:
        #             to_be_removed.append(key)
        #     except AttributeError:
        #         to_be_removed.append(key)
        # for key in to_be_removed:
        #     checkpoint['state_dict'].pop(key)
    
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        self.head = args.head
        self.threshold = args.threshold
        self.caption_eval_epoch = args.caption_eval_epoch
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_inference_len = args.max_inference_len
        self.min_inference_len = args.min_inference_len
        self.llm_tune = args.llm_tune
        self.enable_flash = args.enable_flash
        
        # Reliability training parameters
        self.reliability_weight_max = getattr(args, 'reliability_weight', 0.5)
        self.reliability_warmup_steps = getattr(args, 'reliability_warmup_steps', 500)
        self.reliability_warmup_epochs = getattr(args, 'reliability_warmup_epochs', 2)
        self.use_focal_loss = getattr(args, 'use_focal_loss', False)
        self.focal_alpha = getattr(args, 'focal_alpha', 0.25)
        self.focal_gamma = getattr(args, 'focal_gamma', 2.0)
        
        # GO term extraction parameters
        self.extract_go_terms = getattr(args, 'extract_go_terms', False)
        self.go_extraction_model = getattr(args, 'go_extraction_model', 'gpt-4o')
        self.go_extraction_backend = getattr(args, 'go_extraction_backend', 'api')  # 'api' or 'ollama'
        self.ia_path = getattr(args, 'ia_path', 'evals/Ontology_tools/IA.txt')
        self.test_set_path = getattr(args, 'test_set_path', '')
        self.mf_go_terms_path = getattr(args, 'mf_go_terms_path', 'evals/Ontology_tools/molecular_function_go_terms.csv')
        
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
        self.save_hyperparameters(args)

    def load_from_stage1_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['state_dict']
        state_dict = {k.split('blip2qformer.')[1]:v for k, v in state_dict.items()}
        self.blip2.load_state_dict(state_dict, strict=False)
        return self
    
    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        
        # Create different parameter groups for different modules
        # Get all parameters except reliability_head
        main_params = []
        reliability_params = []
        
        for name, param in self.named_parameters():
            if 'reliability_head' in name:
                reliability_params.append(param)
            else:
                main_params.append(param)
        
        # Set learning rate and weight decay for reliability_head
        reliability_lr = self.args.reliability_lr if self.args.reliability_lr is not None else self.args.init_lr
        reliability_weight_decay = self.args.reliability_weight_decay if self.args.reliability_weight_decay is not None else self.args.weight_decay
        
        # Create optimizer with different learning rates for different parameter groups
        optimizer = optim.AdamW([
            {'params': main_params, 'lr': self.args.init_lr, 'weight_decay': self.args.weight_decay},
            {'params': reliability_params, 'lr': reliability_lr, 'weight_decay': reliability_weight_decay}
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
        if self.enable_flash:
            replace_opt_attn_with_original_attn()
        self.saved_dict_list = []
        self.prediction_list0 = []
        self.target_list0 = []
        self.prediction_list1 = []
        self.target_list1 = []

        self._saved_even_list = []    # list of dicts: {'indices': [...], 'predictions': [...]}

    def on_train_epoch_start(self) -> None:
        # for training even-like collection (indices + predicted text)
        self._train_saved_list = []  # list of dicts: {'indices': [...], 'predictions': [...]}


    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if (dataloader_idx % 2) == 0:

            if self.head == 'generation':

                text_batch = batch[1]
                batch_size = text_batch.input_ids.shape[0]
                collect_now = ((self.trainer.current_epoch + 1) % self.caption_eval_epoch == 0)
                # if collect_now:
                #     # expect (loss, pred_texts)
                #     loss, r_loss, pred_texts = self.blip2(batch[:-1], return_pred=True)
                #     idx_list = batch[-1].detach().cpu().tolist()
                #     # push a dict-of-lists, same as odd path style
                #     # print('pred_texts',pred_texts,len(pred_texts))
                    
                #     self._saved_even_list.append({
                #         'indices': idx_list,
                #         'predictions': pred_texts,   # list[str], len=B
                #     })
                # else:
                loss, r_loss = self.blip2(batch[:-1], return_pred=False)

                self.log(f"dataloader{dataloader_idx}/val_loss", loss,
                        on_step=False, on_epoch=True, prog_bar=True,
                        sync_dist=True, batch_size=batch_size)  # keep tensor, no float()

                self.log(f"dataloader{dataloader_idx}/reliability_loss", r_loss,
                        on_step=False, on_epoch=True, prog_bar=False,
                        sync_dist=True, batch_size=batch_size)

            elif self.head == 'classification':

                prot_batch, go_class = batch
                # print('go_class',go_class)
                batch_size = go_class.size(0)
                loss = self.blip2(batch, return_pred=False)
                self.log(f"dataloader{dataloader_idx}/val_loss", loss, sync_dist=True, batch_size=batch_size)


        elif (dataloader_idx % 2) == 1:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            if self.head == 'generation':
                prot_batch, prompt_batch, r_tensor, target_dict = batch
                ###============== Captioning Results ===================###
                samples = {'prot_batch': prot_batch, 'prompt_batch': prompt_batch, 'reliability': r_tensor}
                predictions, r_pred, avg_conf, emb_out = self.blip2.generate(
                    samples, 
                    do_sample=self.do_sample,
                    num_beams=self.num_beams,
                    max_length=self.max_inference_len,
                    min_length=self.min_inference_len
                )
                target_dict['predictions'] = predictions                     # list[str]
                target_dict['confidences'] = [round(float(x), 4) for x in avg_conf]
                target_dict['predicted_reliability'] = [round(float(x), 4) for x in r_pred]

                # plm_rows = emb_out["plm_mean_fp16"]         # torch.Tensor [B, D_plm], cpu, float16
                # qfm_rows = emb_out["qformer_feats_fp16"]    # torch.Tensor [B, D_llm], cpu, float16

                # target_dict['emb_plm']     = [row.tolist() for row in plm_rows]   # List[List[float16]]
                # target_dict['emb_qformer'] = [row.tolist() for row in qfm_rows]   # List[List[float16]]

                self.saved_dict_list.append(target_dict)                     # CPU+Python

            elif self.head == 'classification':
                prot_batch, go_class, target_dict = batch
                ###============== Inference Results ===================###
                samples = {'prot_batch': prot_batch}
                predictions, avg_conf = self.blip2.inferred_class(
                    samples, 
                    threshold = self.threshold
                )
                target_dict['predictions'] = predictions                 # list[list[str]] or similar format
                target_dict['confidences'] = [round(float(x), 4) for x in avg_conf]
                self.saved_dict_list.append(target_dict)                     # CPU+Python

    def gather_dict_results(self, dict_list):
        if not dict_list:  # No results on this rank
            return []
        list_of_dict_list = [None for _ in range(self.trainer.world_size)]
        dist.all_gather_object(list_of_dict_list, dict_list)
        dict_list = [i for ii in list_of_dict_list for i in ii] ## dict list, each dict has values that are lists of predictions, etc.
        keys = dict_list[0].keys()
        gathered_dict = {} # each value is a list of predictions, etc.
        for key in keys:
            gathered_dict[key] = [i for d in dict_list for i in d[key]]
        dict_list = []

        # print('gathered_dict',gathered_dict)
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
                    # Format: [sequence, text, value, GO_list_string]
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




    def save_results(self, dict_list, log_prefix=""):
        ## save the results
        if log_prefix:
            name = f'{log_prefix}_predictions.txt'
        else:
            name = 'predictions.txt'

        # print(len(dict_list))
        with open(os.path.join(self.logger.log_dir, name), 'w', encoding='utf8') as f:
            for d in dict_list:
                f.write(json.dumps(d, ensure_ascii=True,default=_json_default) + '\n')



    def on_validation_epoch_end(self):
        
        if getattr(self.trainer, "sanity_checking", False):
            return
        if self.enable_flash:
            replace_opt_attn_with_flash_attn()
        if (self.current_epoch+1) % self.caption_eval_epoch != 0:
            return 

        # even_list = self.gather_dict_results(self._saved_even_list)

        # # print('even_list',even_list)
        # self._saved_even_list = []
        # if self.global_rank == 0:
        #     out_even = os.path.join(self.logger.log_dir, f"val_epoch_end_{self.current_epoch+1}.json")
        #     with open(out_even, "w", encoding="utf-8") as f:
        #         for d in even_list:
        #             f.write(json.dumps({
        #                 'indices': d.get('indices'),
        #                 'predictions': d.get('predictions'),
        #             }, ensure_ascii=True) + "\n")



        result_list = self.gather_dict_results(self.saved_dict_list)

        ## empty cache
        self.saved_dict_list = []
        
        if self.global_rank == 0:
            print('Store the result.')
            
            if self.head == 'generation':
                # Sort result_list by indices to handle unordered indices
                result_list_sorted = sorted(result_list, key=lambda x: x.get('indices', float('inf')))
                result_list = result_list_sorted
                
                print(f'result_list sample: {result_list[0] if result_list else "empty"}')
                
                all_predictions = [i['predictions'] for i in result_list]
                all_targets = [i['targets'] for i in result_list]
                all_confidences = [i['confidences'] for i in result_list]
                all_reliability = [i['predicted_reliability'] for i in result_list]
                all_indices = [i.get('indices', idx) for idx, i in enumerate(result_list)]
                
                # Load ground truth GO from test set
                ground_truth_go_dict = self.load_ground_truth_go_from_test_set(result_list)
                all_ground_truth_go = [ground_truth_go_dict.get(idx) for idx in all_indices]
                
                # Add ground truth GO to result_list
                for idx, (result_idx, gt_go) in enumerate(zip(all_indices, all_ground_truth_go)):
                    result_list[idx]['Ground Truth GO'] = gt_go
                
                # Extract GO terms from predictions if enabled
                if self.extract_go_terms:
                    print("Starting GO term extraction from predictions and references...")
                    safe_model_type = self.go_extraction_model.split("/")[-1]
                    
                    # Create cache directory if needed
                    os.makedirs("saved_results", exist_ok=True)
                    
                    # Extract GO terms from predictions
                    prediction_file = os.path.join("saved_results", f"val_prediction_human_seed44_go_terms_{safe_model_type}_epoch{self.current_epoch+1}.pkl")
                    extracted_go_terms = load_or_process(prediction_file, all_predictions, "predictions", safe_model_type)
                    
                    # Extract GO terms from reference texts (targets)
                    reference_file = os.path.join("saved_results", f"val_reference_go_terms_{safe_model_type}_epoch{self.current_epoch+1}.pkl")
                    reference_go_terms = load_or_process(reference_file, all_targets, "reference", safe_model_type)
                    
                    # Calculate empty rates BEFORE filtering
                    pred_empty_rate = empty_rate_nested(extracted_go_terms)
                    ref_empty_rate = empty_rate_nested(reference_go_terms)
                    print(f"Empty rate (extracted_go_terms): {pred_empty_rate:.3%}")
                    print(f"Empty rate (reference_go_terms): {ref_empty_rate:.3%}")
                    
                    # Add extracted GO terms and reference GO terms to result_list
                    # for idx, (pred_go, ref_go) in enumerate(zip(extracted_go_terms, reference_go_terms)):
                    #     if idx < len(result_list):
                    #         result_list[idx]['extracted_go_terms'] = pred_go
                    
                    # Log empty rates
                    self.log("dataset/extracted_go_empty_rate", pred_empty_rate, sync_dist=False)
                    self.log("dataset/reference_go_empty_rate", ref_empty_rate, sync_dist=False)
                    
                    # Compute ontology metrics if ground truth GO exists
                    if all_ground_truth_go:
                        print("Computing ontology metrics for extracted GO terms...")
                        
                        # Load MF GO terms for filtering
                        try:
                            MF_go_terms = read_go_terms_from_csv(self.mf_go_terms_path)
                            print(f"Extracted {len(MF_go_terms)} GO terms under molecular_function namespace.")
                            
                            # Filter to MF namespace
                            all_ground_truth_go_filtered = filter_existing_go_terms(all_ground_truth_go, MF_go_terms)
                            extracted_go_terms_filtered = filter_existing_go_terms(extracted_go_terms, MF_go_terms)
                            reference_go_terms_filtered = filter_existing_go_terms(reference_go_terms, MF_go_terms)
                            
                            # Update result_list with filtered GO terms
                            for idx in range(len(result_list)):
                                # if idx < len(all_ground_truth_go_filtered):
                                #     result_list[idx]['Ground Truth GO'] = all_ground_truth_go_filtered[idx]
                                if idx < len(extracted_go_terms_filtered):
                                    result_list[idx]['extracted_go_terms'] = extracted_go_terms_filtered[idx]
                                # if idx < len(reference_go_terms_filtered):
                                #     result_list[idx]['reference_go_terms'] = reference_go_terms_filtered[idx]
                            
                            # Build joint mask for non-empty pairs
                            mask_joint = build_joint_nonempty_mask(extracted_go_terms_filtered, all_ground_truth_go_filtered)
                            
                            # Filter all three lists using the joint mask
                            all_GO_joint, extracted_go_joint, gt_go_joint = filter_parallel_by_mask(
                                [all_ground_truth_go_filtered, extracted_go_terms_filtered, all_ground_truth_go_filtered], 
                                mask_joint
                            )
                            
                            # Also filter reference GO terms with same mask
                            reference_go_joint = [v for v, m in zip(reference_go_terms_filtered, mask_joint) if m]
                            
                            kept = len(extracted_go_joint)
                            total = len(mask_joint)
                            print(f"Kept pairs (both non-empty): {kept}/{total} ({(kept/total if total else 0.0):.1%})")
                            
                            if kept > 0:
                                # Compute ontology metrics for predictions vs ground truth
                                try:
                                    go_wang_similarity = compute_wang_similarity(gt_go_joint, extracted_go_joint)
                                    go_ia_distance = compute_InfoAccretion_distance(gt_go_joint, extracted_go_joint, 
                                                                                     ia_file=self.ia_path, k=2)
                                    go_jaccard_similarity = compute_jaccard_similarity(gt_go_joint, extracted_go_joint)
                                    
                                    # Compute ontology metrics for reference vs ground truth
                                    ref_wang_similarity = compute_wang_similarity(gt_go_joint, reference_go_joint)
                                    ref_ia_distance = compute_InfoAccretion_distance(gt_go_joint, reference_go_joint,
                                                                                      ia_file=self.ia_path, k=2)
                                    ref_jaccard_similarity = compute_jaccard_similarity(gt_go_joint, reference_go_joint)
                                    
                                    # Log the ontology metrics for predictions
                                    self.log("dataset/go_wang_similarity_prediction", go_wang_similarity, sync_dist=False)
                                    self.log("dataset/go_ia_distance_prediction", go_ia_distance, sync_dist=False)
                                    self.log("dataset/go_jaccard_similarity_prediction", go_jaccard_similarity, sync_dist=False)
                                    
                                    # Log the ontology metrics for reference
                                    self.log("dataset/go_wang_similarity_reference", ref_wang_similarity, sync_dist=False)
                                    self.log("dataset/go_ia_distance_reference", ref_ia_distance, sync_dist=False)
                                    self.log("dataset/go_jaccard_similarity_reference", ref_jaccard_similarity, sync_dist=False)
                                    
                                    print(f'Prediction GO Wang Similarity: {go_wang_similarity:.4f}')
                                    print(f'Prediction GO Information Accretion Distance: {go_ia_distance:.4f}')
                                    print(f'Prediction GO Jaccard Similarity: {go_jaccard_similarity:.4f}')
                                    print(f'Reference GO Wang Similarity: {ref_wang_similarity:.4f}')
                                    print(f'Reference GO Information Accretion Distance: {ref_ia_distance:.4f}')
                                    print(f'Reference GO Jaccard Similarity: {ref_jaccard_similarity:.4f}')
                                    print(f'Valid pairs for GO evaluation: {kept}/{total}')
                                except Exception as e:
                                    print(f"[Warning] Failed to compute ontology metrics: {e}")
                            else:
                                print("[Warning] No valid GO term pairs for evaluation (all predictions or ground truth are empty)")
                        except Exception as e:
                            print(f"[Warning] Failed to load/filter MF GO terms: {e}")
                
                # Save results with extracted GO terms and ground truth GO
                self.save_results(result_list, 'dataset')
                
                log_prefix = 'dataset' ## fixme: this is just a placeholder

                mean_confidences = _mean_conf(all_confidences)
                mean_reliability = _mean_conf(all_reliability)
                ## evaluate captioning
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
                print('avg_confidences',mean_confidences)
                print('mean_reliability',mean_reliability)

            elif self.head == 'classification':
                all_predictions = [i['predictions'] for i in result_list]
                all_targets = [i['Ground Truth GO'] for i in result_list]
                all_confidences = [i['confidences'] for i in result_list]

                mean_confidences = _mean_conf(all_confidences)
                ia_path = "evals/Ontology_tools/IA.txt"
                mean_wang_similarity = compute_wang_similarity(all_targets, all_predictions)
                mean_IA = compute_InfoAccretion_distance(all_targets, all_predictions, ia_file=ia_path, k=2)
                mean_jaccard_similarity  = compute_jaccard_similarity(all_targets, all_predictions)
                self.log(f"wang_similarity", mean_wang_similarity, sync_dist=False)
                self.log(f"Information_Accretion", mean_IA, sync_dist=False)
                self.log(f"jaccard_similarity", mean_jaccard_similarity, sync_dist=False)
                self.log(f"avg_confidences", mean_confidences, sync_dist=False)

                print('wang_similarity ',mean_wang_similarity)
                print('Information_Accretion ',mean_IA)
                print('jaccard_similarity ',mean_jaccard_similarity)
                print('avg_confidences',mean_confidences)

    def _compute_reliability_weight(self):
        """
        Compute dynamic reliability weight with warmup strategy.
        
        Strategy:
        - Phase 1 (warmup_epochs): Gradually increase from 0 to max_weight
        - Phase 2 (after warmup): Use max_weight
        
        This allows the main task to stabilize before introducing reliability loss.
        """
        current_epoch = self.trainer.current_epoch
        global_step = self.trainer.global_step
        
        # Option 1: Epoch-based warmup
        if current_epoch < self.reliability_warmup_epochs:
            # Linear warmup over epochs
            weight = self.reliability_weight_max * (current_epoch / max(self.reliability_warmup_epochs, 1))
        # Option 2: Step-based warmup (more fine-grained)
        elif global_step < self.reliability_warmup_steps:
            # Linear warmup over steps
            weight = self.reliability_weight_max * (global_step / max(self.reliability_warmup_steps, 1))
        else:
            # After warmup, use full weight
            weight = self.reliability_weight_max
        
        return weight

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        if self.head == 'generation':
        
            batch_size = batch[1].input_ids.size(0)
            collect_now = ((self.trainer.current_epoch + 1) % self.caption_eval_epoch == 0)
            # if collect_now:
            #     # expect (loss, pred_texts)
            #     loss, r_loss, pred_texts = self.blip2(batch[:-1], return_pred=True)
            #     idx_list = batch[-1].detach().cpu().tolist()
            #     self._train_saved_list.append({
            #         'indices': idx_list,
            #         'predictions': pred_texts,   # list[str], len=B
            #     })
            # else:
            loss, r_loss = self.blip2(batch[:-1], return_pred=False)
            
            # Compute dynamic reliability weight with warmup
            reliability_weight = self._compute_reliability_weight()
            
            # Logging
            self.log("loss", loss, sync_dist=True, batch_size=batch_size)
            self.log("reliability_loss", r_loss, batch_size=batch_size, sync_dist=True)
            self.log("reliability_weight", reliability_weight, batch_size=batch_size, sync_dist=True)
            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
            
            # Combined loss with dynamic weighting
            total_loss = loss + reliability_weight * r_loss
            return total_loss
        
        elif self.head == 'classification':

            prot_batch, go_class = batch
            batch_size = go_class.size(0)
            loss = self.blip2(batch, return_pred=False)
            self.log("loss", loss, sync_dist=True, batch_size=batch_size)
            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)

            return loss




    # def on_train_epoch_end(self) -> None:
    #     # mirror odd-path style: only process on collection epochs
    #     if (self.trainer.current_epoch + 1) % self.caption_eval_epoch != 0:
    #         return
    #     if not self._train_saved_list:
    #         return
    #     # gather across GPUs just like odd path
    #     # print('self._train_saved_list',self._train_saved_list)
    #     gathered = self.gather_dict_results(self._train_saved_list)  # reuse your helper
    #     self._train_saved_list = []
    #     if self.global_rank == 0:
    #         out = os.path.join(self.logger.log_dir, f"train_epoch_end{self.trainer.current_epoch+1}.json")
    #         with open(out, "w", encoding="utf-8") as f:
    #             for d in gathered:  # d is a flat dict with lists already unrolled by helper
    #                 # keep only what we collected
    #                 f.write(json.dumps({
    #                     'indices': d.get('indices'),
    #                     'predictions': d.get('predictions'),
    #                 }, ensure_ascii=True) + "\n")


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
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        
        # Reliability head training strategy parameters
        parser.add_argument('--reliability_weight', type=float, default=0.5, 
                            help='Maximum weight for reliability loss (will warmup to this value)')
        parser.add_argument('--reliability_warmup_epochs', type=int, default=2, 
                            help='Number of epochs to warmup reliability loss (0 weight -> full weight)')
        parser.add_argument('--reliability_warmup_steps', type=int, default=500, 
                            help='Number of steps to warmup reliability loss (alternative to epoch-based)')
        parser.add_argument('--use_focal_loss', action='store_true', default=False,
                            help='Use focal loss for reliability prediction to handle class imbalance')
        parser.add_argument('--focal_alpha', type=float, default=0.25,
                            help='Alpha parameter for focal loss (weight for positive class)')
        parser.add_argument('--focal_gamma', type=float, default=2.0,
                            help='Gamma parameter for focal loss (focusing parameter)')
        
        # Reliability head specific optimization parameters
        parser.add_argument('--reliability_lr', type=float, default=1e-4, help='learning rate for reliability head (if None, uses init_lr)')
        parser.add_argument('--reliability_weight_decay', type=float, default=0.05, help='weight decay for reliability head (if None, uses weight_decay)')
        parser.add_argument('--stage1_path', type=str, default='')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=10)
        
        # Encoder selection (automatically inferred from plm_model but can be explicit)
        parser.add_argument('--encoder_type', type=str, default='auto', 
                            choices=['auto', 'esm2', 'esmc'],
                            help='Protein encoder type: auto (infer from plm_model), esm2 (HuggingFace ESM2), or esmc (official ESM-C package)')
        
        # GO term extraction parameters
        parser.add_argument('--extract_go_terms', action='store_true', default=False,
                            help='Enable GO term extraction from predictions during validation')
        parser.add_argument('--go_extraction_model', type=str, default='gpt-4o',
                            help='Model to use for GO term extraction (e.g., gpt-4o, llama3.2)')
        parser.add_argument('--go_extraction_backend', type=str, default='api', 
                            choices=['api', 'ollama'],
                            help='Backend for GO term extraction: api (Azure OpenAI) or ollama (local)')
        parser.add_argument('--ia_path', type=str, default='evals/Ontology_tools/IA.txt',
                            help='Path to Information Accretion (IA) file')
        parser.add_argument('--test_set_path', type=str, default='data/SwissProtV3/test_set.json',
                            help='Path to test set JSON file for loading ground truth GO terms')
        parser.add_argument('--mf_go_terms_path', type=str, default='evals/Ontology_tools/molecular_function_go_terms.csv',
                            help='Path to molecular function GO terms CSV file')
        
        return parent_parser



def evaluate_exact_match(predictions, targets):
    acc = 0
    for prediction, target in zip(predictions, targets):
        if prediction.strip() == target.strip():
            acc += 1
    acc = round(acc / len(predictions) * 100, 2)
    return acc