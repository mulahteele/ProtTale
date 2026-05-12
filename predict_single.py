"""
Single-sequence inference for ProtTale.

Usage:
    python predict_single.py --seq MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
    python predict_single.py --fasta input.fa
    python predict_single.py --seq_file seqs.txt   # one sequence per line

Outputs (per sequence) to stdout and optionally to --out_json:
    - prediction: generated Swiss-Prot function text
    - reliability: predicted class in {0.0, 1.0} (binary: 1.0 = reliable)
    - reliability_pos_prob: probability of the positive class (r = 1.0)
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import json
import torch
from argparse import Namespace

from model.blip2_stage2 import Blip2Stage2
from data_provider.stage2_dm import InferenceCollater


def build_args(ckpt_path):
    """Hyperparameters must match those used to train the checkpoint (see train.sh)."""
    return Namespace(
        # Encoder (PLM)
        plm_model="esmc_300m",
        encoder_type="auto",
        num_query_token=4,
        plm_tune="lora",
        plm_lora_r=4,
        plm_lora_alpha=8,
        plm_lora_dropout=0.1,
        # Q-former / BERT
        bert_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        cross_attention_freq=2,
        # LLM
        llm_name="facebook/galactica-1.3b",
        llm_tune="lora",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        peft_dir="",
        peft_config="",
        # Generation
        num_beams=3,
        do_sample=False,
        max_inference_len=128,
        min_inference_len=1,
        # Lengths
        text_max_len=128,
        prot_max_len=1024,
        # Misc training flags (needed by Blip2Stage2.__init__)
        caption_eval_epoch=1,
        inference_on_training_data=False,
        train_reliability_head_only=False,
        report_go_wang_on_test=False,
        report_go_wang_on_val=False,
        save_predictions=False,
        ia_path="evals/tools/IA.txt",
        go_files_tsv_path="evals/tools/go_files.tsv",
        test_set_path="",
        valid_set_path="",
        root="data/SwissProtV3",
        enbale_gradient_checkpointing=False,
        init_checkpoint=ckpt_path,
        stage1_path="",
        stage2_path="",
        # Optimizer knobs referenced in __init__ via save_hyperparameters
        init_lr=1e-4,
        min_lr=1e-5,
        warmup_lr=1e-6,
        warmup_steps=0,
        lr_decay_rate=0.9,
        scheduler="None",
        weight_decay=0.05,
        reliability_lr=1e-4,
        max_epochs=1,
        filename="predict_single",
        seed=42,
        reliability_binary=True,
    )


def read_sequences(args):
    seqs = []
    if args.seq:
        seqs.append(args.seq.strip().upper())
    if args.seq_file:
        with open(args.seq_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    seqs.append(s.upper())
    if args.fasta:
        with open(args.fasta, "r", encoding="utf-8") as f:
            current = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current:
                        seqs.append("".join(current).upper())
                        current = []
                else:
                    current.append(line)
            if current:
                seqs.append("".join(current).upper())
    if not seqs:
        raise ValueError("No sequences provided. Use --seq, --seq_file, or --fasta.")
    return seqs


@torch.no_grad()
def run(args):
    model_args = build_args(args.ckpt)
    model = Blip2Stage2(model_args)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_sd and model_sd[k].shape == v.shape}
    skipped = [k for k in state_dict if k not in filtered]
    if skipped:
        print(f"Skipped {len(skipped)} keys: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    model.load_state_dict(filtered, strict=False)
    print(f"Loaded checkpoint: {args.ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    blip2 = model.blip2
    collater = InferenceCollater(
        tokenizer=blip2.llm_tokenizer,
        prot_tokenizer=blip2.plm_tokenizer,
        text_max_len=model_args.text_max_len,
        prot_max_len=model_args.prot_max_len,
    )

    seqs = read_sequences(args)
    prompt = "Swiss-Prot description: "
    results = []

    for start in range(0, len(seqs), args.batch_size):
        chunk = seqs[start:start + args.batch_size]
        batch = [(s, prompt, "", 0.0, [], i) for i, s in enumerate(chunk)]
        prot_tokens, prompt_tokens, r_tensor, _ = collater(batch)

        prot_tokens = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in prot_tokens.items()
        }
        prompt_tokens = prompt_tokens.to(device)
        r_tensor = r_tensor.to(device)

        samples = {"prot_batch": prot_tokens, "prompt_batch": prompt_tokens, "reliability": r_tensor}
        pred_texts, r_pred, _conf, _emb, r_probs = blip2.generate(
            samples,
            do_sample=model_args.do_sample,
            num_beams=model_args.num_beams,
            max_length=model_args.max_inference_len,
            min_length=model_args.min_inference_len,
        )

        r_pred_list = r_pred.cpu().tolist() if torch.is_tensor(r_pred) else list(r_pred)
        r_probs_list = r_probs.cpu().tolist() if torch.is_tensor(r_probs) else list(r_probs)

        for i, s in enumerate(chunk):
            row = {
                "sequence": s,
                "prediction": pred_texts[i],
                "reliability": round(float(r_pred_list[i]), 4),
                "reliability_pos_prob": round(float(r_probs_list[i][1]), 4),
            }
            results.append(row)

    for r in results:
        print("=" * 60)
        print(f"sequence (len={len(r['sequence'])}): {r['sequence'][:80]}{'...' if len(r['sequence']) > 80 else ''}")
        print(f"prediction  : {r['prediction']}")
        print(f"reliability        : {r['reliability']}")
        print(f"reliability_pos_prob: {r['reliability_pos_prob']}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")
        print(f"\nSaved {len(results)} results to {args.out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default="all_checkpoints/ProtTale_binary_reliability_ft/best_val_reliability_pos_f1.ckpt",
                    help="Path to checkpoint (.ckpt). Must be a binary reliability head checkpoint.")
    ap.add_argument("--seq", type=str, default="", help="Single protein sequence (amino acids).")
    ap.add_argument("--seq_file", type=str, default="", help="Text file with one sequence per line.")
    ap.add_argument("--fasta", type=str, default="", help="FASTA file.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_json", type=str, default="", help="Optional JSONL output path.")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
