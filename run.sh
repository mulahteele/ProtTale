#!/bin/bash
# ==============================================================================
# ProtTale training / evaluation launcher.
#
# Usage:
#     bash run.sh <step>
#
# Steps:
#     stage1              Stage 1: protein-text alignment
#     stage2              Stage 2: protein-seq -> function-text generation
#     reliability_train   Train the reliability head (runs inference on
#                         validation + test data first to build the fine-tune
#                         data if the JSONs are missing)
#     eval                Stage 2 evaluation on test set
#
# All hyperparameters live in configs/default.sh. Override from the shell, e.g.
#     MAX_EPOCHS=10 LORA_R=32 bash run.sh stage2
# ==============================================================================
set -euo pipefail

mkdir -p log all_checkpoints

# Load hyperparameters.
# shellcheck disable=SC1091
source "$(dirname "$0")/configs/default.sh"

STEP="${1:-stage2}"

case "$STEP" in
# ------------------------------------------------------------------------------
# 1. Stage 1 Training (Protein-Text Alignment)
# ------------------------------------------------------------------------------
stage1)
  torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT stage1.py \
    --mode train \
    --filename "$STAGE1_FILENAME" \
    --root "$ROOT_STAGE1" \
    --devices "$DEVICES" \
    --strategy "$STRATEGY" \
    --precision "$PRECISION" \
    --num_workers $NUM_WORKERS \
    --plm_name "$PLM_MODEL" \
    --encoder_type "$ENCODER_TYPE" \
    --num_query_token $NUM_QUERY_TOKEN \
    --plm_tune "$PLM_TUNE" \
    --plm_lora_r $PLM_LORA_R \
    --plm_lora_alpha $PLM_LORA_ALPHA \
    --temperature $TEMPERATURE \
    --w_pairwise_training $W_PAIRWISE_TRAINING \
    --w_pointwise_align $W_POINTWISE_ALIGN \
    --w_pairwise_align $W_PAIRWISE_ALIGN \
    --batch_size $BATCH_SIZE_STAGE1 \
    --max_epochs $MAX_EPOCHS \
    --save_every_n_epochs $SAVE_EVERY_N_EPOCHS \
    --check_val_every_n_epoch $CHECK_VAL_EVERY_N_EPOCH \
    --init_lr $INIT_LR_STAGE1 \
    --warmup_steps $WARMUP_STEPS
  ;;

# ------------------------------------------------------------------------------
# 2. Stage 2 Training (Protein Seq -> Function Text)
#    Requires STAGE1_CKPT (converted.ckpt from Stage 1).
# ------------------------------------------------------------------------------
stage2)
  torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT stage2.py \
    --mode train \
    --stage1_path "$STAGE1_CKPT" \
    --filename "$STAGE2_FILENAME" \
    --seed $SEED \
    --root "$ROOT" \
    --devices "$DEVICES" \
    --strategy "$STRATEGY" \
    --precision "$PRECISION" \
    --num_workers $NUM_WORKERS \
    --plm_model "$PLM_MODEL" \
    --encoder_type "$ENCODER_TYPE" \
    --num_query_token $NUM_QUERY_TOKEN \
    --plm_tune "$PLM_TUNE" \
    --plm_lora_r $PLM_LORA_R \
    --plm_lora_alpha $PLM_LORA_ALPHA \
    --text_max_len $TEXT_MAX_LEN \
    --prot_max_len $PROT_MAX_LEN \
    --max_inference_len $MAX_INFERENCE_LEN \
    --llm_name "$LLM_NAME" \
    --llm_tune "$LLM_TUNE" \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --batch_size $BATCH_SIZE \
    --inference_batch_size $INFERENCE_BATCH_SIZE \
    --init_lr $INIT_LR \
    --max_epochs $MAX_EPOCHS \
    --caption_eval_epoch $CAPTION_EVAL_EPOCH \
    --max_per_cluster $MAX_PER_CLUSTER \
    --save_every_n_epochs $SAVE_EVERY_N_EPOCHS \
    --check_val_every_n_epoch $CHECK_VAL_EVERY_N_EPOCH
  ;;

# ------------------------------------------------------------------------------
# 3. Reliability Head Training
#    Phase A: run Stage 2 inference on validation + test data -> reliability_finetune{,_valid}.json
#             (skipped automatically if both files already exist).
#    Phase B: freeze everything except the reliability head and train on those files.
# ------------------------------------------------------------------------------
reliability_train)
  if [[ -f "$RELIABILITY_FINETUNE_JSON" && -f "$RELIABILITY_FINETUNE_VALID_JSON" ]]; then
    echo "[reliability_train] Fine-tune JSONs already exist — skipping inference phase."
    echo "    $RELIABILITY_FINETUNE_JSON"
    echo "    $RELIABILITY_FINETUNE_VALID_JSON"
  else
    echo "[reliability_train] Phase A: running inference on validation + test data..."
    torchrun --nproc_per_node=1 --master_port=$((MASTER_PORT+1)) stage2.py \
      --mode train \
      --inference_on_training_data \
      --reliability_label_zero \
      --inference_min_go 2 \
      --inference_sample_size 0 \
      --init_checkpoint "$STAGE2_CKPT" \
      --filename "$STAGE2_FILENAME" \
      --root "$ROOT" \
      --devices 1 \
      --precision "$PRECISION" \
      --num_workers $NUM_WORKERS \
      --plm_model "$PLM_MODEL" \
      --encoder_type "$ENCODER_TYPE" \
      --num_query_token $NUM_QUERY_TOKEN \
      --plm_tune "$PLM_TUNE" \
      --plm_lora_r $PLM_LORA_R \
      --plm_lora_alpha $PLM_LORA_ALPHA \
      --text_max_len $TEXT_MAX_LEN \
      --max_inference_len $MAX_INFERENCE_LEN \
      --llm_name "$LLM_NAME" \
      --llm_tune "$LLM_TUNE" \
      --lora_r $LORA_R \
      --lora_alpha $LORA_ALPHA
  fi

  echo "[reliability_train] Phase B: training the reliability head..."
  torchrun --nproc_per_node=$NPROC --master_port=$((MASTER_PORT+2)) stage2.py \
    --mode train \
    --train_reliability_head_only \
    --init_checkpoint "$STAGE2_CKPT" \
    --reliability_finetune_data "$RELIABILITY_FINETUNE_JSON" \
    --reliability_finetune_valid_data "$RELIABILITY_FINETUNE_VALID_JSON" \
    --test_set_path "$RELIABILITY_FINETUNE_VALID_JSON" \
    --filename "$RELIABILITY_FILENAME" \
    --root "$ROOT" \
    --devices "$DEVICES" \
    --strategy "$STRATEGY" \
    --precision "$PRECISION" \
    --num_workers $NUM_WORKERS \
    --plm_model "$PLM_MODEL" \
    --encoder_type "$ENCODER_TYPE" \
    --num_query_token $NUM_QUERY_TOKEN \
    --plm_tune "$PLM_TUNE" \
    --plm_lora_r $PLM_LORA_R \
    --plm_lora_alpha $PLM_LORA_ALPHA \
    --text_max_len $TEXT_MAX_LEN \
    --max_inference_len $MAX_INFERENCE_LEN \
    --llm_name "$LLM_NAME" \
    --llm_tune "$LLM_TUNE" \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --batch_size $BATCH_SIZE \
    --inference_batch_size $INFERENCE_BATCH_SIZE \
    --init_lr $INIT_LR \
    --max_epochs $MAX_EPOCHS \
    --save_every_n_epochs $SAVE_EVERY_N_EPOCHS \
    --check_val_every_n_epoch $CHECK_VAL_EVERY_N_EPOCH
  ;;

# ------------------------------------------------------------------------------
# 4. Stage 2 Evaluation (validation on the test set with the final checkpoint)
# ------------------------------------------------------------------------------
eval)
  torchrun --nproc_per_node=1 --master_port=$((MASTER_PORT+3)) stage2.py \
    --mode eval \
    --init_checkpoint "$RELIABILITY_CKPT" \
    --test_set_path "$ROOT/test_set.json" \
    --filename "$STAGE2_FILENAME" \
    --root "$ROOT" \
    --devices 1 \
    --precision "$PRECISION" \
    --num_workers $NUM_WORKERS \
    --plm_model "$PLM_MODEL" \
    --encoder_type "$ENCODER_TYPE" \
    --num_query_token $NUM_QUERY_TOKEN \
    --plm_tune "$PLM_TUNE" \
    --plm_lora_r $PLM_LORA_R \
    --plm_lora_alpha $PLM_LORA_ALPHA \
    --text_max_len $TEXT_MAX_LEN \
    --max_inference_len $MAX_INFERENCE_LEN \
    --llm_name "$LLM_NAME" \
    --llm_tune "$LLM_TUNE" \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --batch_size $BATCH_SIZE \
    --inference_batch_size $INFERENCE_BATCH_SIZE \
    --init_lr $INIT_LR \
    --max_epochs 1 \
    --caption_eval_epoch $CAPTION_EVAL_EPOCH \
    --save_every_n_epochs $SAVE_EVERY_N_EPOCHS
  ;;

*)
  echo "Unknown step: $STEP"
  echo "Usage: bash run.sh {stage1|stage2|reliability_train|eval}"
  exit 1
  ;;
esac
