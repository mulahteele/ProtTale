# ==============================================================================
# ProtTale hyperparameters. Sourced by run.sh.
#
# All values can be overridden from the calling shell, e.g.
#     MAX_EPOCHS=5 LORA_R=32 bash run.sh stage2
# ==============================================================================

# ---------- Data --------------------------------------------------------------
ROOT="${ROOT:-data/SwissProtV3}"
ROOT_STAGE1="${ROOT_STAGE1:-data/SwissProtV3_stage1}"

# ---------- Hardware / distributed --------------------------------------------
# DEVICES must match the number of GPUs (1 GPU -> "0" NPROC=1, 2 GPUs -> "0,1" NPROC=2)
DEVICES="${DEVICES:-0}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29502}"
STRATEGY="${STRATEGY:-deepspeed}"
PRECISION="${PRECISION:-bf16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# ---------- Protein encoder (PLM) --------------------------------------------
PLM_MODEL="${PLM_MODEL:-esmc_300m}"
ENCODER_TYPE="${ENCODER_TYPE:-auto}"
NUM_QUERY_TOKEN="${NUM_QUERY_TOKEN:-4}"
PLM_TUNE="${PLM_TUNE:-lora}"
PLM_LORA_R="${PLM_LORA_R:-4}"
PLM_LORA_ALPHA="${PLM_LORA_ALPHA:-8}"

# ---------- LLM ---------------------------------------------------------------
LLM_NAME="${LLM_NAME:-facebook/galactica-1.3b}"
LLM_TUNE="${LLM_TUNE:-lora}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# ---------- Sequence / text lengths ------------------------------------------
# PROT_MAX_LEN has quadratic memory effect on the ESM-C attention.
TEXT_MAX_LEN="${TEXT_MAX_LEN:-128}"
PROT_MAX_LEN="${PROT_MAX_LEN:-1024}"
MAX_INFERENCE_LEN="${MAX_INFERENCE_LEN:-128}"

# ---------- Stage 2 (generation) ----------------------------------------------
BATCH_SIZE="${BATCH_SIZE:-8}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
INIT_LR="${INIT_LR:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
CAPTION_EVAL_EPOCH="${CAPTION_EVAL_EPOCH:-1}"
# Max samples per cluster per epoch (when train_set has a cluster_id column).
MAX_PER_CLUSTER="${MAX_PER_CLUSTER:-3}"

# ---------- Stage 1 (alignment) -----------------------------------------------
W_PAIRWISE_TRAINING="${W_PAIRWISE_TRAINING:-1.0}"
W_POINTWISE_ALIGN="${W_POINTWISE_ALIGN:-0.1}"
W_PAIRWISE_ALIGN="${W_PAIRWISE_ALIGN:-0.1}"
TEMPERATURE="${TEMPERATURE:-0.5}"
INIT_LR_STAGE1="${INIT_LR_STAGE1:-5e-4}"
BATCH_SIZE_STAGE1="${BATCH_SIZE_STAGE1:-16}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"

# ---------- Checkpoint / output names ----------------------------------------
STAGE1_FILENAME="${STAGE1_FILENAME:-stage1_ckpt}"
STAGE2_FILENAME="${STAGE2_FILENAME:-ProtTale_stage2}"
RELIABILITY_FILENAME="${RELIABILITY_FILENAME:-ProtTale_reliability_ft}"
SEED="${SEED:-42}"

# Derived paths.
STAGE1_CKPT="${STAGE1_CKPT:-all_checkpoints/${STAGE1_FILENAME}/converted.ckpt}"
STAGE2_CKPT="${STAGE2_CKPT:-all_checkpoints/${STAGE2_FILENAME}/converted.ckpt}"
RELIABILITY_CKPT="${RELIABILITY_CKPT:-all_checkpoints/${RELIABILITY_FILENAME}/best_val_reliability_class1_f1.ckpt}"

# Reliability-head fine-tune data (produced inside step 3).
RELIABILITY_FINETUNE_JSON="${RELIABILITY_FINETUNE_JSON:-all_checkpoints/${STAGE2_FILENAME}/reliability_finetune.json}"
RELIABILITY_FINETUNE_VALID_JSON="${RELIABILITY_FINETUNE_VALID_JSON:-all_checkpoints/${STAGE2_FILENAME}/reliability_finetune_valid.json}"

# Misc env.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Uncomment to reduce CUDA fragmentation-related OOMs:
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
