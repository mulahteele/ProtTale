#!/bin/bash
# ==============================================================================
# Convert a DeepSpeed / Lightning checkpoint into the flat .ckpt that Stage 2
# and predict_single.py can load directly.
#
# Usage:
#     bash convert.sh <input_ckpt_or_dir> <output_ckpt>
#
# Example:
#     bash convert.sh all_checkpoints/stage1_ckpt/epoch=09.ckpt \
#                     all_checkpoints/stage1_ckpt/converted.ckpt
# ==============================================================================
set -euo pipefail

INPUT="${1:-}"
OUTPUT="${2:-}"

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
  echo "Usage: bash convert.sh <input_ckpt_or_dir> <output_ckpt>"
  exit 1
fi

python model/convert.py --input "$INPUT" --output "$OUTPUT"
