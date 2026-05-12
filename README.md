<div align="center">

<img src="assets/prottale_logo.png" alt="ProtTale" width="700"/>

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=plastic&logo=arxiv&logoColor=white)](#)
[![Model](https://img.shields.io/badge/🤗_Model-HuggingFace-FFD21E?style=plastic)](https://huggingface.co/Mulah/ProtTale)
[![Demo](https://img.shields.io/badge/🤗_Demo-Spaces-FFD21E?style=plastic)](https://huggingface.co/spaces/Mulah/ProtTale-demo)

</div>

ProtTale is a multi-stage framework that maps a protein amino-acid sequence to a Swiss-Prot-style function description, together with a reliability score for the generated text. The pipeline has three training stages:

1. **Stage 1** — Protein–text alignment between an ESM-C encoder and a Q-Former.
2. **Stage 2** — Protein-sequence → function-text generation (Q-Former + LLM, LoRA-tuned).
3. **Reliability training** — Freeze everything except the reliability head and train it on Stage 2 predictions over validation / test data.

Once training is done, `predict_single.py` runs inference on one or many sequences.

---

## 1. Setup

### 1.1 Environment

```bash
# Create the environment
conda env create -n ProtTale -f environment.yml

# Activate it
conda activate ProtTale

# Two packages ship with unresolvable dependency pins and must be installed separately:
pip install salesforce-lavis==1.0.2 --no-deps
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install opendelta==0.3.2
```

### 1.2 Data

Expected layout:

```
data/
├── SwissProtV3/               # Stage 2 data
│   ├── train_set.json
│   ├── valid_set.json
│   ├── test_set.json
│   └── unseen_set.json
└── SwissProtV3_stage1/        # Stage 1 data (alignment)
    ├── train_set.json
    ├── valid_set.json
    └── test_set.json
```

These JSON files are not committed (they live behind `.gitignore`). Put your splits under these paths, or override `ROOT` and `ROOT_STAGE1` (see [configs/default.sh](configs/default.sh)).

### 1.3 API keys (optional, only needed by the GO/EC extraction utilities)

The helpers in [evals/tools/extraction.py](evals/tools/extraction.py) call Anthropic and Azure OpenAI. Export these before running evaluation tooling:

```bash
export ANTHROPIC_API_KEY=...
export AZURE_OPENAI_ENDPOINT=...
export AZURE_OPENAI_KEY=...
export AZURE_OPENAI_DEPLOYMENT=o4-mini                  # optional
export AZURE_OPENAI_API_VERSION=2024-12-01-preview      # optional
```

---

## 2. Configuration

All hyperparameters live in [configs/default.sh](configs/default.sh) — one file, one source of truth. [run.sh](run.sh) sources it automatically. Every variable can be overridden from the shell:

```bash
MAX_EPOCHS=10 LORA_R=32 bash run.sh stage2
```

The most commonly tuned knobs:

| Variable | Meaning | Default |
| --- | --- | --- |
| `DEVICES` / `NPROC` | Visible GPUs / processes | `"0"` / `1` |
| `PLM_MODEL` | Protein encoder | `esmc_300m` |
| `LLM_NAME` | Generation model | `facebook/galactica-1.3b` |
| `LORA_R` / `LORA_ALPHA` | LLM LoRA rank / alpha | `16` / `32` |
| `PLM_LORA_R` / `PLM_LORA_ALPHA` | PLM LoRA rank / alpha | `4` / `8` |
| `BATCH_SIZE` | Stage 2 training batch size | `8` |
| `PROT_MAX_LEN` | Max protein length (quadratic memory) | `1024` |
| `MAX_EPOCHS` | Epochs for the current step | `1` |

---

## 3. Training pipeline

All four steps are driven by [run.sh](run.sh):

```bash
bash run.sh <step>
```

### Step 1 — Stage 1: protein-text alignment

```bash
bash run.sh stage1
```

Trains the Q-Former against ESM-C with three alignment losses (`W_PAIRWISE_TRAINING`, `W_POINTWISE_ALIGN`, `W_PAIRWISE_ALIGN`). With DeepSpeed, a flat `converted.ckpt` is written automatically alongside the DeepSpeed checkpoint. If not, convert manually:

```bash
bash convert.sh all_checkpoints/stage1_ckpt/epoch=09.ckpt \
                all_checkpoints/stage1_ckpt/converted.ckpt
```

The path `all_checkpoints/${STAGE1_FILENAME}/converted.ckpt` is what Stage 2 loads by default (`STAGE1_CKPT` in [configs/default.sh](configs/default.sh)).

### Step 2 — Stage 2: function-text generation

```bash
bash run.sh stage2
```

Loads `STAGE1_CKPT`, then fine-tunes the Q-Former + LLM (LoRA) to generate Swiss-Prot-style descriptions. Output checkpoint: `all_checkpoints/${STAGE2_FILENAME}/converted.ckpt`.

### Step 3 — Reliability training

The Stage 2 model is run on the **validation and test data** to produce `reliability_finetune.json` / `reliability_finetune_valid.json`; those files become the supervision for the reliability head. The launcher takes care of both phases in a single command:

```bash
bash run.sh reliability_train
```

* **Phase A** — if the two fine-tune JSONs are missing, it runs Stage 2 inference on validation + test data to build them.
* **Phase B** — freezes every parameter except the reliability head and trains it on those JSONs.

If the JSONs already exist, phase A is skipped automatically; delete them to force re-inference.

Output checkpoint: `all_checkpoints/${RELIABILITY_FILENAME}/checkpoint.ckpt`. This is the final checkpoint used for downstream inference.

### Step 4 — (Optional) Stage 2 evaluation

```bash
bash run.sh eval
```

Runs validation on `test_set.json` with the reliability-trained checkpoint. Pass `--report_go_wang_on_test` or `--report_go_wang_on_val` through `stage2.py` (e.g. by editing the eval block in [run.sh](run.sh)) to report GO Wang / IA / Jaccard scores.

---

## 4. Inference

Single-sequence / batch inference uses [predict_single.py](predict_single.py). It loads the final reliability-trained checkpoint and outputs both the generated function text and a reliability prediction.

### Single sequence

```bash
python predict_single.py \
  --ckpt all_checkpoints/ProtTale_binary_reliability_ft/checkpoint.ckpt \
  --seq MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
```

### Multiple sequences

From a plain text file (one sequence per line):

```bash
python predict_single.py --ckpt <path> --seq_file seqs.txt
```

From a FASTA file:

```bash
python predict_single.py --ckpt <path> --fasta input.fa
```

### Saving results

```bash
python predict_single.py --ckpt <path> --fasta input.fa --out_json results.jsonl
```

Each line in `results.jsonl`:

```json
{
  "sequence": "MKTVRQER...",
  "prediction": "Catalyzes ...",
  "reliability": 1.0,
  "reliability_pos_prob": 0.9123
}
```

`reliability` is the predicted binary class in `{0.0, 1.0}` (`1.0` = reliable, `0.0` = unreliable), and `reliability_pos_prob` is the model's probability for the positive (reliable) class.

The generation hyperparameters (`num_beams`, `max_inference_len`, LoRA sizes, etc.) must match the checkpoint; see `build_args()` in [predict_single.py](predict_single.py) and keep them in sync with [configs/default.sh](configs/default.sh).

---

## 5. Repository layout

```
ProtTale_code/
├── configs/
│   └── default.sh             # All hyperparameters (sourced by run.sh)
├── data_provider/             # Dataset / collate code
├── model/                     # BLIP-2-style stage1 / stage2 modules
├── evals/                     # Evaluation metrics and GO / EC extractors
├── stage1.py                  # Stage 1 entry point
├── stage2.py                  # Stage 2 / reliability entry point
├── predict_single.py          # Single / batch inference
├── run.sh                   # Unified launcher (stage1|stage2|reliability_train|eval)
├── convert.sh                 # Checkpoint conversion wrapper
└── environment.yml            # Conda env spec
```

---
