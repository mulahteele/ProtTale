# ProtTale

**Protein function generation with Pairwise training.**

ProtTale bridges the gap between protein language models (like ESMC) and scientific LLMs (like Galactica) to generate high-quality functional descriptions from amino acid sequences.

---

## Resources
* **Model Checkpoints:** [Link to OneDrive](#)
* **Testing Data:** [Link to OneDrive](#)


#### Sample testing data format for unknown proteins (`test_set.json`):

```bash
[
  [
    "MMRGFKQRLIKKTTGSSSSSSSKKKDKEKEKEKSSTTSSTSKKPASASSSSHGTTHSSASSTGSKSTTEKGKQSGSVPSQGKHHSSSTSKTKTATTPSSSSSSSRSSSVSRSGSSSTKKTSSRKGQEQSKQSQQPSQSQKQGSSSSSAAIMNPTPVLTVTKDDKSTSGEDHAHPTLLGAVSAVPSSPISNASGTAVSSDVENGNSNNNNMNINTSNTQDANHASSQSIDIPRSSHSFERLPTPTKLNPDTDLELIKTPQRHSSSRFEPSRYTPLTKLPNFNEVSPEERIPLFIAKVDQCNTMFDFNDPSFDIQGKEIKRSTLDELIEFLVTNRFTYTNEMYAHVVNMFKINLFRPIPPPVNPVGDIYDPDEDEPVNELAWPHMQAVYEFFLRFVESPDFNHQIAKQYIDQDFILKLLELFDSEDIRERDCLKTTLHRIYGKFLSLRSFIRRSMNNIFLQFIYETEKFNGVAELLEILGSIINGFALPLKEEHKVFLVRILIPLHKVRCLSLYHPQLAYCIVQFLEKDPLLTEEVVMGLLRYWPKINSTKEIMFLNEIEDIFEVIEPLEFIKVEVPLFVQLAKCISSPHFQVAEKVLSYWNNEYFLNLCIENAEVILPIIFPALYELTSQLELDTANGEDSISDPYMLVEQAINSGSWNRAIHAMAFKALKIFLETNPVLYENCNALYLSSVKETQQRKVQREENWSKLEEYVKNLRINNDKDQYTIKNPELRNSFNTASENNTLNEENENDCDSEIQ",
    "unknown function.",
    -1.0,
    ["GO:0000000"]
  ]
]
```



---

## Installation

Follow these steps to set up the environment and install necessary dependencies:

```bash
# 1. Create the environment
conda env create -n ProtTale -f ProtTale_environment.yml

# 2. Activate the environment
conda activate ProtTale

# 3. Install packages with specific dependency requirements
pip install salesforce-lavis==1.0.2 --no-deps
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install opendelta==0.3.2
```

##  Run Inference

To start the protein function generation process, use `torchrun`. Ensure you update the `--root` and `--init_checkpoint` arguments to point to your local data and model paths.

```bash
torchrun --nproc_per_node=1 --master_port=29505 stage2.py \
  --devices 1 \
  --mode eval \
  --init_checkpoint [PATH_TO_CHECKPOINT] \   # Required: pretrained model checkpoint
  --root [PATH_TO_TEST_DATA_ROOT] \           # Required: directory containing test JSON (e.g., ./data/SwissProtV3)
  --filename stage2_func_generation \
  --plm_model esmc_300m \
  --encoder_type auto \
  --num_query_token 2 \
  --plm_tune lora \
  --plm_lora_r 16 \
  --plm_lora_alpha 16 \
  --text_max_len 256 \
  --max_inference_len 256 \
  --llm_name facebook/galactica-6.7b \
  --llm_tune lora \
  --lora_r 16 \
  --lora_alpha 16 \
  --batch_size 16 \
  --init_lr 1e-4 \
  --inference_batch_size 4 \
  --precision 'bf16-mixed' \
  --num_workers 8 \
  --max_epochs 1 \
  --caption_eval_epoch 1 \
  --save_every_n_epochs 1 \
  --head generation
```



