# Environment and usage

This project provides a progressive product quantization tokenizer and an example of using its semantic IDs with a FLASH recommendation backbone. The example trains FLASH without semantic alignment.

## Environment

The tested environment is Linux x86-64, Python 3.10.18, PyTorch 2.6.0 with CUDA 12.4, and an NVIDIA L40 GPU. The remaining pinned dependencies are listed in `requirements.txt`.

```bash
conda create -n ppq-tokenizer python=3.10.18 -y
conda activate ppq-tokenizer
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A CUDA-capable NVIDIA GPU and a compatible driver are needed. The existing environment was tested; creating a completely clean environment with the commands above has not yet been independently tested.

## Input files

Data and generated outputs are not included in Git. Before running, obtain the separately shared input package and place the files under these paths:

```text
data/Beauty/{train.pt,val.pt,test.pt,fit_item_rows.npy,manifest.json}
data/initialization/{codes.npy,initialization.npz}
cache/AmazonReviews2014/Beauty/processed/{id_mapping.json,all_item_seqs.json,qwen3.7-text-embedding.sent_emb}
reference/{ppq_codes.npy,original_selections.json,val_4500.npz,test_4500.npz,val_5500.npz,test_5500.npz}
```

The input files are ordered and matched: preserve their item IDs, shapes and hashes. The example configuration expects the Beauty dataset. Obtain and distribute data only in accordance with the data provider's terms.

## Run

Run from the repository root. Each output directory must not already exist.

```bash
# Small end-to-end check, not a performance measurement.
CUDA_VISIBLE_DEVICES=0 python run.py --run-dir outputs/smoke --smoke

# Train the tokenizer and recommender, then evaluate the full validation/test splits.
CUDA_VISIBLE_DEVICES=0 python run.py --run-dir outputs/full

# Reuse a previously trained SID and train the recommender from scratch.
CUDA_VISIBLE_DEVICES=0 python run.py --run-dir outputs/flash-replay --sid-from reference/ppq_codes.npy
```

`configs/ppq.json` and `configs/flash.json` document the example configuration. The entry point resolves data and output paths at runtime. Evaluation saves predictions and metrics at fixed training nodes. Selecting a model using test results is exploratory and is not an independent test of generalization.
