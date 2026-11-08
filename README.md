# PPQ Tokenizer

A progressive product quantization (PPQ) tokenizer for generative recommendation. The repository includes an example that connects PPQ-generated semantic IDs to a [FLASH](https://github.com/KevinC2015/Flash) recommendation backbone, trained without semantic alignment.

## Quick Start

The current runnable example covers **Beauty**. Python 3.10, a CUDA-capable NVIDIA GPU, and the separately supplied data and initialization files are required. The repository does not download datasets or text embeddings automatically. See [Environment and usage](docs/SETUP.md) for the tested versions, installation commands, and the expected data paths.

```bash
# Check the tokenizer and recommender with a short run.
CUDA_VISIBLE_DEVICES=0 python run.py --run-dir outputs/smoke --smoke

# Train PPQ and then train and evaluate FLASH on Beauty.
CUDA_VISIBLE_DEVICES=0 python run.py --run-dir outputs/beauty

# Alternatively, use a supplied SID and train FLASH from scratch.
CUDA_VISIBLE_DEVICES=0 python run.py --run-dir outputs/beauty-flash --sid-from reference/ppq_codes.npy
```

The smoke run performs a small number of updates and evaluates a subset of examples; it is not a performance result. Each output directory must be new. Configuration details are in `configs/ppq.json` and `configs/flash.json`; see [Code guide](docs/CODE_GUIDE.md).

## Key Hyperparameters

| Parameter | Example value | Description |
| --- | ---: | --- |
| `M` | 32 | Number of SID positions per item |
| `G` | 16 | Number of cumulative PPQ reconstruction groups for Beauty |
| `K` | 256 | Number of entries per codebook |
| PPQ reconstruction | mean over G | Equal weight for each cumulative prefix and full reconstruction |
| Commitment weight | 0.25 | Weight of the commitment loss |
| PPQ training | 200 epochs | Batch size 1024, learning rate 0.001, EMA decay 0.99 |
| `align_weight` | 0 | Semantic alignment is disabled throughout FLASH training |
| FLASH microbatch | 32 | Gradient accumulation is weighted by the number of sequences |
| FLASH warmup | 10000 updates | Cosine schedule with a fixed training budget |

### Dataset configurations

The following settings identify the historical four-dataset experiments reported below. **Only the Beauty configuration is currently runnable through this repository's entry point**; the other three configurations and their data are not included as runnable examples. `M` is the SID length, `G` is the PPQ group count, and `d` is the FLASH hidden dimension.

| Dataset | M | G | K | d | FLASH batch | FLASH learning rate | FLASH updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Beauty | 32 | 16 | 256 | 448 | 256 | 0.01 | 10100 |
| Toys and Games | 32 | 4 | 256 | 448 | 128 | 0.003 | 17400 |
| Sports and Outdoors | 32 | 8 | 256 | 896 | 256 | 0.003 | 14600 |
| CDs and Vinyl | 32 | 8 | 256 | 448 | 256 | 0.001 | 223800 |

## Example Results

Full-catalogue real-item test metrics for **PPQ + FLASH without alignment**, one run per dataset with a fixed tokenizer. Model selection uses NDCG@10, then Recall@10, then the earlier update. Configuration choices were informed by exploratory test results, so these are not independent confirmation results. The scores are for the base recommender only.

### Validation-selected checkpoint

| Dataset | Update | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Beauty | 5500 | 0.04106 | 0.04998 | 0.05871 | 0.08621 |
| Toys and Games | 8500 | 0.04798 | 0.05722 | 0.06800 | 0.09685 |
| Sports and Outdoors | 5500 | 0.02355 | 0.02881 | 0.03388 | 0.05014 |
| CDs and Vinyl | 212610 | 0.03674 | 0.04438 | 0.05431 | 0.07800 |

### Test-selected checkpoint (exploratory)

| Dataset | Update | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Beauty | 4500 | 0.04201 | 0.05042 | 0.06041 | 0.08639 |
| Toys and Games | 8000 | 0.04785 | 0.05740 | 0.06759 | 0.09736 |
| Sports and Outdoors | 4500 | 0.02399 | 0.02981 | 0.03514 | 0.05329 |
| CDs and Vinyl | 167850 | 0.03678 | 0.04461 | 0.05441 | 0.07873 |

The test-selected checkpoint is an exploratory result: test labels were used to choose the checkpoint, and all metrics are reported at the same selected update. The Beauty, Toys, and Sports runs evaluated every 500 updates and at the endpoint; CDs used a predeclared 5595-update interval and endpoint. All datasets also evaluated the initial checkpoint. No independent `final_test` was used.

## Attribution

The recommendation backbone is based on [FLASH](https://github.com/KevinC2015/Flash). Please refer to that project for its original implementation, paper, and license. Additional third-party and data notices are in [THIRD_PARTY.md](THIRD_PARTY.md).

```bibtex
@misc{flash_repository,
  title = {FLASH: Fast LLM-Aligned SimHash for Generative Recommendation},
  howpublished = {\url{https://github.com/KevinC2015/Flash}}
}
```
