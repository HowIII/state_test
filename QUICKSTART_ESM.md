# ESM Perturbation Encoder - Quick Start Guide

Train a State Transition model using ESM2 protein embeddings for genetic perturbations in just 3 steps!

## Prerequisites

- State framework installed and working
- Genetic perturbation datasets in h5 format
- Access to ESM2 embeddings (included by default)

## Step 1: Configure Your Datasets

Edit `esm_genetic_example.toml` with your dataset paths:

```toml
[datasets]
my_data = "/path/to/my/genetic/perturbation/data"

[training]
my_data = "train"

[fewshot."my_data.MyCellType"]
val = ["GENE1", "GENE2"]   # Genes for validation
test = ["GENE3", "GENE4"]  # Genes for testing
```

## Step 2: Run Training

**Simplest command:**
```bash
python train_esm_perturbation.py --config esm_genetic_example.toml
```

**With custom settings:**
```bash
python train_esm_perturbation.py \
    --config esm_genetic_example.toml \
    --output_dir ./my_output \
    --batch_size 16 \
    --max_steps 50000
```

**With W&B logging:**
```bash
python train_esm_perturbation.py \
    --config esm_genetic_example.toml \
    --use_wandb \
    --wandb_project my-project
```

## Step 3: Monitor Training

Training outputs will be saved to the output directory:
```
output_dir/
├── checkpoints/           # Model checkpoints
│   ├── best.ckpt         # Best model (lowest val loss)
│   └── step=N.ckpt       # Periodic checkpoints
├── logs/                  # Training logs
└── config.yaml           # Full configuration used
```

Monitor progress:
- **CSV logs**: `output_dir/logs/metrics.csv`
- **W&B**: Dashboard at wandb.ai (if enabled)
- **Console**: Real-time loss and metrics

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to dataset config | `esm_genetic_example.toml` |
| `--output_dir` | Where to save results | `$HOME/state_esm_output` |
| `--batch_size` | Batch size | 8 |
| `--max_steps` | Max training steps | 100000 |
| `--lr` | Learning rate | 3e-4 |
| `--use_wandb` | Enable W&B logging | False |
| `--embed_key` | Embedding key from data | `X_uce` |

## What Makes This Different?

Instead of one-hot encoding genes (sparse, no semantic info):
```
"STAT1" → [0, 0, 0, 1, 0, 0, ..., 0]  (10,000+ dims)
```

ESM uses protein embeddings (dense, semantic):
```
"STAT1" → ESM2 embedding → [0.23, -0.45, ..., 1.2]  (5,120 dims)
```

**Benefits:**
- Better generalization to unseen genes
- Leverages protein sequence information
- Enables zero-shot prediction

## Need Help?

- **Full documentation**: See `ESM_PERTURBATION_README.md`
- **Troubleshooting**: Check README troubleshooting section
- **Examples**: See example configurations in repo

## Quick Test

Test if everything is set up correctly:

```bash
# Check if script is executable
python train_esm_perturbation.py --help

# Verify model registration
python -c "from state.tx.models import ESMStateTransitionModel; print('ESM model loaded!')"

# Test ESM embeddings loading
python -c "import torch; emb = torch.load('/large_storage/ctc/ML/data/cell/misc/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt', weights_only=False); print(f'Loaded {len(emb)} genes')"
```

All tests pass? You're ready to train! 🚀
