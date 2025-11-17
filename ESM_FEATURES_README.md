# Training with ESM2 Perturbation Features (Colab Notebook Approach)

This guide follows the approach from the [Virtual Cell Challenge Colab notebook](https://colab.research.google.com/drive/1QKOtYP7bMpdgDJEipDxaJqOchv7oQ-_l).

## Quick Start (2 Steps!)

### Step 1: Create ESM Perturbation Features

```bash
python create_esm_pert_features.py \
    --from-toml my_config.toml \
    --output ESM2_pert_features.pt
```

### Step 2: Train with ESM Features

```bash
python train_with_esm_features.py \
    --config my_config.toml \
    --pert-features ESM2_pert_features.pt \
    --output-dir ./output
```

That's it! No model modifications needed.

## How It Works

### Traditional One-Hot Approach
```
Gene Name → One-hot vector → Model
   "STAT1" → [0,0,1,0,...,0]  (sparse, 10k+ dims)
```

### ESM Features Approach (This Method)
```
Step 1 (Pre-processing):
Gene Name → ESM2 Lookup → Save to file
   "STAT1" → [0.23, -0.45, ..., 1.2]  → ESM2_pert_features.pt

Step 2 (Training):
Load ESM2_pert_features.pt → Data Module → Model
                                ↓
                    Uses ESM embeddings as pert_emb
                    (No model changes needed!)
```

## Detailed Usage

### Creating Perturbation Features File

The `create_esm_pert_features.py` script creates a dictionary mapping gene names to their ESM2 embeddings.

**Option 1: From TOML config (recommended)**
```bash
python create_esm_pert_features.py \
    --from-toml config.toml \
    --output ESM2_pert_features.pt
```

This automatically extracts all unique gene names from your datasets.

**Option 2: From gene list file**
```bash
# Create a text file with gene names (one per line)
echo "STAT1" > genes.txt
echo "IRF1" >> genes.txt
echo "TP53" >> genes.txt

python create_esm_pert_features.py \
    --gene-list genes.txt \
    --output ESM2_pert_features.pt
```

**Option 3: Custom ESM embeddings**
```bash
python create_esm_pert_features.py \
    --from-toml config.toml \
    --esm-embeddings /path/to/custom/esm.pt \
    --output ESM2_pert_features.pt
```

### Training with ESM Features

**Basic usage:**
```bash
python train_with_esm_features.py \
    --config config.toml \
    --pert-features ESM2_pert_features.pt
```

**Virtual Cell Challenge example (matching Colab):**
```bash
python train_with_esm_features.py \
    --config competition_support_set/starter.toml \
    --pert-features competition_support_set/ESM2_pert_features.pt \
    --pert-col target_gene \
    --control-pert non-targeting \
    --model state_sm \
    --max-steps 40000 \
    --ckpt-every-n-steps 20000 \
    --output-dir competition \
    --name first_run
```

**With W&B logging:**
```bash
python train_with_esm_features.py \
    --config config.toml \
    --pert-features ESM2_pert_features.pt \
    --use-wandb \
    --wandb-project my-project \
    --wandb-entity my-entity \
    --wandb-tags esm baseline
```

**Advanced customization:**
```bash
python train_with_esm_features.py \
    --config config.toml \
    --pert-features ESM2_pert_features.pt \
    --model state_lg \
    --batch-size 16 \
    --max-steps 100000 \
    training.lr=1e-4 \
    model.kwargs.hidden_dim=1024
```

### Direct CLI Usage

You can also use the State CLI directly:

```bash
state tx train \
    data.kwargs.toml_config_path="config.toml" \
    data.kwargs.perturbation_features_file="ESM2_pert_features.pt" \
    data.kwargs.pert_col="gene" \
    data.kwargs.control_pert="non-targeting" \
    model=state_sm \
    training.max_steps=40000 \
    output_dir="./output" \
    name="my_run"
```

## File Formats

### ESM2_pert_features.pt Format

The perturbation features file is a PyTorch dictionary saved with `torch.save()`:

```python
{
    "STAT1": torch.Tensor([...]),  # shape: [5120]
    "IRF1": torch.Tensor([...]),   # shape: [5120]
    "TP53": torch.Tensor([...]),   # shape: [5120]
    # ... more genes
    "non-targeting": torch.Tensor([0, 0, ..., 0]),  # zero embedding for control
}
```

**Load and inspect:**
```python
import torch

pert_features = torch.load("ESM2_pert_features.pt")
print(f"Number of genes: {len(pert_features)}")
print(f"Embedding dim: {next(iter(pert_features.values())).shape}")
print(f"Genes: {list(pert_features.keys())[:10]}")
```

### TOML Config Format

```toml
[datasets]
dataset1 = "/path/to/dataset1"
dataset2 = "/path/to/dataset2"

[training]
dataset1 = "train"
dataset2 = "train"

[fewshot."dataset1.CellType"]
val = ["GENE1", "GENE2"]
test = ["GENE3", "GENE4"]
```

## Comparison: Two Approaches for ESM Perturbations

This repository provides **two approaches** for using ESM embeddings:

### Approach 1: Data Module (This File - Simpler ✅)

**Files:**
- `create_esm_pert_features.py` - Create ESM features file
- `train_with_esm_features.py` - Training script

**How it works:**
1. Pre-compute ESM embeddings → save to `.pt` file
2. Pass to data module via `perturbation_features_file`
3. Use **standard model** (no modifications)

**Pros:**
- ✅ Simpler (no model changes)
- ✅ Matches Colab notebook approach
- ✅ Easy to swap different perturbation features
- ✅ Features computed once, reused across runs

**Cons:**
- Requires pre-computing features
- Less flexible for on-the-fly changes

**Use when:**
- Following the Virtual Cell Challenge Colab
- You have a fixed set of perturbations
- You want maximum simplicity

### Approach 2: Model Architecture (Alternative)

**Files:**
- `src/state/tx/models/esm_perturbation.py` - ESM model
- `train_esm_perturbation.py` - Training script

**How it works:**
1. Load ESM embeddings in **model initialization**
2. Model maps genes → ESM embeddings on-the-fly
3. Use `model=esm_state`

**Pros:**
- ✅ More flexible (no pre-computation needed)
- ✅ Can handle new genes dynamically
- ✅ Cleaner for zero-shot scenarios

**Cons:**
- More complex model architecture
- ESM embeddings loaded with model

**Use when:**
- You need maximum flexibility
- Genes change frequently
- You're doing zero-shot experiments

## Examples

### Example 1: Virtual Cell Challenge

```bash
# Create features
python create_esm_pert_features.py \
    --from-toml competition_support_set/starter.toml \
    --output competition_support_set/ESM2_pert_features.pt

# Train
python train_with_esm_features.py \
    --config competition_support_set/starter.toml \
    --pert-features competition_support_set/ESM2_pert_features.pt \
    --pert-col target_gene \
    --control-pert non-targeting \
    --max-steps 40000
```

### Example 2: Custom Genetic Screen

```bash
# Create gene list
cat > my_genes.txt << EOF
STAT1
IRF1
NFKB1
TP53
MYC
EOF

# Create features
python create_esm_pert_features.py \
    --gene-list my_genes.txt \
    --output my_esm_features.pt

# Train
python train_with_esm_features.py \
    --config my_config.toml \
    --pert-features my_esm_features.pt \
    --output-dir ./my_output
```

### Example 3: Multi-dataset Training

```toml
# config.toml
[datasets]
replogle_k562 = "/data/replogle/k562"
replogle_rpe1 = "/data/replogle/rpe1"
my_screen = "/data/my_screen"

[training]
replogle_k562 = "train"
replogle_rpe1 = "train"
my_screen = "train"
```

```bash
# Extract genes from all datasets
python create_esm_pert_features.py \
    --from-toml config.toml \
    --output multi_dataset_esm.pt

# Train jointly
python train_with_esm_features.py \
    --config config.toml \
    --pert-features multi_dataset_esm.pt \
    --model state_lg \
    --max-steps 100000
```

## Troubleshooting

### Issue: Gene not found in ESM embeddings

**Symptom:**
```
WARNING - Missing ESM embeddings for 5 genes: ['GENE1', 'GENE2', ...]
```

**Solution:**
The script automatically assigns zero embeddings to missing genes. To fix properly:
1. Verify gene symbol spelling (case-sensitive)
2. Generate custom ESM embedding for that gene
3. Or filter it from your dataset

### Issue: "No genes found!"

**Solution:**
1. Check your TOML file has `[datasets]` section
2. Verify dataset paths exist
3. Check h5 files have perturbation columns (target_gene, gene, pert, etc.)

### Issue: Feature file dimension mismatch

**Symptom:**
```
RuntimeError: Expected input dimension X, got Y
```

**Solution:**
Regenerate the ESM features file - you may have mixed different ESM versions.

## FAQ

**Q: What's the difference from the other ESM approach?**
A: This approach (data-level) is simpler and matches the Colab. The other approach (model-level) is more flexible. Choose based on your needs.

**Q: Can I use this with chemical perturbations?**
A: Yes, but you'd need chemical embeddings (e.g., Morgan fingerprints) instead of ESM. Create a similar features file mapping compound names to embeddings.

**Q: Do I need to retrain if I add new genes?**
A: Yes, you'll need to regenerate the features file and retrain. Or use Approach 2 (model-level) for dynamic genes.

**Q: Can I mix ESM and one-hot encoding?**
A: Not simultaneously. Choose one encoding method per training run.

**Q: How much faster is this than one-hot?**
A: ESM embeddings are denser (5120 dims) vs sparse one-hot (10k+ dims). Training speed is similar, but generalization is much better.

## Citation

If you use ESM perturbation features in your work, please cite:

1. **State framework**: [Add State citation when published]
2. **ESM2**:
   ```
   @article{lin2022language,
     title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
     author={Lin, Zeming and Akin, Halil and others},
     journal={bioRxiv},
     year={2022}
   }
   ```

## Support

- **Documentation**: See this README and `ESM_PERTURBATION_README.md`
- **Examples**: Check example commands above
- **Issues**: Open an issue on GitHub
