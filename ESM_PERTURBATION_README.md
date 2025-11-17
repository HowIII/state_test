# ESM-Based Perturbation Encoder

This implementation extends the State Transition model to use **ESM2 protein embeddings** as the perturbation encoder instead of one-hot encoding for genetic perturbations.

## Overview

### What is the ESM Perturbation Encoder?

The ESM Perturbation Encoder leverages pre-trained protein embeddings from the Evolutionary Scale Modeling (ESM2) model to represent genetic perturbations. Instead of using sparse one-hot encodings, this approach:

1. **Maps gene names** (e.g., "STAT1", "TP53") to their ESM2 protein embeddings
2. **Projects embeddings** through an MLP to the model's hidden dimension
3. **Enables better generalization** by encoding semantic relationships between genes

### Why Use ESM Embeddings for Perturbations?

**Advantages over one-hot encoding:**
- **Semantic information**: Genes with similar protein sequences/functions have similar embeddings
- **Better zero-shot generalization**: Can predict effects of unseen perturbations
- **Dimensionality reduction**: ESM embeddings (5120-dim) are denser than one-hot (10k+ dims)
- **Transfer learning**: Leverages knowledge from protein language models

## Installation

The ESM perturbation encoder is integrated into the existing State codebase. No additional installation is required beyond the standard State dependencies.

## Quick Start

### 1. Prepare Your Configuration

Create a TOML configuration file specifying your datasets (see `esm_genetic_example.toml`):

```toml
[datasets]
my_dataset = "/path/to/genetic/perturbation/data"

[training]
my_dataset = "train"

[fewshot."my_dataset.CellType"]
val = ["GENE1", "GENE2"]
test = ["GENE3", "GENE4"]
```

### 2. Run Training

**Basic usage:**
```bash
python train_esm_perturbation.py --config esm_genetic_example.toml
```

**With custom parameters:**
```bash
python train_esm_perturbation.py \\
    --config esm_genetic_example.toml \\
    --output_dir ./my_output \\
    --batch_size 16 \\
    --max_steps 50000 \\
    --lr 1e-4 \\
    --use_wandb
```

**With Hydra overrides:**
```bash
python train_esm_perturbation.py \\
    --config esm_genetic_example.toml \\
    model.kwargs.hidden_dim=1024 \\
    model.kwargs.esm_encoder_layers=3 \\
    training.val_freq=1000
```

### 3. Using the Standard CLI

You can also use the standard State CLI:

```bash
state tx train \\
    data.kwargs.toml_config_path="esm_genetic_example.toml" \\
    data.kwargs.embed_key=X_uce \\
    model=esm_state \\
    output_dir="$HOME/esm_output"
```

## Architecture

### Model Components

1. **ESM Perturbation Encoder**
   - Input: Gene name → ESM2 embedding (5120-dim for human genes)
   - Projection: MLP layers to hidden_dim
   - Output: Perturbation embedding in hidden space

2. **Basal Cell Encoder**
   - Input: Control cell expression (embeddings or counts)
   - Projection: MLP to hidden_dim
   - Output: Basal state embedding

3. **Transformer Backbone**
   - Input: Perturbation embedding + Basal embedding (combined)
   - Architecture: LLaMA-style transformer (configurable layers/heads)
   - Output: Transformed cell state

4. **Decoder**
   - Input: Transformer output
   - Projection: MLP to output space (gene expression)
   - Output: Predicted perturbed cell state

### Data Flow

```
Gene Name (e.g., "STAT1")
    ↓
ESM2 Lookup → [5120-dim embedding]
    ↓
ESM Encoder (MLP) → [hidden_dim]
    ↓              ↗
    + ← Basal Encoder(control cells)
    ↓
Transformer Backbone
    ↓
Output Decoder → Predicted perturbed state
```

## Configuration

### Model Configuration

The ESM model configuration is in `src/state/configs/model/esm_state.yaml`:

```yaml
name: esm_state

kwargs:
  # Standard State Transition parameters
  hidden_dim: 696
  cell_set_len: 512
  n_encoder_layers: 1
  n_decoder_layers: 1

  # ESM-specific parameters
  use_esm_for_pert: true
  esm_embeddings_path: /large_storage/ctc/ML/data/cell/misc/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt
  gene_to_esm_map_path: null  # Not needed for dict format
  esm_encoder_layers: 2  # Number of MLP layers for ESM projection
```

### Key Parameters

**ESM Parameters:**
- `use_esm_for_pert`: Enable/disable ESM perturbation encoding (default: true)
- `esm_embeddings_path`: Path to ESM2 embeddings file
- `gene_to_esm_map_path`: Optional gene→index mapping (for tensor format)
- `esm_encoder_layers`: Number of MLP layers to project ESM embeddings (default: 2)

**Training Parameters:**
- `batch_size`: Number of cell sentences per batch (default: 8)
- `cell_set_len`: Sequence length (number of cells per sentence) (default: 512)
- `hidden_dim`: Hidden dimension for all encoders/decoders (default: 696)
- `lr`: Learning rate (default: 3e-4)
- `max_steps`: Maximum training steps (default: 100000)

**Loss Parameters:**
- `distributional_loss`: Loss function (energy, sinkhorn, mse) (default: energy)
- `blur`: Blur parameter for distributional losses (default: 0.05)
- `predict_residual`: Predict residual changes vs. absolute states (default: true)

## File Structure

```
state_test/
├── train_esm_perturbation.py          # Standalone training script
├── esm_genetic_example.toml            # Example configuration
├── ESM_PERTURBATION_README.md          # This file
│
├── src/state/
│   ├── configs/
│   │   └── model/
│   │       └── esm_state.yaml          # ESM model configuration
│   │
│   └── tx/
│       ├── models/
│       │   ├── esm_perturbation.py     # ESM model implementation
│       │   └── __init__.py             # Model registration
│       │
│       ├── data/
│       │   └── esm_utils.py            # ESM data utilities
│       │
│       └── utils/
│           └── __init__.py             # Model factory (updated)
```

## ESM Embeddings

### Default Embeddings

The system uses pre-computed ESM2 embeddings for human genes:

- **Path**: `/large_storage/ctc/ML/data/cell/misc/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt`
- **Format**: Dictionary `{gene_name: embedding_tensor}`
- **Dimensions**: 5120 (ESM2-3B model)
- **Coverage**: ~19,790 human genes

### Custom ESM Embeddings

To use custom ESM embeddings:

1. **Dictionary format** (recommended):
   ```python
   {
       "GENE1": torch.Tensor([...]),  # shape: [5120]
       "GENE2": torch.Tensor([...]),
       ...
   }
   ```
   Save as `.pt` file and specify path in config.

2. **Tensor format**:
   - Embeddings: `torch.Tensor` of shape `[num_genes, esm_dim]`
   - Mapping: Dictionary `{gene_name: index}`
   - Specify both paths in config

### Generating ESM Embeddings

To generate ESM embeddings for new genes:

```python
from transformers import AutoTokenizer, EsmModel
import torch

# Load ESM2 model
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D")

# Get protein sequence for gene (from Ensembl/UniProt)
protein_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"

# Tokenize and get embeddings
inputs = tokenizer(protein_seq, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

# Save
gene_embeddings = {"MY_GENE": embedding.squeeze()}
torch.save(gene_embeddings, "my_esm_embeddings.pt")
```

## Examples

### Example 1: Basic Training on Replogle Dataset

```bash
# Edit esm_genetic_example.toml to point to your data
# Then run:
python train_esm_perturbation.py \\
    --config esm_genetic_example.toml \\
    --output_dir ./replogle_esm_output \\
    --batch_size 8 \\
    --max_steps 100000
```

### Example 2: Zero-shot Gene Prediction

Hold out specific genes for testing:

```toml
# In your config file:
[fewshot."replogle_k562.K562"]
test = ["STAT1", "IRF1", "NFKB1"]  # Test genes
```

```bash
python train_esm_perturbation.py --config my_zeroshot_config.toml
```

The model will train without seeing these genes and can still predict their effects using ESM embeddings!

### Example 3: Fine-tuning with Custom ESM Embeddings

```bash
python train_esm_perturbation.py \\
    --config esm_genetic_example.toml \\
    --esm_embeddings_path /path/to/my_custom_esm.pt \\
    --output_dir ./custom_esm_output
```

### Example 4: Using with W&B Logging

```bash
python train_esm_perturbation.py \\
    --config esm_genetic_example.toml \\
    --use_wandb \\
    --wandb_project my-esm-project \\
    --output_dir ./wandb_output
```

## Comparison with One-Hot Encoding

| Feature | One-Hot Encoding | ESM Embeddings |
|---------|-----------------|----------------|
| **Dimensionality** | ~10,000-50,000 | 5,120 |
| **Semantic info** | None (sparse) | Rich (dense) |
| **Zero-shot** | Poor | Good |
| **Gene similarity** | All orthogonal | Continuous space |
| **Training speed** | Fast (sparse) | Moderate (dense) |
| **Best for** | Seen perturbations | Generalization |

## Troubleshooting

### Issue: Gene not found in ESM embeddings

**Error**: `Gene XYZ not found in ESM embeddings, using zero vector`

**Solution**:
1. Verify gene name matches ESM dictionary keys (case-sensitive)
2. Generate custom ESM embedding for that gene
3. Or, filter out that gene from your dataset

### Issue: CUDA out of memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `batch_size`: `--batch_size 4`
2. Reduce `cell_set_len`: `model.kwargs.cell_set_len=256`
3. Reduce `hidden_dim`: `model.kwargs.hidden_dim=512`
4. Use gradient accumulation: `training.accumulate_grad_batches=2`

### Issue: ESM embeddings file not found

**Error**: `FileNotFoundError: /large_storage/.../ESM2.pt`

**Solution**:
Specify custom path: `--esm_embeddings_path /path/to/your/esm.pt`

## Advanced Usage

### Mixed Perturbation Types

For datasets with both genetic and chemical perturbations, you can:
1. Use ESM for genetic perturbations
2. Use chemical embeddings (e.g., Morgan fingerprints) for compounds
3. Combine in the data loader

### Multi-Species Support

To work with multiple species:
1. Load species-specific ESM embeddings
2. Use `esm2-scbasecamp` embeddings (503k genes, cross-species)
3. Specify in config: `embeddings.current: esm2-scbasecamp`

### Transfer Learning

Pre-train on large genetic perturbation datasets, then fine-tune:

```bash
# Pre-train
python train_esm_perturbation.py --config pretrain.toml --output_dir ./pretrain

# Fine-tune
python train_esm_perturbation.py \\
    --config finetune.toml \\
    model.kwargs.init_from=./pretrain/checkpoints/best.ckpt
```

## Citation

If you use the ESM perturbation encoder, please cite:

1. **State framework**: [State paper citation]
2. **ESM2 model**:
   ```
   @article{lin2022language,
     title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
     author={Lin, Zeming and Akin, Halil and others},
     journal={bioRxiv},
     year={2022}
   }
   ```

## Support

For questions or issues:
1. Check this README and example configurations
2. Review error messages and troubleshooting section
3. Open an issue on the State GitHub repository

## License

This code is part of the State framework and follows the same license.
