# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**State** is a machine learning framework for predicting cellular responses to perturbations across diverse contexts. It includes two main components:

1. **State Transition (ST)** - Models for predicting perturbation effects on cell states
2. **State Embedding (SE)** - Foundation models for embedding and annotating cellular data

The repository supports both traditional one-hot perturbation encoding and ESM2 protein embedding-based perturbation encoding for genetic perturbations.

## Installation & Environment

**Package Manager**: Uses `uv` (not pip/conda)

```bash
# Install from PyPI
uv tool install arc-state

# Install from source (for development)
git clone git@github.com:ArcInstitute/state.git
cd state
uv run state

# Editable install for development
uv tool install -e .
```

**Python Version**: 3.10-3.12 (specified in pyproject.toml, currently using 3.11)

**Optional Dependencies**:
```bash
# For vector database features
uv tool install ".[vectordb]"
uv sync --extra vectordb
```

## Common Commands

### State Transition (ST) Training

**Basic training**:
```bash
state tx train \
  data.kwargs.toml_config_path="examples/fewshot.toml" \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.num_workers=12 \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=TARGET1 \
  training.max_steps=40000 \
  training.val_freq=100 \
  training.batch_size=8 \
  model=pertsets \
  output_dir="$HOME/state" \
  name="test"
```

**ESM-based perturbation training (Approach 1 - Colab method)**:
```bash
# Step 1: Create ESM features
python create_esm_pert_features.py \
  --from-toml config.toml \
  --output ESM2_pert_features.pt

# Step 2: Train
python train_with_esm_features.py \
  --config config.toml \
  --pert-features ESM2_pert_features.pt
```

**ESM-based perturbation training (Approach 2 - Model method)**:
```bash
python train_esm_perturbation.py \
  --config esm_genetic_example.toml \
  --output_dir ./output
```

### Data Preprocessing

**Training data preprocessing**:
```bash
state tx preprocess_train \
  --adata /path/to/raw_data.h5ad \
  --output /path/to/preprocessed_training_data.h5ad \
  --num_hvgs 2000
```

**Inference data preprocessing** (creates control template):
```bash
state tx preprocess_infer \
  --adata /path/to/real_data.h5ad \
  --output /path/to/control_template.h5ad \
  --control_condition "DMSO" \
  --pert_col "treatment" \
  --seed 42
```

### Model Evaluation & Inference

**Evaluate on test set**:
```bash
state tx predict \
  --output-dir $HOME/state/test/ \
  --checkpoint final.ckpt
```

**Inference on new data**:
```bash
state tx infer \
  --output $HOME/state/test/ \
  --output_dir /path/to/model/ \
  --checkpoint /path/to/model/final.ckpt \
  --adata /path/to/anndata/processed.h5 \
  --pert_col gene \
  --embed_key X_hvg
```

### State Embedding (SE) Commands

**Train embedding model**:
```bash
state emb fit --conf ${CONFIG}
```

**Generate embeddings**:
```bash
state emb transform \
  --model-folder /path/to/SE-600M \
  --checkpoint /path/to/checkpoint.ckpt \
  --input /path/to/data.h5ad \
  --output /path/to/output.h5ad
```

**Vector database operations**:
```bash
# Build database
state emb transform \
  --model-folder /path/to/SE-600M \
  --input /path/to/data.h5ad \
  --lancedb tmp/state_embeddings.lancedb \
  --gene-column gene_symbols

# Query database
state emb query \
  --lancedb tmp/state_embeddings.lancedb \
  --input tmp/query.h5ad \
  --output tmp/similar_cells.csv \
  --k 3
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_bidirectional_models.py
```

### Code Quality

**Linter**: Uses `ruff` with 120 character line length

```bash
# Run linter (auto-fix enabled)
ruff check .

# Format code
ruff format .
```

## Architecture & Code Structure

### Directory Organization

```
src/state/
├── __main__.py           # CLI entry point
├── _cli/                 # CLI implementation
│   ├── _emb/            # Embedding commands
│   └── _tx/             # Transition commands
├── configs/             # Hydra configuration files
│   ├── config.yaml      # ST default config
│   ├── state-defaults.yaml  # SE default config
│   ├── data/            # Data configs
│   ├── model/           # Model configs (pertsets, esm_state, state, etc.)
│   ├── training/        # Training configs
│   └── wandb/           # W&B configs
├── emb/                 # State Embedding implementation
│   ├── data/
│   ├── nn/
│   ├── train/
│   └── vectordb.py
└── tx/                  # State Transition implementation
    ├── callbacks/       # Training callbacks
    ├── data/            # Data loading and preprocessing
    │   └── esm_utils.py  # ESM embedding utilities
    ├── models/          # Model architectures
    │   ├── base.py      # Base perturbation model
    │   ├── state_transition.py  # Main ST model
    │   ├── esm_perturbation.py  # ESM-based ST model
    │   ├── scgpt/       # scGPT integration
    │   └── scvi/        # scVI integration
    └── utils/           # Utilities (transformer backbones, etc.)
```

### Key Model Architectures

**State Transition Models** (in `src/state/tx/models/`):
- `StateTransitionPerturbationModel` - Main ST model with one-hot perturbation encoding
- `ESMStateTransitionModel` - ST model with ESM2 protein embeddings for perturbations
- `PerturbMeanPerturbationModel` - Baseline mean-based model
- `PseudobulkPerturbationModel` - Pseudobulk-level predictions
- Integration models: scGPT, scVI

**Model Components**:
1. **Perturbation Encoder** - Encodes perturbation information (one-hot or ESM embeddings)
2. **Basal Cell Encoder** - Encodes control cell expression
3. **Transformer Backbone** - LLaMA or GPT2-based transformer (configurable bidirectional attention)
4. **Decoder** - Projects to gene expression space

### ESM Perturbation Encoder: Two Approaches

The repository provides **two complete implementations** for using ESM2 embeddings:

**Approach 1: Pre-computed Features (Recommended for Virtual Cell Challenge)**
- Files: `create_esm_pert_features.py`, `train_with_esm_features.py`
- Pre-compute ESM embeddings into a features file
- Use standard models (state, state_sm, state_lg) with pert_features parameter
- Simpler, matches Colab notebook approach
- See: `ESM_FEATURES_README.md`

**Approach 2: Dynamic Model Integration**
- Files: `train_esm_perturbation.py`, `src/state/tx/models/esm_perturbation.py`
- ESM embeddings loaded into model, lookup happens during forward pass
- More flexible for zero-shot experiments
- Requires model config: `src/state/configs/model/esm_state.yaml`
- See: `ESM_PERTURBATION_README.md`, `QUICKSTART_ESM.md`

### TOML Configuration System

Experiments are configured via TOML files that define datasets, training splits, and evaluation scenarios.

**Structure**:
```toml
[datasets]
dataset_name = "/path/to/data/"

[training]
dataset_name = "train"

[zeroshot]  # Hold out entire cell types
"dataset_name.cell_type" = "test"

[fewshot]   # Hold out specific perturbations within cell types
[fewshot."dataset_name.cell_type"]
val = ["GENE1", "GENE2"]
test = ["GENE3", "GENE4"]
```

**Important**:
- Cell types not in `[zeroshot]` automatically participate in training
- Perturbations not in `[fewshot]` automatically go to training set
- No conflicts between zeroshot and fewshot for same cell type
- See `examples/fewshot.toml`, `examples/zeroshot.toml`, `examples/mixed.toml`

### Hydra Configuration

The CLI uses Hydra for configuration management. Override parameters using dot notation:

```bash
state tx train \
  data.kwargs.batch_size=64 \
  model.kwargs.hidden_dim=512 \
  training.lr=0.001 \
  model=esm_state
```

Available model configs (in `src/state/configs/model/`):
- `pertsets` - Default ST model
- `esm_state` - ESM perturbation encoder
- `state`, `state_sm`, `state_lg` - Standard variants
- `scgpt-genetic`, `scgpt-chemical` - scGPT variants
- `cpa`, `scvi` - Baseline models

### Data Format Requirements

**h5ad files** (AnnData format):
- Matrix format: CSR (compressed sparse row)
- Required in `var`: `gene_name` column
- Typical `.obsm` keys: `X_hvg` (highly variable genes), `X_uce` (UCE embeddings)
- Perturbation annotations in `.obs[pert_col]`
- Cell type annotations in `.obs[cell_type_key]`

## Development Workflow

### Adding a New Model

1. Create model class in `src/state/tx/models/your_model.py`
2. Inherit from `PerturbationModel` base class
3. Register in `src/state/tx/models/__init__.py`
4. Create config file in `src/state/configs/model/your_model.yaml`
5. Add model factory logic in `src/state/tx/utils/__init__.py` if needed

### ESM Embeddings

**Default path**: `/large_storage/ctc/ML/data/cell/misc/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt`

**Format**: Dictionary `{gene_name: torch.Tensor([5120-dim])}`

**Coverage**: ~19,790 human genes from ESM2-3B model

**Custom embeddings**: Specify via `esm_embeddings_path` in config

### Transformer Backbones

The codebase supports custom transformer backbones (LLaMA, GPT2) with configurable bidirectional attention:

```yaml
transformer_backbone_key: llama  # or gpt2
transformer_backbone_kwargs:
  bidirectional_attention: false  # Set true for non-causal attention
  max_position_embeddings: 512
  hidden_size: 696
  num_hidden_layers: 8
  num_attention_heads: 12
```

**Important**: Bidirectional attention modifies the attention mask to allow full context access (non-causal).

### Loss Functions

Available distributional losses (in `model.kwargs.distributional_loss`):
- `energy` - Energy distance (default)
- `sinkhorn` - Sinkhorn divergence
- `mse` - Mean squared error
- `combined` - Weighted combination

**Blur parameter**: Controls smoothness of distributional losses (default: 0.05)

## Important Notes

### Singularity/Containerization

Build and run via Singularity:
```bash
singularity build state.sif singularity.def
singularity run state.sif --help

# With GPU and volume mounts
singularity run --nv -B /large_storage:/large_storage state.sif emb transform ...
```

### Associated Repositories

- **Evaluation**: [cell-eval](https://github.com/ArcInstitute/cell-eval)
- **Data loading**: [cell-load](https://github.com/ArcInstitute/cell-load)

### License Considerations

- Code: CC BY-NC-SA 4.0
- Model weights: Arc Research Institute State Model Non-Commercial License
- See `LICENSE`, `MODEL_LICENSE.md`, `MODEL_ACCEPTABLE_USE_POLICY.md`
