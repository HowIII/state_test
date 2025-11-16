# CLAUDE.md - AI Assistant Guide for State Repository

This document provides comprehensive guidance for AI assistants working with the State codebase. State is a machine learning framework for predicting cellular responses to perturbations across diverse contexts.

## Table of Contents
1. [Repository Overview](#repository-overview)
2. [Project Structure](#project-structure)
3. [Development Workflows](#development-workflows)
4. [Code Conventions](#code-conventions)
5. [Testing Practices](#testing-practices)
6. [Common Commands](#common-commands)
7. [Configuration Management](#configuration-management)
8. [Key Files Reference](#key-files-reference)
9. [Git Practices](#git-practices)
10. [Dependencies](#dependencies)

---

## Repository Overview

**Project**: State - Cellular Perturbation Prediction Framework
**Package Name**: `arc-state`
**Version**: 0.9.31
**License**: CC BY-NC-SA 4.0 (code), Arc Research Institute Non-Commercial License (model weights)
**Python Version**: 3.10-3.12 (currently using 3.11)
**Package Manager**: `uv` (Astral's modern Python package manager)

### Purpose
State predicts how cells respond to perturbations (genetic modifications, chemical treatments) using:
1. **State Transition (ST)**: Models that predict cellular response to perturbations
2. **State Embedding (SE)**: Foundation models that learn cell representations from transcriptomic data

### Associated Repositories
- **cell-eval**: Model evaluation framework
- **cell-load**: Data loaders and preprocessing utilities

---

## Project Structure

```
/home/user/state_test/
├── src/state/                      # Main Python package (17,955 lines across 80 files)
│   ├── __main__.py                 # CLI entry point with Hydra configuration
│   ├── _cli/                       # CLI command implementations
│   │   ├── _emb/                   # Embedding model commands (fit, transform, query)
│   │   └── _tx/                    # Transition model commands (train, predict, infer)
│   ├── tx/                         # Transition Model Module (perturbation prediction)
│   │   ├── models/                 # Model implementations
│   │   │   ├── state_transition.py # Main STATE model (Transformer-based)
│   │   │   ├── scgpt/              # scGPT-based models
│   │   │   ├── scvi/               # scVI-based models
│   │   │   ├── cpa/                # CPA models
│   │   │   ├── decoders.py         # Gene expression decoders
│   │   │   └── [baselines]         # Various baseline models
│   │   ├── data/dataset/           # Data loading (PerturbationDataset)
│   │   └── callbacks/              # Training callbacks (FLOPs, speed monitoring)
│   ├── emb/                        # Embedding Model Module (foundation models)
│   │   ├── nn/                     # Neural network components
│   │   │   ├── model.py            # StateEmbeddingModel
│   │   │   ├── flash_transformer.py # Efficient transformer
│   │   │   └── loss.py             # Wasserstein, KL, MMD losses
│   │   ├── train/                  # Training infrastructure
│   │   └── vectordb.py             # LanceDB integration
│   └── configs/                    # Hydra configuration files
│       ├── config.yaml             # Main TX config
│       ├── state-defaults.yaml     # EMB config with defaults
│       ├── model/                  # 15+ model variants
│       ├── data/                   # Data configurations
│       ├── training/               # Training hyperparameters
│       └── wandb/                  # W&B logging config
├── tests/                          # Test suite
│   ├── test_bidirectional_models.py # Bidirectional attention tests
│   └── test_callbacks.py           # Callback tests
├── examples/                       # Example TOML configs
│   ├── zeroshot.toml               # Zero-shot evaluation
│   ├── fewshot.toml                # Few-shot evaluation
│   └── mixed.toml                  # Combined evaluation
├── scripts/                        # Utility scripts
├── pyproject.toml                  # Project metadata
├── ruff.toml                       # Linter config
└── README.md                       # User documentation
```

### Module Breakdown

#### TX (Transition) Module - `src/state/tx/`
**Purpose**: Predict cellular response to perturbations
**Key Files**:
- `tx/models/state_transition.py`: Main STATE model with transformer backbone
- `tx/models/base.py`: Base PerturbationModel class
- `tx/data/dataset/scgpt_perturbation_dataset.py`: Data loading from h5 files

**Architecture**:
- Backbone: Configurable transformer (LLaMA or GPT2, default LLaMA)
- Input: Perturbation embeddings + cell type + cell context (control cells)
- Output: Predicted perturbed cell gene expression
- Loss: Combined Sinkhorn + Energy loss (optimal transport-based)

#### EMB (Embedding) Module - `src/state/emb/`
**Purpose**: Learn cell embeddings from transcriptomic data
**Key Files**:
- `emb/nn/model.py`: StateEmbeddingModel using FlashTransformer
- `emb/nn/flash_transformer.py`: Efficient transformer implementation
- `emb/nn/loss.py`: Embedding losses (Wasserstein, KL, MMD)

**Architecture**:
- Model: FlashTransformer encoder with learnable CLS token
- Input: Gene token embeddings (ESM2-derived)
- Output: Cell embeddings
- Optimization: Warmup + cosine annealing scheduler

---

## Development Workflows

### Setting Up Development Environment

```bash
# Clone repository
git clone git@github.com:ArcInstitute/state.git
cd state

# Install in editable mode for development
uv tool install -e .

# Install with optional dependencies
uv sync --all-extras

# Install development dependencies
uv sync --group dev
```

### Training a Transition Model

1. **Create TOML configuration** (see `examples/fewshot.toml`):
```toml
[datasets]
your_dataset = "/path/to/data/"

[training]
your_dataset = "train"

[fewshot]
[fewshot."your_dataset.cell_type"]
val = ["GENE1", "GENE2"]
test = ["GENE3", "GENE4"]
```

2. **Run training**:
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
  name="experiment_name"
```

3. **Run prediction**:
```bash
state tx predict --output-dir $HOME/state/experiment_name/ --checkpoint final.ckpt
```

### Training an Embedding Model

```bash
state emb fit --conf ${CONFIG}
```

### Running Inference

```bash
state tx infer \
  --output $HOME/state/test/ \
  --output_dir /path/to/model/ \
  --checkpoint /path/to/model/final.ckpt \
  --adata /path/to/anndata/processed.h5 \
  --pert_col gene \
  --embed_key X_hvg
```

### Data Preprocessing

**For Training**:
```bash
state tx preprocess_train \
  --adata /path/to/raw_data.h5ad \
  --output /path/to/preprocessed_training_data.h5ad \
  --num_hvgs 2000
```

**For Inference** (creates control template):
```bash
state tx preprocess_infer \
  --adata /path/to/real_data.h5ad \
  --output /path/to/control_template.h5ad \
  --control_condition "DMSO" \
  --pert_col "treatment" \
  --seed 42
```

---

## Code Conventions

### Python Style
- **Line Length**: 120 characters (enforced by Ruff)
- **Linter**: Ruff (configured in `ruff.toml`)
- **Ignored Rules**: E722 (bare except)
- **Auto-fix**: Enabled for all fixable rules

### Type Hints
- Type hints are used throughout the codebase (`py.typed` marker present)
- Use Pyright for static type checking (configured in `pyproject.toml`)

### Code Organization
1. **Imports**: Standard library → Third-party → Local imports
2. **Class Structure**: PyTorch Lightning modules inherit from `LightningModule`
3. **Configuration**: Use Hydra dataclasses for config validation
4. **Logging**: Use `wandb` for experiment tracking

### Naming Conventions
- **Files**: Snake_case (e.g., `state_transition.py`)
- **Classes**: PascalCase (e.g., `StateTransitionModel`)
- **Functions/Variables**: Snake_case (e.g., `compute_loss`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_HIDDEN_DIM`)

### Documentation
- Use docstrings for all public classes and functions
- Include type hints in function signatures
- Reference line numbers when discussing code: `file_path:line_number`

### Model Development
- All models inherit from `tx/models/base.py:PerturbationModel` (TX) or use Lightning directly (EMB)
- Implement required methods: `forward()`, `training_step()`, `validation_step()`
- Use callbacks for monitoring (see `tx/callbacks/`)

---

## Testing Practices

### Test Framework
- **Framework**: pytest
- **Location**: `/home/user/state_test/tests/`

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_bidirectional_models.py

# Run with verbose output
pytest -v tests/
```

### Test Coverage
Current tests focus on:
1. **Bidirectional Attention**: Tests in `test_bidirectional_models.py`
   - Non-causal masking configuration
   - Attention output differences vs. causal models
   - Gradient flow validation

2. **Training Callbacks**: Tests in `test_callbacks.py`
   - FLOPs calculation accuracy
   - Callback timing and logging
   - Forward vs. backward FLOPs estimation

### Writing Tests
- Use pytest fixtures for model configurations
- Create fake models/trainers for isolated testing
- Test both forward and backward passes
- Validate gradient flow for custom components

---

## Common Commands

### Package Management
```bash
# Install from PyPI
uv tool install arc-state

# Install from source (editable)
uv tool install -e .

# Sync dependencies
uv sync --all-extras

# Update dependencies
uv sync --upgrade
```

### CLI Commands

**Transition Models**:
```bash
state tx train        # Train a transition model
state tx predict      # Predict on test set from TOML config
state tx infer        # Inference on new data
state tx preprocess_train  # Preprocess training data
state tx preprocess_infer  # Create control template for inference
```

**Embedding Models**:
```bash
state emb fit         # Train embedding model
state emb transform   # Generate embeddings from trained model
state emb query       # Query vector database
state emb preprocess  # Preprocess data
state emb eval        # Evaluate model
```

### Vector Database (optional feature)
```bash
# Build database
state emb transform \
  --model-folder /path/to/model \
  --input /path/to/data.h5ad \
  --lancedb tmp/state_embeddings.lancedb \
  --gene-column gene_symbols

# Query database
state emb query \
  --lancedb tmp/state_embeddings.lancedb \
  --input tmp/query_cells.h5ad \
  --output tmp/similar_cells.csv \
  --k 3
```

### Development Tools
```bash
# Run linter
ruff check src/

# Auto-fix issues
ruff check --fix src/

# Format code
ruff format src/

# Type checking
pyright src/
```

---

## Configuration Management

### Hydra Configuration System

State uses **Hydra** for hierarchical, composable configurations.

#### Configuration Groups

**Location**: `src/state/configs/`

1. **Model Configs** (`configs/model/`):
   - `state.yaml`: Main STATE model (default)
   - `state_lg.yaml`: Large variant
   - `state_sm.yaml`: Small variant
   - `scgpt-genetic.yaml`, `scgpt-chemical.yaml`: scGPT variants
   - `cpa.yaml`: CPA model
   - `scvi.yaml`: scVI model
   - Baselines: `embedsum.yaml`, `context_mean.yaml`, etc.

2. **Data Configs** (`configs/data/`):
   - `perturbation.yaml`: TX data configuration
   - `default.yaml`: EMB data configuration

3. **Training Configs** (`configs/training/`):
   - `default.yaml`: Learning rate, batch size, max steps, etc.

4. **W&B Configs** (`configs/wandb/`):
   - `default.yaml`: Logging configuration

#### Overriding Configuration

**Command-line overrides**:
```bash
state tx train \
  model=state_lg \
  model.kwargs.hidden_dim=512 \
  training.lr=1e-4 \
  training.batch_size=16 \
  data.kwargs.num_workers=8
```

**Config composition**:
```bash
# Use specific model + custom training config
state tx train model=scgpt-genetic training=my_training_config
```

### TOML Experiment Configuration

User-defined TOML files specify data splits for experiments.

#### Structure:
```toml
[datasets]
dataset_name = "/path/to/data/"

[training]
dataset_name = "train"  # Include in training

[zeroshot]  # Hold out entire cell types
"dataset_name.cell_type" = "test"
"dataset_name.another_cell_type" = "val"

[fewshot]  # Perturbation-level splits
[fewshot."dataset_name.cell_type"]
val = ["GENE1", "GENE2"]
test = ["GENE3", "GENE4"]
# All other perturbations go to training
```

#### Examples:
- `examples/zeroshot.toml`: Evaluate on unseen cell types
- `examples/fewshot.toml`: Limited perturbation examples
- `examples/mixed.toml`: Combined zeroshot + fewshot

---

## Key Files Reference

### Critical Source Files

| File | Purpose | Lines |
|------|---------|-------|
| `src/state/__main__.py` | CLI entry point | Entry |
| `src/state/tx/models/state_transition.py` | Main STATE model | Core |
| `src/state/tx/models/base.py` | Base model interface | Base |
| `src/state/emb/nn/model.py` | Embedding model | Core |
| `src/state/emb/nn/flash_transformer.py` | Efficient transformer | Core |
| `src/state/tx/data/dataset/scgpt_perturbation_dataset.py` | Data loading | Data |

### Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, dependencies, build config |
| `ruff.toml` | Linter configuration (line-length: 120) |
| `src/state/configs/config.yaml` | Main TX Hydra config |
| `src/state/configs/state-defaults.yaml` | EMB Hydra config |
| `.python-version` | Python version (3.11) |

### Important Model Components

**Transition Models** (`src/state/tx/models/`):
- `state_transition.py:StateTransitionModel`: Main model class
- `decoders.py`: Gene expression decoders
- `scgpt/model.py`: scGPT architecture (976 lines)
- `utils.py`: MLP builders, backbone initialization, LoRA utilities

**Embedding Models** (`src/state/emb/nn/`):
- `model.py:StateEmbeddingModel`: Main embedding model
- `flash_transformer.py`: Memory-efficient transformer
- `loss.py`: Wasserstein, KL divergence, MMD losses

### Data Files

**AnnData Requirements**:
- Format: CSR matrix (sparse)
- Required columns in `.var`: `gene_name`
- Expression data: `.X` or `.obsm['X_hvg']`
- Metadata: Cell type, perturbation, batch columns in `.obs`

---

## Git Practices

### Branching Strategy

**Development Branches**:
- Feature branches: Start with `claude/` prefix
- Branch naming: `claude/claude-md-{session-id}`
- CRITICAL: Branch must end with matching session ID for push to succeed

### Git Operations

**Pushing Changes**:
```bash
# Always use -u flag for new branches
git push -u origin <branch-name>

# Retry on network failures (up to 4 times with exponential backoff: 2s, 4s, 8s, 16s)
```

**Fetching/Pulling**:
```bash
# Prefer specific branches
git fetch origin <branch-name>
git pull origin <branch-name>

# Retry on network failures (same exponential backoff)
```

### Commit Message Guidelines

1. **Format**: Imperative mood, concise (50 chars or less for subject)
2. **Structure**:
   ```
   Short summary (50 chars max)

   Detailed explanation if needed.
   - Bullet points for multiple changes
   - Focus on "why" not "what"
   ```

3. **Examples** (from recent commits):
   - "Make Llama bidirectional attention opt-in"
   - "Add tests and fix small edge case"
   - "Make ones" (simple commits can be terse)

### Recent Development Focus
- Bidirectional attention for Llama models
- Batch token support
- Bug fixes for context mean and distributed sampling
- Testing improvements

---

## Dependencies

### Core ML Stack

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.7.0 | Deep learning framework |
| `transformers` | >=4.52.3 | Pre-trained models (Llama, GPT2) |
| `pytorch-lightning` | Implicit | Training orchestration |
| `scanpy` | >=1.11.2 | Single-cell RNA analysis |
| `anndata` | >=0.11.4 | Cell data structure |
| `geomloss` | >=0.2.6 | Optimal transport losses |
| `peft` | >=0.11.0 | LoRA fine-tuning |

### State Ecosystem

| Package | Version | Purpose |
|---------|---------|---------|
| `cell-load` | >=0.8.3 | Data loading and preprocessing |
| `cell-eval` | >=0.5.22 | Model evaluation framework |

### Configuration & Tooling

| Package | Version | Purpose |
|---------|---------|---------|
| `hydra-core` | >=1.3.2 | Configuration management |
| `wandb` | >=0.19.11 | Experiment tracking |
| `uv` | Latest | Package manager |
| `ruff` | >=0.11.11 | Linting and formatting |
| `vulture` | >=2.14 | Dead code detection |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `lancedb` | >=0.24.0 | Vector database (install with `uv sync --extra vectordb`) |

### Scientific Computing

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >=2.2.6 | Numerical arrays |
| `pandas` | >=2.2.3 | Data frames |
| `scipy` | >=1.15.0 | Scientific algorithms |
| `scikit-learn` | >=1.6.1 | ML utilities |

---

## Special Features and Patterns

### 1. Bidirectional Attention
Recent addition (opt-in flag) for non-causal attention in transformer backbones:
- Location: `tx/models/state_transition.py`
- Usage: Set `bidirectional=True` in model config
- Tested in: `tests/test_bidirectional_models.py`

### 2. LoRA Fine-Tuning
PEFT integration for parameter-efficient fine-tuning:
- Implementation: `tx/models/utils.py`
- Supports: Llama and GPT2 backbones
- Config: Specify LoRA parameters in model config

### 3. Optimal Transport Losses
Multiple loss functions for distribution matching:
- **Sinkhorn**: Approximate optimal transport (via geomloss)
- **Energy Distance**: Alternative to Wasserstein
- **Combined**: Weighted sum of multiple losses

### 4. Multi-Task Support
- Genetic perturbations (gene knockdown/knockout)
- Chemical perturbations (compound treatments)
- Zero-shot generalization (unseen cell types)
- Few-shot learning (limited perturbation examples)

### 5. Callback System
Extensive monitoring during training:
- **FLOPs Counting**: `tx/callbacks/cumulative_flops.py`
- **Batch Speed**: `tx/callbacks/batch_speed_monitor.py`
- **Model Utilization**: `tx/callbacks/model_flops_utilization.py`

### 6. Vector Database
Optional LanceDB integration for similarity search:
- Build: `state emb transform --lancedb <path>`
- Query: `state emb query --lancedb <path>`
- Use case: Find similar cells in large datasets

---

## Common Workflows for AI Assistants

### Adding a New Model

1. **Create model file** in `src/state/tx/models/` or `src/state/emb/nn/`
2. **Inherit from base class**:
   - TX: `PerturbationModel` from `tx/models/base.py`
   - EMB: `LightningModule` from PyTorch Lightning
3. **Implement required methods**:
   - `__init__()`: Initialize architecture
   - `forward()`: Define forward pass
   - `training_step()`: Training logic
   - `validation_step()`: Validation logic
   - `configure_optimizers()`: Optimizer setup
4. **Create config** in `src/state/configs/model/`
5. **Add tests** in `tests/`
6. **Update documentation**

### Debugging Training Issues

1. **Check logs**: W&B dashboard or console output
2. **Verify data loading**:
   ```bash
   # Test data loading separately
   python -c "from state.tx.data.dataset import *; ..."
   ```
3. **Reduce batch size** if OOM errors
4. **Enable gradient clipping** in training config
5. **Use smaller model** for quick iteration (e.g., `model=state_sm`)

### Working with AnnData

**Required structure**:
```python
import anndata as ad

# Load
adata = ad.read_h5ad("data.h5ad")

# Expected attributes
adata.X              # Gene expression (CSR matrix)
adata.obsm['X_hvg']  # Highly variable genes (if preprocessed)
adata.var            # Must contain 'gene_name' column
adata.obs            # Contains cell_type, perturbation, batch columns
```

**Preprocessing**:
```python
import scanpy as sc

# Normalize
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# Select HVGs
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata.obsm['X_hvg'] = adata[:, adata.var.highly_variable].X
```

---

## Containerization

### Singularity Support

**Build container**:
```bash
singularity build state.sif singularity.def
```

**Run container**:
```bash
# Basic usage
singularity run state.sif --help

# With GPU and volume mounts
singularity run --nv -B /data:/data state.sif emb transform \
  --model-folder /data/models/SE-600M \
  --checkpoint /data/models/SE-600M/checkpoint.ckpt \
  --input /data/input.h5ad \
  --output /data/output.h5ad
```

---

## License and Usage Restrictions

### Code License
- **License**: CC BY-NC-SA 4.0
- **File**: `LICENSE`
- **Permissions**: Non-commercial use, sharing with attribution, share-alike

### Model License
- **License**: Arc Research Institute State Model Non-Commercial License
- **Files**: `MODEL_LICENSE.md`, `MODEL_ACCEPTABLE_USE_POLICY.md`
- **Restrictions**: Non-commercial use only
- **Citation Required**: Must cite State paper for publications

---

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure package is installed
uv sync --all-extras

# Check Python version
python --version  # Should be 3.10-3.12
```

**CUDA/GPU Issues**:
```bash
# Verify PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU if needed
state tx train ... device=cpu
```

**Data Format Issues**:
- Ensure CSR matrix format: `adata.X = adata.X.tocsr()`
- Check gene_name column: `'gene_name' in adata.var.columns`
- Verify perturbation column exists: `pert_col in adata.obs.columns`

**Configuration Errors**:
- Check TOML syntax
- Verify paths exist
- Ensure cell types and perturbations match data

---

## Additional Resources

- **Paper**: https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2
- **Repository**: https://github.com/ArcInstitute/state
- **Evaluation Framework**: https://github.com/ArcInstitute/cell-eval
- **Data Loaders**: https://github.com/ArcInstitute/cell-load
- **HuggingFace Models**: https://huggingface.co/arcinstitute
- **Colab Tutorials**: See README.md for links to training/inference notebooks

---

## Quick Reference: File Locations

```
Key Directories:
├── src/state/tx/models/        # Transition model implementations
├── src/state/emb/nn/           # Embedding model implementations
├── src/state/configs/          # Hydra configurations
├── tests/                      # Test suite
└── examples/                   # Example TOML configs

Entry Points:
├── src/state/__main__.py       # Main CLI
├── src/state/_cli/_tx/         # TX commands
└── src/state/_cli/_emb/        # EMB commands

Configuration:
├── pyproject.toml              # Package config
├── ruff.toml                   # Linter config
└── src/state/configs/          # Model/training configs
```

---

**Last Updated**: 2025-11-16
**Repository State**: Based on commit 65953fa and branch structure
**Codebase Statistics**: ~17,955 lines of Python code across 80 files
