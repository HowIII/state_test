#!/usr/bin/env python
"""
Train State model with ESM2 perturbation features (following Colab notebook approach).

This script follows the approach from the Virtual Cell Challenge Colab notebook:
https://colab.research.google.com/drive/1QKOtYP7bMpdgDJEipDxaJqOchv7oQ-_l

Instead of modifying the model, we provide pre-computed ESM embeddings to the
data module, which uses them instead of one-hot encoding.

Usage:
    # Step 1: Create ESM perturbation features
    python create_esm_pert_features.py \\
        --from-toml config.toml \\
        --output ESM2_pert_features.pt

    # Step 2: Train with ESM features
    python train_with_esm_features.py \\
        --config config.toml \\
        --pert-features ESM2_pert_features.pt \\
        --output-dir ./output

Example (matching Colab notebook):
    python train_with_esm_features.py \\
        --config competition_support_set/starter.toml \\
        --pert-features competition_support_set/ESM2_pert_features.pt \\
        --pert-col target_gene \\
        --control-pert non-targeting \\
        --model state_sm \\
        --max-steps 40000
"""

import argparse
import os
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train State with ESM perturbation features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to TOML dataset configuration file"
    )

    parser.add_argument(
        "--pert-features",
        type=str,
        required=True,
        help="Path to ESM2 perturbation features file (.pt)"
    )

    # Data configuration
    parser.add_argument(
        "--pert-col",
        type=str,
        default="gene",
        help="Column name for perturbations (default: gene)"
    )

    parser.add_argument(
        "--control-pert",
        type=str,
        default="non-targeting",
        help="Control perturbation name (default: non-targeting)"
    )

    parser.add_argument(
        "--cell-type-key",
        type=str,
        default="cell_type",
        help="Column name for cell types (default: cell_type)"
    )

    parser.add_argument(
        "--batch-col",
        type=str,
        default="gem_group",
        help="Column name for batch (default: gem_group)"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loading workers (default: 8)"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="state_sm",
        help="Model name (default: state_sm). Options: state, state_sm, state_lg"
    )

    # Training configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for checkpoints and logs (default: output)"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="esm_run",
        help="Run name for logging (default: esm_run)"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=40000,
        help="Maximum training steps (default: 40000)"
    )

    parser.add_argument(
        "--ckpt-every-n-steps",
        type=int,
        default=20000,
        help="Save checkpoint every N steps (default: 20000)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: from model config)"
    )

    # W&B configuration
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="state-esm",
        help="W&B project name (default: state-esm)"
    )

    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="",
        help="W&B entity name"
    )

    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="+",
        default=[],
        help="W&B tags (space-separated)"
    )

    # Additional Hydra overrides
    parser.add_argument(
        "hydra_overrides",
        nargs="*",
        help="Additional Hydra overrides (e.g., training.lr=1e-4)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Verify files exist
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    if not Path(args.pert_features).exists():
        print(f"Error: Perturbation features file not found: {args.pert_features}")
        sys.exit(1)

    print("=" * 80)
    print("STATE Training with ESM2 Perturbation Features")
    print("=" * 80)
    print(f"Configuration file:       {args.config}")
    print(f"Perturbation features:    {args.pert_features}")
    print(f"Model:                    {args.model}")
    print(f"Output directory:         {args.output_dir}")
    print(f"Run name:                 {args.name}")
    print(f"Max steps:                {args.max_steps}")
    print(f"Checkpoint frequency:     {args.ckpt_every_n_steps}")
    print(f"Perturbation column:      {args.pert_col}")
    print(f"Control perturbation:     {args.control_pert}")
    print("=" * 80)

    # Build command for state tx train
    cmd_parts = ["state", "tx", "train"]

    # Data configuration
    cmd_parts.extend([
        f"data.kwargs.toml_config_path={args.config}",
        f"data.kwargs.perturbation_features_file={args.pert_features}",
        f"data.kwargs.num_workers={args.num_workers}",
        f"data.kwargs.batch_col={args.batch_col}",
        f"data.kwargs.pert_col={args.pert_col}",
        f"data.kwargs.cell_type_key={args.cell_type_key}",
        f"data.kwargs.control_pert={args.control_pert}",
    ])

    # Training configuration
    cmd_parts.extend([
        f"training.max_steps={args.max_steps}",
        f"training.ckpt_every_n_steps={args.ckpt_every_n_steps}",
    ])

    if args.batch_size is not None:
        cmd_parts.append(f"training.batch_size={args.batch_size}")

    # Model configuration
    cmd_parts.append(f"model={args.model}")

    # Output configuration
    cmd_parts.extend([
        f"output_dir={args.output_dir}",
        f"name={args.name}",
    ])

    # W&B configuration
    if args.use_wandb:
        cmd_parts.append("use_wandb=true")
        cmd_parts.append(f"wandb.project={args.wandb_project}")

        if args.wandb_entity:
            cmd_parts.append(f"wandb.entity={args.wandb_entity}")

        if args.wandb_tags:
            tags_str = "[" + ",".join(args.wandb_tags) + "]"
            cmd_parts.append(f"wandb.tags={tags_str}")
    else:
        cmd_parts.append("use_wandb=false")

    # Add any additional overrides
    if args.hydra_overrides:
        cmd_parts.extend(args.hydra_overrides)

    # Print command
    print("\nExecuting command:")
    print(" \\\n  ".join(cmd_parts))
    print()

    # Execute
    import subprocess
    result = subprocess.run(cmd_parts)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
