#!/usr/bin/env python
"""
Training script for ESM-based perturbation encoder.

This script trains a State Transition model that uses ESM2 protein embeddings
to encode genetic perturbations instead of one-hot encoding.

Usage:
    python train_esm_perturbation.py --config esm_config.toml --output_dir ./output

Example with overrides:
    python train_esm_perturbation.py \\
        --config esm_config.toml \\
        --output_dir ./esm_output \\
        model.kwargs.hidden_dim=512 \\
        training.batch_size=32
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ESM-based perturbation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="esm_genetic_example.toml",
        help="Path to TOML config file with dataset configurations"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="$HOME/state_esm_output",
        help="Output directory for model checkpoints and logs"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="esm_state",
        help="Model name (default: esm_state)"
    )

    parser.add_argument(
        "--embed_key",
        type=str,
        default="X_uce",
        help="Embedding key to use from the dataset (default: X_uce)"
    )

    parser.add_argument(
        "--esm_embeddings_path",
        type=str,
        default=None,
        help="Path to ESM embeddings (default: uses system default)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size (default: 8)"
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=100000,
        help="Maximum training steps (default: 100000)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="esm-perturbation",
        help="W&B project name (default: esm-perturbation)"
    )

    parser.add_argument(
        "hydra_overrides",
        nargs="*",
        help="Additional Hydra configuration overrides (e.g., model.kwargs.hidden_dim=512)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Expand environment variables in paths
    output_dir = os.path.expandvars(args.output_dir)
    config_path = os.path.expandvars(args.config)

    print("=" * 80)
    print("ESM Perturbation Encoder Training")
    print("=" * 80)
    print(f"Configuration file: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Embedding key: {args.embed_key}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.lr}")
    print("=" * 80)

    # Build Hydra overrides
    overrides = [
        f"data.kwargs.toml_config_path={config_path}",
        f"data.kwargs.embed_key={args.embed_key}",
        f"model={args.model}",
        f"output_dir={output_dir}",
        f"training.batch_size={args.batch_size}",
        f"training.max_steps={args.max_steps}",
        f"training.lr={args.lr}",
        f"use_wandb={str(args.use_wandb).lower()}",
    ]

    if args.use_wandb:
        overrides.append(f"wandb.project={args.wandb_project}")

    if args.esm_embeddings_path is not None:
        overrides.append(f"model.kwargs.esm_embeddings_path={args.esm_embeddings_path}")

    # Add any additional overrides from command line
    if args.hydra_overrides:
        overrides.extend(args.hydra_overrides)

    print("\nHydra overrides:")
    for override in overrides:
        print(f"  - {override}")
    print()

    # Initialize Hydra and run training
    config_dir = str(Path(__file__).parent / "src" / "state" / "configs")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="tx-defaults", overrides=overrides)

        # Import and run training
        from state._cli._tx._train import run_tx_train

        print("Starting training...\n")
        run_tx_train(cfg)


if __name__ == "__main__":
    main()
