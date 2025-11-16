"""
Utility functions for working with ESM embeddings in perturbation datasets.

This module provides functions to:
1. Load ESM embeddings
2. Create mappings from gene names to ESM indices
3. Build perturbation maps that use ESM indices instead of one-hot encodings
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_esm_embeddings(
    embeddings_path: Union[str, Path],
    gene_to_idx_path: Optional[Union[str, Path]] = None,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Load ESM embeddings and gene-to-index mapping.

    Args:
        embeddings_path: Path to ESM embeddings file (.pt or .torch)
            Can be either:
            - Dictionary format: {gene_name: embedding_tensor}
            - Tensor format: [num_genes, embedding_dim]
        gene_to_idx_path: Optional path to gene->index mapping (required for tensor format)

    Returns:
        Tuple of (embeddings_tensor, gene_to_idx_dict)
    """
    logger.info(f"Loading ESM embeddings from {embeddings_path}")

    esm_data = torch.load(embeddings_path, weights_only=False)

    if isinstance(esm_data, dict):
        # Dictionary format: {gene_name: embedding}
        gene_to_idx = {gene: idx for idx, gene in enumerate(esm_data.keys())}
        embeddings = torch.vstack(list(esm_data.values()))
        logger.info(f"Loaded {len(gene_to_idx)} genes with embedding dim {embeddings.shape[1]}")

    elif isinstance(esm_data, torch.Tensor):
        # Tensor format: requires separate mapping file
        embeddings = esm_data

        if gene_to_idx_path is None:
            raise ValueError(
                "gene_to_idx_path is required when embeddings_path contains a tensor. "
                "Provide a mapping file or use a dictionary format embeddings file."
            )

        logger.info(f"Loading gene-to-index mapping from {gene_to_idx_path}")
        gene_to_idx = torch.load(gene_to_idx_path, weights_only=False)

        if not isinstance(gene_to_idx, dict):
            raise ValueError(
                f"Expected gene_to_idx to be a dictionary, got {type(gene_to_idx)}"
            )

        logger.info(f"Loaded mapping for {len(gene_to_idx)} genes, embeddings shape: {embeddings.shape}")

    else:
        raise ValueError(
            f"Unexpected ESM embeddings format: {type(esm_data)}. "
            "Expected dict or torch.Tensor"
        )

    return embeddings, gene_to_idx


def build_esm_pert_map(
    perturbation_names: List[str],
    gene_to_esm_idx: Dict[str, int],
    esm_embeddings: torch.Tensor,
    use_embeddings: bool = False,
) -> Dict[str, Union[int, torch.Tensor]]:
    """
    Build a perturbation map that maps perturbation names to ESM indices or embeddings.

    Args:
        perturbation_names: List of perturbation names (e.g., gene names)
        gene_to_esm_idx: Dictionary mapping gene names to ESM embedding indices
        esm_embeddings: ESM embedding tensor [num_genes, esm_dim]
        use_embeddings: If True, map to embeddings directly; if False, map to indices

    Returns:
        Dictionary mapping perturbation names to ESM indices (int) or embeddings (tensor)
    """
    pert_map = {}
    missing_genes = []

    for pert_name in perturbation_names:
        # Handle control perturbations (DMSO, non-targeting, etc.)
        if pert_name.lower() in ['dmso', 'non-targeting', 'ctrl', 'control', 'dmso_tf']:
            # Use zero embedding/index for control
            if use_embeddings:
                pert_map[pert_name] = torch.zeros(esm_embeddings.shape[1])
            else:
                pert_map[pert_name] = 0  # Will be handled specially
        elif pert_name in gene_to_esm_idx:
            idx = gene_to_esm_idx[pert_name]
            if use_embeddings:
                pert_map[pert_name] = esm_embeddings[idx]
            else:
                pert_map[pert_name] = idx
        else:
            # Gene not found in ESM embeddings
            missing_genes.append(pert_name)
            if use_embeddings:
                pert_map[pert_name] = torch.zeros(esm_embeddings.shape[1])
            else:
                pert_map[pert_name] = -1  # Sentinel value

    if missing_genes:
        logger.warning(
            f"Found {len(missing_genes)} perturbations not in ESM embeddings: "
            f"{missing_genes[:10]}{'...' if len(missing_genes) > 10 else ''}"
        )

    logger.info(f"Built ESM perturbation map for {len(pert_map)} perturbations")

    return pert_map


def create_esm_pert_map_from_datasets(
    dataset_configs: List[Dict],
    esm_embeddings_path: str,
    gene_to_idx_path: Optional[str] = None,
    use_embeddings: bool = False,
) -> Tuple[Dict[str, Union[int, torch.Tensor]], torch.Tensor, Dict[str, int]]:
    """
    Create ESM perturbation map from dataset configurations.

    Args:
        dataset_configs: List of dataset configuration dictionaries
        esm_embeddings_path: Path to ESM embeddings
        gene_to_idx_path: Optional path to gene->index mapping
        use_embeddings: If True, return embeddings; if False, return indices

    Returns:
        Tuple of (pert_map, esm_embeddings, gene_to_idx)
    """
    # Load ESM embeddings
    esm_embeddings, gene_to_idx = load_esm_embeddings(
        esm_embeddings_path,
        gene_to_idx_path,
    )

    # Collect all unique perturbation names from datasets
    # This would need to be implemented based on your dataset structure
    # For now, return the components needed to build the map
    logger.info("ESM embeddings and mappings loaded. Build perturbation map using build_esm_pert_map()")

    return {}, esm_embeddings, gene_to_idx


def get_default_esm_config() -> Dict[str, str]:
    """
    Get default ESM configuration paths.

    Returns:
        Dictionary with 'embeddings_path' and optional 'gene_to_idx_path'
    """
    return {
        'embeddings_path': '/large_storage/ctc/ML/data/cell/misc/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
        'gene_to_idx_path': None,  # Not needed for dict format
    }
