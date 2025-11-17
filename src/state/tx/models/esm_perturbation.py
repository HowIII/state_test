"""
ESM-based Perturbation Encoder Model

This module extends the StateTransitionPerturbationModel to use ESM2 embeddings
as the perturbation encoder instead of one-hot encoding. This allows the model
to leverage pre-trained protein embeddings to better represent genetic perturbations.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from .state_transition import StateTransitionPerturbationModel
from .utils import build_mlp

logger = logging.getLogger(__name__)


class ESMPerturbationEncoder(nn.Module):
    """
    Encoder that uses ESM2 embeddings for genetic perturbations.

    Instead of using one-hot encoding, this encoder:
    1. Maps gene names to their ESM2 embeddings
    2. Processes the embeddings through an MLP to project to hidden_dim

    Args:
        gene_to_esm_map: Dictionary mapping gene names to ESM embedding indices
        esm_embeddings: Tensor of ESM embeddings [num_genes, esm_dim]
        hidden_dim: Output dimension for the encoder
        n_layers: Number of MLP layers
        dropout: Dropout rate
        activation: Activation function class
    """

    def __init__(
        self,
        gene_to_esm_map: Dict[str, int],
        esm_embeddings: torch.Tensor,
        hidden_dim: int,
        n_layers: int = 2,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()

        self.gene_to_esm_map = gene_to_esm_map
        self.esm_dim = esm_embeddings.shape[1]
        self.hidden_dim = hidden_dim

        # Register ESM embeddings as a buffer (not trained)
        self.register_buffer('esm_embeddings', esm_embeddings)

        # MLP to project ESM embeddings to hidden_dim
        self.projection = build_mlp(
            in_dim=self.esm_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            activation=activation,
        )

    def get_esm_embedding(self, gene_name: str) -> torch.Tensor:
        """Get ESM embedding for a single gene."""
        if gene_name in self.gene_to_esm_map:
            idx = self.gene_to_esm_map[gene_name]
            return self.esm_embeddings[idx]
        else:
            # Return zero embedding for unknown genes
            logger.warning(f"Gene {gene_name} not found in ESM embeddings, using zero vector")
            return torch.zeros(self.esm_dim, device=self.esm_embeddings.device)

    def forward(self, pert_esm_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pert_esm_indices: Tensor of ESM embedding indices [batch, seq_len, 1] or [batch, seq_len]
                             or directly ESM embeddings [batch, seq_len, esm_dim]

        Returns:
            Encoded perturbations [batch, seq_len, hidden_dim]
        """
        # If input is indices, look up embeddings
        if pert_esm_indices.shape[-1] == 1:
            pert_esm_indices = pert_esm_indices.squeeze(-1)

        if pert_esm_indices.dim() == 2:  # [batch, seq_len] - indices
            # Look up ESM embeddings
            pert_embeddings = self.esm_embeddings[pert_esm_indices.long()]
        else:  # Already embeddings [batch, seq_len, esm_dim]
            pert_embeddings = pert_esm_indices

        # Project to hidden_dim
        return self.projection(pert_embeddings)


class ESMStateTransitionModel(StateTransitionPerturbationModel):
    """
    State Transition model that uses ESM embeddings for perturbation encoding.

    This extends StateTransitionPerturbationModel by replacing the one-hot
    perturbation encoder with an ESM embedding-based encoder.

    Additional Args:
        esm_embeddings_path: Path to the ESM embeddings file (.pt or .torch)
        gene_to_esm_map_path: Path to gene name -> ESM index mapping (.pt or .torch)
        use_esm_for_pert: If True, use ESM embeddings for perturbations (default: True)
    """

    def __init__(
        self,
        esm_embeddings_path: Optional[str] = None,
        gene_to_esm_map_path: Optional[str] = None,
        use_esm_for_pert: bool = True,
        esm_encoder_layers: int = 2,
        **kwargs
    ):
        self.esm_embeddings_path = esm_embeddings_path
        self.gene_to_esm_map_path = gene_to_esm_map_path
        self.use_esm_for_pert = use_esm_for_pert
        self.esm_encoder_layers = esm_encoder_layers

        # Load ESM embeddings and mapping if provided
        self.esm_embeddings = None
        self.gene_to_esm_map = None

        if use_esm_for_pert:
            if esm_embeddings_path is None:
                # Use default from state-defaults.yaml
                esm_embeddings_path = "/large_storage/ctc/ML/data/cell/misc/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"
                logger.info(f"Using default ESM embeddings path: {esm_embeddings_path}")

            logger.info(f"Loading ESM embeddings from {esm_embeddings_path}")
            esm_data = torch.load(esm_embeddings_path, weights_only=False)

            # Handle different formats
            if isinstance(esm_data, dict):
                # Dictionary format: {gene_name: embedding}
                self.gene_to_esm_map = {gene: idx for idx, gene in enumerate(esm_data.keys())}
                self.esm_embeddings = torch.vstack(list(esm_data.values()))
                logger.info(f"Loaded ESM embeddings for {len(self.gene_to_esm_map)} genes, dim={self.esm_embeddings.shape[1]}")
            elif isinstance(esm_data, torch.Tensor):
                # Tensor format: requires separate mapping file
                self.esm_embeddings = esm_data
                if gene_to_esm_map_path is not None:
                    self.gene_to_esm_map = torch.load(gene_to_esm_map_path, weights_only=False)
                    logger.info(f"Loaded gene mapping for {len(self.gene_to_esm_map)} genes")
                else:
                    raise ValueError("gene_to_esm_map_path required when ESM embeddings are in tensor format")

        # Initialize parent class
        super().__init__(**kwargs)

    def _build_networks(self, lora_cfg=None):
        """
        Override to build ESM-based perturbation encoder.
        """
        # Build ESM perturbation encoder if using ESM
        if self.use_esm_for_pert and self.esm_embeddings is not None:
            logger.info("Building ESM-based perturbation encoder")
            self.pert_encoder = ESMPerturbationEncoder(
                gene_to_esm_map=self.gene_to_esm_map,
                esm_embeddings=self.esm_embeddings,
                hidden_dim=self.hidden_dim,
                n_layers=self.esm_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            # Fall back to standard one-hot encoder
            logger.info("Building standard MLP perturbation encoder")
            self.pert_encoder = build_mlp(
                in_dim=self.pert_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )

        # Build the rest of the networks using parent class
        # Basal encoder
        if self.use_basal_projection:
            self.basal_encoder = build_mlp(
                in_dim=self.input_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.basal_encoder = nn.Linear(self.input_dim, self.hidden_dim)

        # Import here to avoid circular dependency
        from .utils import get_transformer_backbone, apply_lora

        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            self.transformer_backbone_key,
            self.transformer_backbone_kwargs,
        )

        # Optionally wrap backbone with LoRA adapters
        if lora_cfg and lora_cfg.get("enable", False):
            self.transformer_backbone = apply_lora(
                self.transformer_backbone,
                self.transformer_backbone_key,
                lora_cfg,
            )

        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        if self.output_space == "all":
            self.final_down_then_up = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 8),
                nn.GELU(),
                nn.Linear(self.output_dim // 8, self.output_dim),
            )
