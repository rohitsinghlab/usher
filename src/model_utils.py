"""
Model utilities for saving, loading, and applying alignment models.

This module provides utilities to save trained alignment models to disk and apply them to new data.

Usage Example
-------------
# 1. Train and save model
from spot_alignment_sinkhorn_features_only import align_features_sinkhorn
from model_utils import save_alignment_model, load_alignment_model, apply_alignment_model

model, mapping, concat_adata, T_full = align_features_sinkhorn(
    adata_source, adata_target,
    e_step_method='fgw',
    ...
)
save_alignment_model(model, 'alignment_model.pt')

# 2. Load and apply to new data
model = load_alignment_model('alignment_model.pt', device='cuda')

# IMPORTANT: New data MUST be in same PCA/feature space as training data!
adata_new_transformed = apply_alignment_model(
    model,
    adata_new_pca,  # Must have same n_features as training
    normalize_output=False,  # Set True for L2-normalized output
    device='cuda'
)

Notes
-----
- Model expects RAW features (same preprocessing as training)
- If training used PCA, new data must be projected to SAME PCA space
- L2-normalization is optional and done AFTER transformation
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import anndata as ad


class FeatureTransform(nn.Module):
    """Feature transform with optional residual connection.

    - If hidden_dim is None: Simple linear transformation W(x) = Ax + b
    - If hidden_dim is set: Non-linear MLP with residual connection (when input_dim == output_dim)

    Residual connection learns the DIFFERENCE: x + ΔW(x) = target
    This is better for batch correction where source and target are similar.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_residual = (hidden_dim is not None) and (input_dim == output_dim)

        if hidden_dim is None:
            # Simple linear transformation
            self.net = nn.Linear(input_dim, output_dim, bias=True)
        else:
            # Non-linear MLP: input -> hidden -> output
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            # Residual connection: learn the difference/correction
            # x + ΔW(x) where ΔW is the learned correction
            return x + self.net(x)
        else:
            # Standard: learn full transformation
            return self.net(x)

    def init_identity(self):
        """
        Initialize linear layer weights to identity matrix (diagonal ones) and bias to zero.

        For linear layer:
        - If input_dim == output_dim: Initialize W as identity matrix (diagonal ones)
        - If input_dim != output_dim: Initialize W as identity for min(input_dim, output_dim), rest as zeros
        - Bias is always initialized to zero

        For MLP:
        - Initialize the LAST linear layer to identity (if possible)
        - This makes the initial transformation close to identity

        This is useful when source and target are already similar and we want to learn
        small corrections rather than a completely new transformation.
        """
        with torch.no_grad():
            if isinstance(self.net, nn.Linear):
                # Simple linear layer
                weight = self.net.weight  # (output_dim, input_dim)
                bias = self.net.bias

                # Initialize to identity (diagonal ones)
                nn.init.zeros_(weight)
                min_dim = min(self.input_dim, self.output_dim)
                for i in range(min_dim):
                    weight[i, i] = 1.0

                # Zero bias
                nn.init.zeros_(bias)

                logging.info(f"Initialized linear layer to identity (W: {weight.shape}, diagonal ones for min_dim={min_dim})")

            elif isinstance(self.net, nn.Sequential):
                # MLP: Initialize the LAST linear layer to identity
                last_linear = None
                for m in reversed(list(self.net.modules())):
                    if isinstance(m, nn.Linear):
                        last_linear = m
                        break

                if last_linear is not None:
                    weight = last_linear.weight  # (output_dim, hidden_dim)
                    bias = last_linear.bias

                    # Initialize to identity (if square) or zeros
                    nn.init.zeros_(weight)
                    min_dim = min(weight.shape[0], weight.shape[1])
                    for i in range(min_dim):
                        weight[i, i] = 1.0

                    # Zero bias
                    nn.init.zeros_(bias)

                    logging.info(f"Initialized MLP last layer to identity (W: {weight.shape}, diagonal ones for min_dim={min_dim})")
                else:
                    logging.warning("No linear layer found in MLP for identity initialization")

    def init_random(self):
        """Initialize with Xavier/Kaiming random initialization.

        Better for when source and target have different distributions,
        such as in spatial alignment tasks.
        """
        with torch.no_grad():
            if isinstance(self.net, nn.Linear):
                # Xavier uniform initialization
                nn.init.xavier_uniform_(self.net.weight)
                nn.init.zeros_(self.net.bias)
                logging.info(f"Initialized linear layer with Xavier uniform: {self.net.weight.shape}")

            elif isinstance(self.net, nn.Sequential):
                # Initialize all linear layers in MLP
                for m in self.net.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                logging.info(f"Initialized MLP layers with Xavier uniform")


def _to_dense_float32(x) -> np.ndarray:
    """Convert sparse or dense array to dense float32."""
    if hasattr(x, "toarray"):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def save_alignment_model(
    model: FeatureTransform,
    save_path: str,
    feature_mean: Optional[torch.Tensor] = None,
    feature_std: Optional[torch.Tensor] = None,
    gene_names: Optional[list] = None
):
    """
    Save trained alignment model to disk.

    Parameters
    ----------
    model : FeatureTransform
        Trained model from align_features_sinkhorn
    save_path : str
        Path to save the model (e.g., 'model.pt')
    feature_mean : torch.Tensor, optional
        Mean per feature column used for standardization (from align_features_sinkhorn)
        None if cosine metric was used
    feature_std : torch.Tensor, optional
        Std per feature column used for standardization (from align_features_sinkhorn)
        None if cosine metric was used
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'input_dim': model.input_dim,
        'output_dim': model.output_dim,
        'use_residual': model.use_residual,
        'feature_mean': feature_mean.cpu() if feature_mean is not None else None,
        'feature_std': feature_std.cpu() if feature_std is not None else None,
        'gene_names': gene_names,
    }

    torch.save(save_dict, save_path)

    if feature_mean is not None and feature_std is not None:
        logging.info(f"Model saved to {save_path} (with feature scaler)")
    else:
        logging.info(f"Model saved to {save_path} (no scaler - cosine metric)")


def load_alignment_model(load_path: str, device: Optional[str] = None, hidden_dim: Optional[int] = None):
    """
    Load trained alignment model from disk.

    Parameters
    ----------
    load_path : str
        Path to saved model (e.g., 'model.pt')
    device : str, optional
        Device to load model to ('cuda' or 'cpu')

    Returns
    -------
    model : FeatureTransform
        Loaded model ready for inference
    feature_mean : torch.Tensor or None
        Mean per feature column for standardization (None if not used)
    feature_std : torch.Tensor or None
        Std per feature column for standardization (None if not used)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(load_path, map_location=device)

    # Reconstruct model architecture
    model = FeatureTransform(
        input_dim=checkpoint['input_dim'],
        output_dim=checkpoint['output_dim'],
        hidden_dim=hidden_dim  # Will be inferred from state_dict
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load scaler (if present)
    feature_mean = checkpoint.get('feature_mean', None)
    feature_std = checkpoint.get('feature_std', None)
    gene_names = checkpoint.get('gene_names', None)

    if feature_mean is not None and feature_std is not None:
        feature_mean = feature_mean.to(device)
        feature_std = feature_std.to(device)
        logging.info(f"Model loaded from {load_path} to {device} (with feature scaler)")
    else:
        logging.info(f"Model loaded from {load_path} to {device} (no scaler - cosine metric)")

    return model, feature_mean, feature_std, gene_names


def apply_alignment_model(
    model: FeatureTransform,
    adata_new: ad.AnnData,
    feature_mean: Optional[torch.Tensor] = None,
    feature_std: Optional[torch.Tensor] = None,
    normalize_output: bool = False,
    device: Optional[str] = None,
    gene_names: Optional[list] = None
) -> ad.AnnData:
    """
    Apply trained alignment model to new data.

    Parameters
    ----------
    model : FeatureTransform
        Trained model from align_features_sinkhorn or load_alignment_model
    adata_new : ad.AnnData
        New data to transform (MUST be in same space as training data)
        - Same PCA/feature space (same number of features)
        - Same preprocessing as training data
    feature_mean : torch.Tensor, optional
        Mean per feature column for standardization (from load_alignment_model)
        None if model was trained with cosine metric
    feature_std : torch.Tensor, optional
        Std per feature column for standardization (from load_alignment_model)
        None if model was trained with cosine metric
    normalize_output : bool
        Whether to L2-normalize the output (default: False)
        Set to True if you need normalized features for downstream distance computation
    device : str, optional
        Device to use ('cuda' or 'cpu')

    Returns
    -------
    adata_transformed : ad.AnnData
        AnnData with transformed features in .X

    Notes
    -----
    - Input data (adata_new.X) should be RAW features (same as training)
    - If training data was PCA, new data should be projected to same PCs
    - If feature_mean/feature_std are provided (euclidean metric):
      1. Input is standardized
      2. Model is applied in standardized space
      3. Output is unstandardized back to original scale
    - If feature_mean/feature_std are None (cosine metric):
      Model is applied directly without standardization

    Example
    -------
    >>> # Load model with scaler
    >>> model, feature_mean, feature_std = load_alignment_model('alignment_model.pt', device='cuda')
    >>>
    >>> # Apply to new data (must be in same PCA space!)
    >>> adata_new_transformed = apply_alignment_model(
    ...     model,
    ...     adata_new_pca,
    ...     feature_mean=feature_mean,
    ...     feature_std=feature_std,
    ...     normalize_output=False,
    ...     device='cuda'
    ... )
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    # Move scaler to device if provided
    if feature_mean is not None:
        feature_mean = feature_mean.to(device)
    if feature_std is not None:
        feature_std = feature_std.to(device)

    # Load and convert data
    X_new = _to_dense_float32(adata_new.X)
    features_new = torch.from_numpy(X_new).to(device)

    # Check dimensions
    if features_new.shape[1] != model.input_dim:
        raise ValueError(
            f"Input dimension mismatch: model expects {model.input_dim} features, "
            f"but data has {features_new.shape[1]} features. "
            f"Ensure new data is in the same PCA/feature space as training data."
        )

    # Transform with optional standardization
    with torch.no_grad():
        if feature_mean is not None and feature_std is not None:
            # Euclidean metric: standardize → transform → unstandardize
            from m_step_utils import standardize_features, unstandardize_features

            features_std = standardize_features(features_new, feature_mean, feature_std)
            features_transformed_std = model(features_std)
            features_transformed = unstandardize_features(features_transformed_std, feature_mean, feature_std)

            logging.info("Applied model with feature standardization (euclidean metric)")
        else:
            # Cosine metric: direct transformation
            features_transformed = model(features_new)
            logging.info("Applied model without standardization (cosine metric)")

        # Optional L2 normalization
        if normalize_output:
            features_transformed = F.normalize(features_transformed, p=2, dim=1)

    # Create output AnnData
    adata_transformed = ad.AnnData(
        X=features_transformed.cpu().numpy(),
        obs=adata_new.obs.copy(),
        var=pd.DataFrame(index=[f'feature_{i}' for i in range(features_transformed.shape[1])])
    )

    logging.info(f"Transformed {features_new.shape[0]} cells: {features_new.shape[1]} → {features_transformed.shape[1]} features")
    if normalize_output:
        logging.info("  Output L2-normalized")

    return adata_transformed
