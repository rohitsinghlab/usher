#!/usr/bin/env python3
"""
Example usage of the SAMEv2 spatial alignment toolkit.

This script demonstrates how to:
1. Load spatial transcriptomics data
2. Run alignment with different strategies
3. Save and load trained models
4. Apply models to new data
"""

import os
import sys
import torch
import anndata as ad
import numpy as np
import pandas as pd

# Add src to path
sys.path.append('src')

from spot_alignment_sinkhorn_features_only import align_features_sinkhorn
from model_utils import save_alignment_model, load_alignment_model, apply_alignment_model


def main():
    # =========================================================================
    # 1. Load your data
    # =========================================================================
    print("Loading data...")

    # Replace with your actual data paths
    # adata_source = ad.read_h5ad('path/to/source.h5ad')
    # adata_target = ad.read_h5ad('path/to/target.h5ad')

    # For testing, you can create synthetic data:
    n_cells_source = 1000
    n_cells_target = 1200
    n_genes = 50

    # Create synthetic source data
    X_source = np.random.randn(n_cells_source, n_genes).astype(np.float32)
    coords_source = np.random.rand(n_cells_source, 2) * 100

    adata_source = ad.AnnData(
        X=X_source,
        obs=pd.DataFrame(index=[f'cell_s_{i}' for i in range(n_cells_source)]),
        var=pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    )
    adata_source.obsm['X_spatial'] = coords_source

    # Create synthetic target data (with different distribution)
    X_target = np.random.randn(n_cells_target, n_genes).astype(np.float32) * 1.5 + 0.5
    coords_target = np.random.rand(n_cells_target, 2) * 100

    adata_target = ad.AnnData(
        X=X_target,
        obs=pd.DataFrame(index=[f'cell_t_{i}' for i in range(n_cells_target)]),
        var=pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    )
    adata_target.obsm['X_spatial'] = coords_target

    print(f"Source data: {adata_source.shape}")
    print(f"Target data: {adata_target.shape}")

    # =========================================================================
    # 2. Run spatial alignment
    # =========================================================================
    print("\nRunning spatial alignment...")

    # Choose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Run alignment with spatial windowing
    model, mapping, concat_adata, T_full, feature_mean, feature_std = align_features_sinkhorn(
        adata_source,
        adata_target,
        sampling_strategy='spatial',  # Use spatial windowing
        e_step_method='fgw',          # Fused Gromov-Wasserstein
        metric='cosine',              # Cosine distance
        n_iters=5,                    # Number of E-M iterations
        gamma=0.8,                    # Weight for features vs spatial
        epsilon=0.1,                  # Entropic regularization
        lr=0.001,                     # Learning rate
        steps_per_iter=50,            # Training steps per M-step
        lambda_cross=1.0,             # Cross-domain loss weight
        lambda_struct=0.1,            # Structure preservation weight
        lambda_var=0.5,               # Distribution matching weight (SWD + MMD)
        device=device,
        verbose=True
    )

    print(f"\nAlignment complete!")
    print(f"Mapping shape: {mapping.shape}")
    print(f"Transport matrix shape: {T_full.shape if T_full is not None else 'None'}")

    # =========================================================================
    # 3. Save the trained model
    # =========================================================================
    print("\nSaving model...")

    model_path = 'alignment_model.pt'
    save_alignment_model(
        model,
        model_path,
        feature_mean=feature_mean,
        feature_std=feature_std,
        gene_names=adata_source.var_names.tolist()
    )
    print(f"Model saved to {model_path}")

    # =========================================================================
    # 4. Load and apply to new data
    # =========================================================================
    print("\nLoading model and applying to new data...")

    # Load the model
    loaded_model, loaded_mean, loaded_std, gene_names = load_alignment_model(
        model_path,
        device=device
    )

    # Create new test data (must have same genes/features)
    X_new = np.random.randn(500, n_genes).astype(np.float32)
    adata_new = ad.AnnData(
        X=X_new,
        obs=pd.DataFrame(index=[f'cell_new_{i}' for i in range(500)]),
        var=pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    )

    # Apply the trained model
    adata_transformed = apply_alignment_model(
        loaded_model,
        adata_new,
        feature_mean=loaded_mean,
        feature_std=loaded_std,
        normalize_output=False,
        device=device
    )

    print(f"Transformed data shape: {adata_transformed.shape}")

    # =========================================================================
    # 5. Analyze results
    # =========================================================================
    print("\nAnalyzing alignment results...")

    # Check mapping statistics
    if mapping is not None:
        print(f"Number of mapped cells: {len(mapping)}")
        print(f"Unique targets mapped: {mapping['target_idx'].nunique()}")

        # Check confidence scores
        if 'score' in mapping.columns:
            print(f"Mean mapping confidence: {mapping['score'].mean():.4f}")
            print(f"Confidence range: [{mapping['score'].min():.4f}, {mapping['score'].max():.4f}]")

    # Clean up
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"\nCleaned up: removed {model_path}")

    print("\nExample complete!")


if __name__ == "__main__":
    main()