#!/usr/bin/env python
"""
Test script to verify both celltype and spatial sampling strategies work correctly.
"""

import numpy as np
import anndata as ad
import torch
import logging
from spot_alignment_sinkhorn_features_only import align_features_sinkhorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_data(n_cells=500, n_genes=100, n_celltypes=5, add_spatial=True):
    """Create synthetic test data with cell types and optionally spatial coordinates."""

    # Random expression matrix
    X = np.random.randn(n_cells, n_genes).astype(np.float32)
    X = np.abs(X)  # Make positive

    # Create AnnData object
    adata = ad.AnnData(X)

    # Add cell type annotations
    cell_types = np.random.choice([f"CellType_{i}" for i in range(n_celltypes)], n_cells)
    adata.obs['celltype'] = cell_types

    # Add cell type probabilities
    celltype_probs = np.zeros((n_cells, n_celltypes), dtype=np.float32)
    for i, ct in enumerate(adata.obs['celltype']):
        ct_idx = int(ct.split('_')[1])
        celltype_probs[i, ct_idx] = 0.8  # High probability for true type
        # Add some noise to other types
        other_probs = np.random.dirichlet(np.ones(n_celltypes - 1)) * 0.2
        other_indices = [j for j in range(n_celltypes) if j != ct_idx]
        for j, idx in enumerate(other_indices):
            celltype_probs[i, idx] = other_probs[j]

    adata.obsm['X_celltype_probs'] = celltype_probs

    # Add UMAP embedding for geosketch
    adata.obsm['X_umap'] = np.random.randn(n_cells, 2).astype(np.float32)

    # Add spatial coordinates if requested
    if add_spatial:
        # Create grid-like spatial coordinates with some noise
        grid_size = int(np.ceil(np.sqrt(n_cells)))
        x_coords = np.repeat(np.arange(grid_size), grid_size)[:n_cells]
        y_coords = np.tile(np.arange(grid_size), grid_size)[:n_cells]

        # Add noise
        x_coords = x_coords.astype(np.float32) + np.random.randn(len(x_coords)) * 0.2
        y_coords = y_coords.astype(np.float32) + np.random.randn(len(y_coords)) * 0.2

        spatial_coords = np.column_stack([x_coords, y_coords]).astype(np.float32)
        adata.obsm['X_spatial'] = spatial_coords

    return adata


def test_celltype_sampling():
    """Test celltype-based sampling strategy."""
    print("\n" + "="*80)
    print("Testing CELLTYPE sampling strategy")
    print("="*80)

    # Create test data
    adata_source = create_test_data(n_cells=300, n_genes=50, add_spatial=False)
    adata_target = create_test_data(n_cells=400, n_genes=50, add_spatial=False)

    print(f"Source data: {adata_source.shape}")
    print(f"Target data: {adata_target.shape}")

    try:
        # Run alignment with celltype sampling
        model, mapping, concat_adata, T_full, feature_mean, feature_std = align_features_sinkhorn(
            adata_source, adata_target,
            sampling_strategy='celltype',
            e_step_method='ot',
            m_step_method='transfer',
            sketch_size=100,
            use_stratified_pairing=False,
            cell_type_col='celltype',
            celltype_probs_layer='X_celltype_probs',
            n_iters=3,
            epsilon=0.1,
            debug_plots_path=None
        )

        print("✓ Celltype sampling completed successfully!")
        print(f"  Mapping shape: {mapping.shape}")
        print(f"  Model: {type(model)}")
        return True

    except Exception as e:
        print(f"✗ Celltype sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spatial_sampling():
    """Test spatial windowing strategy."""
    print("\n" + "="*80)
    print("Testing SPATIAL sampling strategy")
    print("="*80)

    # Create test data with spatial coordinates
    adata_source = create_test_data(n_cells=400, n_genes=50, add_spatial=True)
    adata_target = create_test_data(n_cells=500, n_genes=50, add_spatial=True)

    print(f"Source data: {adata_source.shape}")
    print(f"Target data: {adata_target.shape}")
    print(f"Spatial coords shape: {adata_source.obsm['X_spatial'].shape}")

    try:
        # Run alignment with spatial sampling
        model, mapping, concat_adata, T_full, feature_mean, feature_std = align_features_sinkhorn(
            adata_source, adata_target,
            sampling_strategy='spatial',
            e_step_method='ot',
            m_step_method='transfer',
            spatial_key='X_spatial',
            window_height=None,  # Auto-compute
            window_width=None,   # Auto-compute
            window_overlap=0.1,
            spatial_knn=20,
            n_windows_target=10,
            n_iters=3,
            epsilon=0.1,
            debug_plots_path=None
        )

        print("✓ Spatial sampling completed successfully!")
        print(f"  Mapping shape: {mapping.shape}")
        print(f"  Model: {type(model)}")
        return True

    except Exception as e:
        print(f"✗ Spatial sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test that old code still works (backward compatibility)."""
    print("\n" + "="*80)
    print("Testing BACKWARD COMPATIBILITY (default celltype sampling)")
    print("="*80)

    # Create test data
    adata_source = create_test_data(n_cells=200, n_genes=50, add_spatial=False)
    adata_target = create_test_data(n_cells=250, n_genes=50, add_spatial=False)

    print(f"Source data: {adata_source.shape}")
    print(f"Target data: {adata_target.shape}")

    try:
        # Run alignment WITHOUT specifying sampling_strategy (should default to celltype)
        model, mapping, concat_adata, T_full, feature_mean, feature_std = align_features_sinkhorn(
            adata_source, adata_target,
            # sampling_strategy not specified - should default to 'celltype'
            e_step_method='ot',
            m_step_method='transfer',
            sketch_size=100,
            use_stratified_pairing=True,
            cell_type_col='celltype',
            n_iters=2,
            epsilon=0.1,
            debug_plots_path=None
        )

        print("✓ Backward compatibility test passed!")
        print(f"  Mapping shape: {mapping.shape}")
        return True

    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing sampling strategies for spatial alignment")
    print("=" * 80)

    results = []

    # Test celltype sampling
    results.append(("Celltype Sampling", test_celltype_sampling()))

    # Test spatial sampling
    results.append(("Spatial Sampling", test_spatial_sampling()))

    # Test backward compatibility
    results.append(("Backward Compatibility", test_backward_compatibility()))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s}: {status}")
        if not passed:
            all_passed = False

    print("="*80)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - Please review errors above")

    exit(0 if all_passed else 1)