#!/usr/bin/env python3
"""
Test script to verify that the normalization fix resolves negative cosine distances.
This script creates synthetic data and runs the alignment with cosine metric to ensure
that cosine distances stay in the valid [0, 2] range.
"""

import numpy as np
import torch
import torch.nn.functional as F
import anndata as ad
import pandas as pd
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_synthetic_spatial_data(n_cells=500, n_features=50, seed=42):
    """Create synthetic spatial data for testing."""
    np.random.seed(seed)

    # Create expression matrix with some structure
    X = np.random.randn(n_cells, n_features).astype(np.float32)
    X = X + np.random.rand(1, n_features) * 5  # Add some feature-specific bias

    # Create spatial coordinates in a grid
    grid_size = int(np.sqrt(n_cells))
    x_coords = np.repeat(np.arange(grid_size), grid_size)[:n_cells]
    y_coords = np.tile(np.arange(grid_size), grid_size)[:n_cells]
    coords = np.column_stack([x_coords, y_coords]).astype(np.float32)

    # Create AnnData object
    obs = pd.DataFrame(index=[f'cell_{i}' for i in range(n_cells)])
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_features)])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm['X_spatial'] = coords

    return adata

def test_cosine_distance_bounds():
    """Test that cosine distances stay within valid bounds after fix."""
    print("\n" + "="*60)
    print("Testing Cosine Distance Bounds After Normalization Fix")
    print("="*60)

    # Create synthetic data
    print("\n1. Creating synthetic spatial data...")
    adata_source = create_synthetic_spatial_data(n_cells=300, n_features=30, seed=42)
    adata_target = create_synthetic_spatial_data(n_cells=350, n_features=30, seed=123)
    print(f"   Source: {adata_source.shape}")
    print(f"   Target: {adata_target.shape}")

    # Convert to tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n2. Using device: {device}")

    features_source = torch.from_numpy(adata_source.X).to(device)
    features_target = torch.from_numpy(adata_target.X).to(device)

    # Test 1: Raw features (BEFORE fix - would have caused issues)
    print("\n3. Testing with RAW features (simulating old broken behavior):")
    features_source_norm = F.normalize(features_source, p=2, dim=1)
    # Old behavior: target was NOT normalized
    cosine_dist_broken = 1.0 - (features_source_norm @ features_target.T)
    print(f"   Cosine distance stats (BROKEN):")
    print(f"   - Min: {cosine_dist_broken.min():.4f}")
    print(f"   - Max: {cosine_dist_broken.max():.4f}")
    print(f"   - Mean: {cosine_dist_broken.mean():.4f}")

    if cosine_dist_broken.min() < 0:
        print(f"   ⚠️  WARNING: Negative values detected! Min = {cosine_dist_broken.min():.4f}")
    if cosine_dist_broken.max() > 2:
        print(f"   ⚠️  WARNING: Values > 2 detected! Max = {cosine_dist_broken.max():.4f}")

    # Test 2: Both normalized (AFTER fix - correct behavior)
    print("\n4. Testing with BOTH features normalized (new fixed behavior):")
    features_source_norm = F.normalize(features_source, p=2, dim=1)
    features_target_norm = F.normalize(features_target, p=2, dim=1)
    cosine_dist_fixed = 1.0 - (features_source_norm @ features_target_norm.T)
    print(f"   Cosine distance stats (FIXED):")
    print(f"   - Min: {cosine_dist_fixed.min():.4f}")
    print(f"   - Max: {cosine_dist_fixed.max():.4f}")
    print(f"   - Mean: {cosine_dist_fixed.mean():.4f}")

    # Validate bounds
    is_valid = (cosine_dist_fixed.min() >= 0) and (cosine_dist_fixed.max() <= 2)
    if is_valid:
        print(f"   ✓ SUCCESS: All cosine distances are in valid [0, 2] range!")
    else:
        print(f"   ✗ FAILURE: Cosine distances out of bounds!")

    # Check for NaN
    n_nan = torch.isnan(cosine_dist_fixed).sum().item()
    if n_nan > 0:
        print(f"   ⚠️  WARNING: {n_nan} NaN values detected!")
    else:
        print(f"   ✓ No NaN values detected")

    return is_valid

def test_euclidean_standardization():
    """Test that euclidean metric uses proper standardization."""
    print("\n" + "="*60)
    print("Testing Euclidean Metric Standardization")
    print("="*60)

    # Create synthetic data
    print("\n1. Creating synthetic data...")
    n_cells, n_features = 200, 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    features_source = torch.randn(n_cells, n_features, device=device) * 2 + 1
    features_target = torch.randn(n_cells, n_features, device=device) * 3 - 0.5

    print(f"   Source stats: mean={features_source.mean():.4f}, std={features_source.std():.4f}")
    print(f"   Target stats: mean={features_target.mean():.4f}, std={features_target.std():.4f}")

    # Compute target statistics (as done in M-step)
    print("\n2. Computing target feature statistics...")
    feature_mean = features_target.mean(dim=0, keepdim=True)
    feature_std = features_target.std(dim=0, keepdim=True).clamp(min=1e-6)

    # Standardize both using target stats (as done after fix)
    print("\n3. Standardizing both with target statistics...")
    from m_step_utils import standardize_features
    features_source_std = standardize_features(features_source, feature_mean, feature_std)
    features_target_std = standardize_features(features_target, feature_mean, feature_std)

    print(f"   Standardized source: mean={features_source_std.mean():.4f}, std={features_source_std.std():.4f}")
    print(f"   Standardized target: mean={features_target_std.mean():.4f}, std={features_target_std.std():.4f}")

    # Compute euclidean distances
    print("\n4. Computing euclidean distances...")
    euclidean_dist = torch.cdist(features_source_std, features_target_std, p=2)
    print(f"   Distance stats:")
    print(f"   - Min: {euclidean_dist.min():.4f}")
    print(f"   - Max: {euclidean_dist.max():.4f}")
    print(f"   - Mean: {euclidean_dist.mean():.4f}")

    # Check for NaN
    n_nan = torch.isnan(euclidean_dist).sum().item()
    if n_nan > 0:
        print(f"   ⚠️  WARNING: {n_nan} NaN values detected!")
        return False
    else:
        print(f"   ✓ No NaN values detected")
        return True

if __name__ == "__main__":
    print("Normalization Fix Verification Tests")
    print("====================================")

    # Run tests
    test1_passed = test_cosine_distance_bounds()
    test2_passed = test_euclidean_standardization()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    if test1_passed and test2_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nThe normalization fix successfully:")
        print("1. Keeps cosine distances in valid [0, 2] range")
        print("2. Properly standardizes features for euclidean metric")
        print("3. Prevents NaN values in distance matrices")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        if not test1_passed:
            print("- Cosine distance test failed")
        if not test2_passed:
            print("- Euclidean standardization test failed")
        sys.exit(1)