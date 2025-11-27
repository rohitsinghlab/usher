#!/usr/bin/env python3
"""
Example usage of the cleaned-up spot_alignment_sinkhorn_features_only.py

This demonstrates the simplified workflow:
1. Geosketch source (reference) to reduce size
2. Geosketch target (query) into batches  
3. Run Sinkhorn multiple times until all target batches done
4. No class probabilities
5. Only 'global' M-step for now
"""

import numpy as np
import anndata as ad
from spot_alignment_sinkhorn_features_only import align_features_sinkhorn

def create_dummy_data():
    """Create dummy AnnData objects for testing."""
    # Source (reference): large dataset
    n_source = 10000
    n_target = 2000
    n_features = 100
    
    # Create random features
    X_source = np.random.randn(n_source, n_features).astype(np.float32)
    X_target = np.random.randn(n_target, n_features).astype(np.float32)
    
    # Create AnnData objects
    adata_source = ad.AnnData(X=X_source)
    adata_target = ad.AnnData(X=X_target)
    
    return adata_source, adata_target

def main():
    """Main example function."""
    print("Creating dummy data...")
    adata_source, adata_target = create_dummy_data()
    
    print(f"Source: {adata_source.shape}")
    print(f"Target: {adata_target.shape}")
    
    print("\nRunning alignment with geosketch...")
    
    # Run alignment with geosketch
    model, mapping, concat_adata = align_features_sinkhorn(
        adata_source, adata_target,
        
        # Core parameters
        e_step_method='ot',  # 'ot' or 'gw'
        m_step_method='global',  # Only 'global' for now
        
        # Geosketch parameters
        use_geosketch=True,
        sketch_size_source=2000,  # Sketch source to 2k cells
        sketch_size_target=500,   # Process target in batches of 500
        data_is_pca=True,  # Data is already PCA-transformed
        
        # OT parameters
        epsilon=0.1,
        sinkhorn_iters=1000,
        balanced_ot=True,  # True=balanced OT, False=unbalanced OT
        
        # Training parameters
        n_iters=10,  # Fewer iterations for demo
        steps_per_iter=20,
        lr=1e-3,
        
        # Debug
        debug=True,
        debug_plots_path='debug_plots_geosketch',
        debug_plot_freq=2,  # Generate plots every 2 iterations
    )
    
    print(f"\nAlignment completed!")
    print(f"Model type: {type(model)}")
    print(f"Mapping shape: {mapping.shape}")
    print(f"Valid mappings: {np.sum(mapping >= 0)}/{len(mapping)}")
    print(f"Concatenated data shape: {concat_adata.shape}")
    
    # Show some statistics
    valid_mappings = mapping[mapping >= 0]
    if len(valid_mappings) > 0:
        print(f"Mapping range: {valid_mappings.min()} to {valid_mappings.max()}")
        print(f"Unique targets mapped to: {len(np.unique(valid_mappings))}")

if __name__ == "__main__":
    main()
