"""
Test script for spatial windowing in spot_alignment_gw_knn_windowed_fast.py
"""

from spot_alignment_gw_knn_windowed_fast import align_spots_gw_knn_windowed_fast

# Example usage with spatial windows
model, mapping, T, concat_adata = align_spots_gw_knn_windowed_fast(
    metabAD_pca, pcfAD_pca,
    
    # k-NN constraint
    spatial_knn=50,
    
    # SPATIAL WINDOWING (NEW!)
    window_height=100.0,      # Height in coordinate units
    window_width=100.0,       # Width in coordinate units
    window_overlap=0.1,       # 10% overlap between adjacent windows
    # If None, auto-determines based on tissue extent (~3x3 grid)
    
    # FGW parameters
    alpha=0.5,
    epsilon=0.01,             # Increased from 1e-3 for smoother optimization
    fgw_max_iter=50,
    
    # Training
    n_iters=30,
    steps_per_iter=20,
    
    # Loss weights
    lambda_cross=1.0,
    lambda_struct=1.0,
    
    # Ground cost
    ground_w_spatial=1.0,     # Use spatial cost
    ground_w_cosine=0.0,      # Don't use feature cost in ground
    
    # Debug
    debug_plots_path="figures_spatial_gw",
)

print(f"\n✅ Alignment complete!")
print(f"Mapping shape: {mapping.shape}")
print(f"Transport sparsity: {(T > 1e-6).sum() / T.size * 100:.2f}%")

# Check UMAP mixing
import scanpy as sc
fig, ax = plt.subplots(figsize=(8, 8))
sc.pl.umap(
    concat_adata,
    color='type',
    palette={'query_transformed': '#1f77b4', 'template': '#ff7f0e'},
    ax=ax,
    show=False,
    title='Final UMAP (Spatial Windowing)'
)
plt.savefig('figures_spatial_gw/final_umap.png', dpi=150, bbox_inches='tight')
print("Saved final UMAP to figures_spatial_gw/final_umap.png")