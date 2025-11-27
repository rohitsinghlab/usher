import logging
import os
from typing import Optional, Tuple, Dict, List

import anndata as ad
import numpy as np
import ot
import ot.backend as otb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import rapids_singlecell as rsc
import pandas as pd
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import from model_utils
# For saving/loading/applying models, see model_utils.py:
#   - save_alignment_model()
#   - load_alignment_model()
#   - apply_alignment_model()
from model_utils import FeatureTransform, _to_dense_float32


# Import from graph_utils
from graph_utils import compute_knn_graph_distance, apply_cell_type_constraints

# Import from gs_utils
from gs_utils import geosketch_subsample, geosketch_target_batches, geosketch_stratified_pairing

# Import from e_step_utils and m_step_utils
from e_step_utils import compute_transport_batch
from m_step_utils import (
    aggregate_training_data_from_batches,
    train_global_model,
    apply_transfer_method,
    unstandardize_features
)

# Import from plot_utils
from plot_utils import (
    plot_dual_umap,
    plot_weight_heatmap,
    plot_convergence,
    plot_transfer_debug_umap,
    plot_spatial_channels,
    plot_spatial_mapping
)



def align_features_sinkhorn(
    adata_a: ad.AnnData, #Source dataset
    adata_b: ad.AnnData, #Target dataset
    # Core parameters
    e_step_method: str = 'ot',  # 'ot', 'gw', or 'fgw'
    m_step_method: str = 'transfer',  # 'global' or 'transfer'

    # Sampling strategy selection
    sampling_strategy: str = 'celltype',  # 'celltype' or 'spatial'

    # Celltype sampling parameters (used when sampling_strategy='celltype')
    sketch_size: int = 1000,  # Group size for stratified pairing (if use_stratified_pairing=True) or target batch size (if False)
    sketch_obsm_key: str = 'X_umap',
    use_stratified_pairing: bool = True,  # Use stratified pairing (geosketch both source and target into k groups)
    stratified_pairing_fix: bool = True,  # If True, use fixed groups across iterations; if False, resample each iteration
    cell_type_col: str = 'annotation_level_0',  # Column containing cell type info

    # Spatial sampling parameters (used when sampling_strategy='spatial')
    spatial_key: str = 'X_spatial',  # Key in .obsm for spatial coordinates
    window_height: Optional[float] = None,  # Height of spatial windows (auto-computed if None)
    window_width: Optional[float] = None,  # Width of spatial windows (auto-computed if None)
    window_overlap: float = 0.1,  # Overlap between spatial windows
    spatial_knn: int = 50,  # Number of feature k-NN for spatial constraints
    n_windows_target: int = 20,  # Target number of windows if auto-computing dimensions

    # OT/GW parameters
    epsilon: float = 0.1,
    sinkhorn_iters: int = 1000,
    balanced_ot: bool = True,  # Whether to use balanced or unbalanced OT

    use_linear_assignment: bool = False,  # Use Hungarian algorithm for globally optimal 1-1 matching (overrides top-k)

    # kNN graph parameters
    knn_k: int = 15,  # Number of nearest neighbors for kNN graph
    metric: str = 'cosine',  # Distance metric for kNN, UMAP ('cosine' or 'euclidean')
    m_step_metric: str = 'euclidean',  # Distance metric for M-step loss ('cosine' or 'euclidean')
    use_knn_graph: bool = True,  # Whether to use kNN graph distances instead of direct distances

    # Fused GW parameters (for 'fgw' e_step_method)
    celltype_probs_layer: str = 'X_celltype_probs',  # Key in .obsm for cell type probabilities
    alpha: float = 0.5,  # Balance between structure (GW) and features (Wasserstein): 1.0=pure GW, 0.0=pure Wasserstein
    gamma: float = 0.5,  # Weight for feature distance in M matrix: M = (1-gamma)*M_celltype + gamma*M_features (0=celltype only, 1=features only, iter 0 uses gamma=0)
    sketch_pca_components: int = 50,
    # Training parameters
    n_iters: int = 30,
    steps_per_iter: int = 50,  # Number of gradient steps per M-step (only for MLP)
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_cross: float = 1.0,  # Weight for cross-domain alignment loss (only for MLP)
    lambda_struct: float = 1.0,  # Weight for structure preservation loss (only for MLP)
    lambda_var: float = 0.1,  # Weight for variance preservation loss (only for MLP)
    structure_sample_size: Optional[int] = 2048,  # Sample size for structure loss (only for MLP)
    hidden_dim: Optional[int] = None,  # Hidden dimension for MLP (None = linear, only for MLP)
    dropout: float = 0.0,  # Dropout rate for MLP (only for MLP)
    init_strategy: str = 'auto',  # Model initialization: 'identity', 'random', or 'auto' (auto-select based on sampling strategy)
    device: Optional[str] = None,
    debug_plots_path: Optional[str] = None,
    entropy_percentile: float = 30.0,
    confidence_percentile: float = 70.0,
    verbose: bool = True,
) -> Tuple[nn.Module, np.ndarray, ad.AnnData]:
    """
    CORRECTED E-M style features-only alignment using geosketch sampling.

    CORRECTED WORKFLOW:
    1. Sample source (reference) to reduce size using geosketch
    2. Sample target (query) into batches using geosketch
    3. E-M iterations (OUTER LOOP): shared model across all batches
    4. Within each E-M iteration: process all target batches
    5. No longer use class probabilities

    Parameters
    ----------
    adata_a : ad.AnnData
        Source dataset (reference)
    adata_b : ad.AnnData
        Target dataset (query)
    e_step_method : str
        Method for E-step transport plan computation:
        - 'ot': Pure Wasserstein using cell type feature cost M - faster
        - 'gw': Pure Gromov-Wasserstein using structure (C1, C2) - preserves structure, slower
        - 'fgw': Fused Gromov-Wasserstein combining features (M) and structure (C1, C2) - balanced approach
    m_step_method : str
        Method for M-step optimization:
        - 'global': Learn a neural network transformation
        - 'transfer': Direct expression transfer from mapped cells (weighted by transport)
    sketch_size : int
        Group/batch size for sampling (default: 1000)
        - If use_stratified_pairing=True: Size of each paired group (both source and target)
        - If use_stratified_pairing=False: 

    sketch_obsm_key : str
        Key in .obsm for pre-computed embeddings to use for geosketch (default: 'X_umap')
        Commonly used keys: 'X_umap', 'X_pca', 'X_tsne'
        If key not found in AnnData, will fallback to computing PCA
    sketch_pca_components : int
        Number of PCA components to compute if sketch_obsm_key not found in .obsm (default: 50)
        DEPRECATED: Only used as fallback
    data_is_pca : bool
        DEPRECATED: Ignored when using sketch_obsm_key from AnnData
    use_stratified_pairing : bool
        Whether to use stratified pairing (default: False)
        - True: Geosketch both source and target into k groups, process pairs (source[i] ↔ target[i])
        - False: Use original behavior (sample source once, batch target)
    stratified_pairing_fix : bool
        Whether to use fixed groups across iterations (default: True, only relevant if use_stratified_pairing=True)
        - True: Use same groups across all EM iterations (fix=True in geosketch_stratified_pairing)
        - False: Resample groups each iteration (fix=False, seed changes each iteration)
    cell_type_col : str
        Column name containing cell type information for constraints and visualization
    epsilon : float
        Sinkhorn regularization parameter
    sinkhorn_iters : int
        Max iterations for Sinkhorn algorithm
    balanced_ot : bool
        Whether to use balanced or unbalanced Optimal Transport:
        - True (balanced): Enforces strict marginal constraints (sum of rows = sum of columns)
          Best when source and target have similar total mass/importance
          More stable but may be restrictive for datasets with different sizes
        - False (unbalanced): Relaxes marginal constraints, allows partial matching
          Better for datasets with very different sizes or when some cells should remain unmatched
          More flexible but requires tuning of marginal relaxation parameters
    sparsify_transport : bool
        Whether to sparsify transport plan by keeping only top-k entries per row (default: True)
        Reduces smoothing effect of T @ features_b by making each source map to fewer targets
        Helps preserve biological signal in transferred expressions
    transport_top_k : int
        Number of top entries to keep per row when sparsifying (default: 5)
        Each source cell will map to at most this many target cells
        Lower values = less smoothing but may lose alignment quality
        Higher values = more smoothing but better coverage
        Ignored if use_linear_assignment=True
    use_linear_assignment : bool
        Use Hungarian algorithm (linear sum assignment) for globally optimal 1-1 matching (default: False)
        When True, finds the best bijective mapping (each source → 1 target, each target ← ≤1 source)
        Guarantees perfect diversity (no target overuse) but results in hard assignments
        Overrides transport_top_k if enabled
        Recommended when T is very flat and diversity is critical
    knn_k : int
        Number of nearest neighbors for kNN graph construction
    metric : str
        Distance metric for kNN graph, UMAP, and M-step loss ('cosine' or 'euclidean')
        - 'cosine': Direction-based similarity, better for normalized gene expression (default)
          Loss: 1 - cosine_similarity, Structure: cosine similarity matrices
        - 'euclidean': Magnitude-aware distance, preserves absolute expression levels
          Loss: L2 distance, Structure: Euclidean distance matrices
        This single parameter ensures consistency across all distance computations

    use_knn_graph : bool
        Whether to use kNN graph-based shortest path distances instead of direct distances:
        - True: Compute kNN graph, then shortest path distances (better captures manifold structure)
        - False: Use direct cosine/Euclidean distances (faster but less accurate for complex manifolds)
    cell_type_penalty : float
        Penalty factor for cross-cell-type distances in kNN graph (0=disabled, default=0.5)
    celltype_probs_layer : str
        Key in .obsm for cell type probabilities (default: 'X_celltype_probs')
        Used for FGW and OT methods to compute feature cost matrix M
        Falls back to one-hot encoding from cell_type_col if not present

    alpha : float
        Balance between structure (GW) and features (Wasserstein) for FGW method:
        - 1.0: Pure Gromov-Wasserstein (structure only)
        - 0.0: Pure Wasserstein (features only)
        - 0.5 (default): Balanced fusion
    gamma : float
        Weight for feature distance in M matrix computation (OT and FGW):
        M = (1 - gamma) * M_celltype + gamma * M_features
        - 0.0: Use only cell type distances (default for iteration 0)
        - 1.0: Use only transformed feature distances
        - 0.5 (default): Balanced combination
        Note: Iteration 0 always uses gamma=0 regardless of this setting
    n_iters : int
        Number of E-M iterations
    steps_per_iter : int
        Number of gradient steps per M-step
    lr : float
        Learning rate for optimizer
    weight_decay : float
        Weight decay for optimizer
    lambda_cross : float
        Weight for cross-domain alignment loss
    lambda_struct : float
        Weight for structure preservation loss
    structure_sample_size : int, optional
        Sample size for structure preservation loss (for memory efficiency)
    hidden_dim : int, optional
        Hidden dimension for MLP (None = linear)
    dropout : float
        Dropout rate for MLP
    init_strategy : str
        Model weight initialization: 'identity', 'random', or 'auto' (auto-selects based on sampling_strategy)
    device : str, optional
        Device to use ('cuda' or 'cpu')
    debug : bool
        Enable debug logging
    debug_plots_path : str, optional
        Path to save debug plots
    debug_plot_freq : int
        Frequency of debug plot generation
    clear_cuda_cache_each_iter : bool
        Whether to clear CUDA cache after each iteration
        
    Returns
    -------
    model : FeatureTransform
        Trained transformation model
    mapping : np.ndarray
        Hard assignment from query to template
    concat_adata : ad.AnnData
        Concatenated transformed source + target for UMAP
    T_full : torch.Tensor
        Full transport matrix (n_source, n_target)
    feature_mean : torch.Tensor or None
        Mean per feature column used for standardization (None if m_step_metric='cosine')
    feature_std : torch.Tensor or None
        Std per feature column used for standardization (None if m_step_metric='cosine')
    """
    
    # --- Device & data ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    logging.info(f"Using device: {device}")

    if debug_plots_path:
        os.makedirs(debug_plots_path, exist_ok=True)
        subdirs = ["umap", "heatmap", "umap_transfer", "convergence"]
        # Add channels directory for spatial sampling
        if sampling_strategy == 'spatial':
            subdirs.append("channels")
        for subdir in subdirs:
            os.makedirs(os.path.join(debug_plots_path, subdir), exist_ok=True)

        # Set up file logging to debug_plots_path
        log_file = os.path.join(debug_plots_path, 'alignment.log')

        # Create a file handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler to root logger
        logger = logging.getLogger()
        logger.addHandler(file_handler)

        # Also keep console output
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)
        logging.info(f"Logging to file: {log_file}")

    # Load data
    X_a = _to_dense_float32(adata_a.X)
    X_b = _to_dense_float32(adata_b.X)

    features_a = torch.from_numpy(X_a).to(device_t)
    features_b = torch.from_numpy(X_b).to(device_t)

    n_a_orig, d_a = features_a.shape
    n_b_orig, d_b = features_b.shape
    logging.info(f"Source (A): {n_a_orig} spots, {d_a} features. Target (B): {n_b_orig} spots, {d_b} features.")

    # Validate sampling strategy
    if sampling_strategy not in ['celltype', 'spatial']:
        raise ValueError(f"sampling_strategy must be 'celltype' or 'spatial', got: {sampling_strategy}")

    # Validate parameters
    if e_step_method not in ['ot', 'gw', 'fgw']:
        raise ValueError(f"e_step_method must be 'ot', 'gw', or 'fgw', got: {e_step_method}")
    if m_step_method not in ['global', 'transfer']:
        raise ValueError(f"m_step_method must be 'global' or 'transfer', got: {m_step_method}")



    # Validate metric
    if metric not in ['cosine', 'euclidean']:
        raise ValueError(f"metric must be 'cosine' or 'euclidean', got: {metric}")

    logging.info(f"E-step method: {e_step_method}")
    logging.info(f"M-step method: {m_step_method}")
    if m_step_method == 'global':
        logging.info(f"M-step method: {m_step_method}" + ( f" (lr={lr}, steps_per_iter={steps_per_iter})"))
    logging.info(f"Distance metric (kNN/UMAP/M-step loss): {metric}")
    logging.info(f"kNN graph: {'enabled' if use_knn_graph else 'disabled'} (k={knn_k})")
    if e_step_method in ['ot', 'fgw']:
        logging.info(f"Gamma (feature weight in M): {gamma} (iter 0 uses gamma=0)")
    if e_step_method == 'ot':
        logging.info(f"OT type: {'balanced' if balanced_ot else 'unbalanced'}")
        if not balanced_ot:
            logging.info("Unbalanced OT: allows partial matching, tau=0.8 for marginal relaxation")

    # ===== Prepare batches based on sampling strategy =====
    logging.info(f"=== Using {sampling_strategy.upper()} sampling strategy ===")

    batches = []
    auxiliary_data = {}
    sketch_to_original = None
    window_info = {}
    n_a = n_a_orig  # Default: use full source dataset

    if sampling_strategy == 'spatial':
        # SPATIAL WINDOWING approach
        from spatial_utils import prepare_spatial_batches

        batches, auxiliary_data, window_info = prepare_spatial_batches(
            adata_a, adata_b,
            spatial_key=spatial_key,
            window_height=window_height,
            window_width=window_width,
            window_overlap=window_overlap,
            n_windows_target=n_windows_target,
            spatial_knn=spatial_knn
        )

        # No source subsampling for spatial windowing
        logging.info(f"Spatial windowing: {len(batches)} windows created")

    else:  # sampling_strategy == 'celltype'
        # CELLTYPE-BASED approach (geosketch)
        from celltype_utils import prepare_celltype_batches

        batches, auxiliary_data, sketch_to_original = prepare_celltype_batches(
            adata_a, adata_b,
            sketch_size=sketch_size,
            use_stratified_pairing=use_stratified_pairing,
            stratified_pairing_fix=stratified_pairing_fix,
            celltype_probs_layer=celltype_probs_layer,
            cell_type_col=cell_type_col,
            sketch_obsm_key=sketch_obsm_key,
            sketch_pca_components=sketch_pca_components,
            e_step_method=e_step_method,
            seed=2025
        )

        # Update n_a if source was subsampled
        n_a = auxiliary_data['n_source']

        # Subset features if source was subsampled
        if sketch_to_original is not None:
            features_a = features_a[sketch_to_original]
            logging.info(f"Source features subsetted to {features_a.shape}")

    # Extract auxiliary features for transport computation
    celltype_probs_a = auxiliary_data.get('celltype_probs_a')
    celltype_probs_b = auxiliary_data.get('celltype_probs_b')
    coords_a = auxiliary_data.get('coords_a')
    coords_b = auxiliary_data.get('coords_b')
    knn_indices_spatial = auxiliary_data.get('knn_indices')  # For spatial windowing

    # Determine which auxiliary features to use based on sampling strategy
    if sampling_strategy == 'spatial':
        auxiliary_features_source = coords_a
        auxiliary_features_target = coords_b
    else:
        auxiliary_features_source = celltype_probs_a
        auxiliary_features_target = celltype_probs_b


    # Initialize shared model and optimizer (across all batches)
    model = FeatureTransform(input_dim=d_a, output_dim=d_b, hidden_dim=hidden_dim, dropout=dropout).to(device_t)

    # Initialize model weights based on init_strategy parameter
    if init_strategy == 'auto':
        # Auto-select based on sampling strategy
        if sampling_strategy == 'celltype':
            model.init_identity()
            logging.info("Auto-selected identity initialization for celltype sampling")
        else:
            model.init_random()
            logging.info("Auto-selected random initialization for spatial sampling")
    elif init_strategy == 'identity':
        model.init_identity()
    elif init_strategy == 'random':
        model.init_random()
    else:
        raise ValueError(f"Unknown init_strategy: {init_strategy}. Choose 'identity', 'random', or 'auto'")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize features_a_transformed for transfer method
    features_a_transformed = None

    # Storage for feature standardization (euclidean M-step only)
    global_feature_mean = None
    global_feature_std = None

    # Storage for convergence metrics (for plotting)
    convergence_data = []  # List of dicts: [{iter, batch_idx, n_changed, n_total_valid, pct_changed}, ...]
    prev_iter_mappings = {}  # Dict: batch_idx -> mapping_batch (np.ndarray) for convergence tracking

    # Helper function to apply model with standardization (for euclidean metric only)
    def apply_model_with_scaling(features_in: torch.Tensor) -> torch.Tensor:
        """Apply model with standardization/unstandardization if using euclidean metric."""
        if global_feature_mean is not None and global_feature_std is not None:
            # Standardize input
            from m_step_utils import standardize_features
            features_std = standardize_features(features_in, global_feature_mean, global_feature_std)
            # Apply model
            output_std = model(features_std)
            # Unstandardize output
            output = unstandardize_features(output_std, global_feature_mean, global_feature_std)
            return output
        else:
            # No standardization (cosine metric or first iteration)
            return model(features_in)

    # E-M iterations (outer loop) - shared across all batches
    for it in tqdm(range(n_iters), desc="E-M Alignment"):
        logging.info(f"E-M Iteration {it + 1}/{n_iters}")

        # Resample groups if using rotating stratified pairing (celltype sampling only)
        if sampling_strategy == 'celltype' and use_stratified_pairing and not stratified_pairing_fix:
            logging.info(f"Resampling stratified groups for iteration {it + 1} (fix=False)")
            from celltype_utils import prepare_celltype_batches

            batches, auxiliary_data, _ = prepare_celltype_batches(
                adata_a, adata_b,
                sketch_size=sketch_size,
                use_stratified_pairing=True,
                stratified_pairing_fix=False,
                celltype_probs_layer=celltype_probs_layer,
                cell_type_col=cell_type_col,
                sketch_obsm_key=sketch_obsm_key,
                sketch_pca_components=sketch_pca_components,
                e_step_method=e_step_method,
                seed=2025 + it  # Different seed each iteration
            )

            # Update auxiliary features
            celltype_probs_a = auxiliary_data.get('celltype_probs_a')
            celltype_probs_b = auxiliary_data.get('celltype_probs_b')
            auxiliary_features_source = celltype_probs_a
            auxiliary_features_target = celltype_probs_b

            logging.info(f"Resampled {len(batches)} paired groups for iteration {it + 1}")

        # Initialize features_a_transformed for transfer method (only once)
        if m_step_method == 'transfer' and it == 0:
            features_a_transformed = features_a.clone()
            logging.info(f"Transfer method: Initialized features_a_transformed with source features (shape: {features_a.shape})")

        # ===== PHASE 1: E-STEP FOR ALL BATCHES =====
        logging.info(f"=== E-STEP: Computing transport for all {len(batches)} batches ===")
        batch_results = []

        for batch_idx, (source_batch_indices, target_batch_indices) in enumerate(batches):
            # Determine which source features to use (iteration 0 vs later)
            if it == 0:
                features_source_base = features_a
            else:
                if m_step_method == 'transfer':
                    features_source_base = features_a_transformed
                else:  # global
                    # Apply model transformation (with standardization if euclidean)
                    with torch.no_grad():
                        features_source_base = apply_model_with_scaling(features_a)

            # Normalize features based on metric for OT computation
            if metric == 'cosine':
                # L2-normalize both source and target for cosine distance
                features_source_normalized = F.normalize(features_source_base, p=2, dim=1)
                features_target_normalized = F.normalize(features_b, p=2, dim=1)
                logging.debug(f"  Using cosine metric: L2-normalized both source and target features")
            else:  # euclidean
                # Standardize both using target's mean/std (if available from M-step)
                if global_feature_mean is not None and global_feature_std is not None:
                    from m_step_utils import standardize_features
                    features_source_normalized = standardize_features(features_source_base, global_feature_mean, global_feature_std)
                    features_target_normalized = standardize_features(features_b, global_feature_mean, global_feature_std)
                    logging.debug(f"  Using euclidean metric: Standardized both source and target with target stats")
                else:
                    # First iteration or no scaler available: use raw features
                    features_source_normalized = features_source_base
                    features_target_normalized = features_b
                    logging.debug(f"  Using euclidean metric: No standardization (first iteration)")

            # Compute transport using modular function
            # Handle different sampling strategies
            if source_batch_indices is None:
                # For non-stratified celltype sampling: use all source cells
                source_indices_batch = np.arange(n_a)
            else:
                # For stratified pairing or spatial windowing: use specified indices
                source_indices_batch = source_batch_indices

            # Prepare k-NN constraint for spatial windowing
            knn_constraint_indices = None
            if sampling_strategy == 'spatial' and knn_indices_spatial is not None:
                # Extract k-NN indices for this window's source cells
                knn_constraint_indices = knn_indices_spatial[source_indices_batch]

            result = compute_transport_batch(
                source_indices=source_indices_batch,
                target_indices=target_batch_indices,
                features_source_all=features_source_normalized,
                features_target_all=features_target_normalized,
                auxiliary_features_source=auxiliary_features_source,
                auxiliary_features_target=auxiliary_features_target,
                gamma=gamma,
                epsilon=epsilon,
                metric=metric,
                balanced=balanced_ot,
                use_linear_assignment=use_linear_assignment,
                device=device_t,
                iteration=it,
                e_step_method=e_step_method,
                entropy_percentile=entropy_percentile,
                confidence_percentile=confidence_percentile,
                # GW/FGW parameters
                knn_k=knn_k,
                use_knn_graph=use_knn_graph,
                alpha=alpha,
                verbose=verbose,
                # Spatial-specific parameters
                knn_constraint_indices=knn_constraint_indices,
                knn_penalty_weight=5.0,  # Could make this a parameter later,
                sampling_strategy = sampling_strategy
            )

            batch_results.append(result)
            logging.info(f"  Batch {batch_idx+1}/{len(batches)}: Transport computed")

        # Track convergence by comparing mappings with previous iteration
        for batch_idx, result in enumerate(batch_results):
            T = result['T']
            mapping_curr = T.argmax(dim=1).cpu().numpy()

            if it > 0 and batch_idx in prev_iter_mappings:
                mapping_prev = prev_iter_mappings[batch_idx]
                n_changed = np.sum(mapping_curr != mapping_prev)
                n_total = len(mapping_curr)
                pct_changed = 100.0 * n_changed / n_total if n_total > 0 else 0.0

                convergence_data.append({
                    'iter': it,
                    'batch_idx': batch_idx,
                    'n_changed': n_changed,
                    'n_total_valid': n_total,
                    'pct_changed': pct_changed
                })

                logging.info(f"  Convergence: Batch {batch_idx}: {n_changed}/{n_total} cells changed ({pct_changed:.1f}%)")

            # Store current mapping for next iteration
            prev_iter_mappings[batch_idx] = mapping_curr

        # ===== PHASE 2: M-STEP ON AGGREGATED DATA =====
        logging.info(f"=== M-STEP: Training on aggregated data from all batches ===")

        if m_step_method == 'global':
            # Aggregate training data from ALL batches
            source_agg, target_agg, agg_stats = aggregate_training_data_from_batches(
                batch_results=batch_results,
                features_source_all=features_a,
                features_target_all=features_b,
                entropy_percentile=entropy_percentile,
                confidence_percentile=confidence_percentile
            )

            # Train model ONCE on aggregated data
            step_losses, feature_mean, feature_std = train_global_model(
                model=model,
                optimizer=optimizer,
                source_features=source_agg,
                target_features=target_agg,
                steps_per_iter=steps_per_iter,
                lambda_cross=lambda_cross,
                lambda_struct=lambda_struct,
                lambda_var=lambda_var,
                metric=m_step_metric,
                structure_sample_size=structure_sample_size,
                device=device_t,
                features_target_all=features_b  # Pass FULL target dataset for correct scaler
            )

            # Store scaler for later (needed to unstandardize model outputs)
            if it == 0:
                # First iteration: initialize scaler storage
                global_feature_mean = feature_mean
                global_feature_std = feature_std
            elif feature_mean is not None:
                # Update scaler (should be consistent across iterations for euclidean)
                global_feature_mean = feature_mean
                global_feature_std = feature_std

            avg_loss = np.mean(step_losses) if step_losses else 0.0
            logging.info(f"M-step: Trained on {len(source_agg)} aggregated cells, avg loss={avg_loss:.6f}")

        elif m_step_method == 'transfer':
            # Apply transfer ONCE using all transport plans
            features_a_transformed = apply_transfer_method(
                batch_results=batch_results,
                features_target_all=features_b,
                n_source_total=n_a,
                device=device_t
            )

            logging.info(f"M-step: Applied transfer to {n_a} source cells")

        # ===== DEBUG PLOTS (runs for both methods) =====
        # Generate debug plots AFTER processing all batches (every iteration)
        if debug_plots_path:
            logging.info(f"Generating debug plots for iteration {it+1}...")

            # UMAP visualization using ALL target data (not just one batch)
            if m_step_method == 'transfer':
                # For transfer method, use the TRANSFERRED features (target expressions transferred to source)
                # This shows the actual result of the transfer method
                a_hat_cpu = features_a_transformed.detach().cpu().numpy()
                logging.info("Transfer method: Using TRANSFERRED source features for UMAP (target expressions transferred to source)")
            else:
                # For global method, use model output
                with torch.no_grad():
                    model.eval()  # Set to eval mode for inference

                    # Variance diagnostics BEFORE global transformation
                    source_var_before = features_a.var(dim=0).mean().item()
                    source_std_before = features_a.std(dim=0).mean().item()
                    source_range_before = (features_a.max() - features_a.min()).item()
                    logging.info(f"VARIANCE CHECK - Source BEFORE global transform: var={source_var_before:.6f}, std={source_std_before:.6f}, range={source_range_before:.6f}")

                    a_hat_global = apply_model_with_scaling(features_a)

                    # Variance diagnostics AFTER global transformation
                    trans_var_after = a_hat_global.var(dim=0).mean().item()
                    trans_std_after = a_hat_global.std(dim=0).mean().item()
                    trans_range_after = (a_hat_global.max() - a_hat_global.min()).item()
                    logging.info(f"VARIANCE CHECK - Source AFTER global transform: var={trans_var_after:.6f}, std={trans_std_after:.6f}, range={trans_range_after:.6f}")
                    logging.info(f"VARIANCE PRESERVATION RATIO (global): {trans_std_after/source_std_before:.4f}")

                    a_hat_cpu = a_hat_global.detach().cpu().numpy()

            # Use sketched observations if geosketch was applied
            if sketch_to_original is not None:
                obs_sketched = adata_a.obs.iloc[sketch_to_original].copy()
                # Also subset spatial coordinates if present
                obsm_dict = {}
                if spatial_key and spatial_key in adata_a.obsm:
                    obsm_dict[spatial_key] = adata_a.obsm[spatial_key][sketch_to_original]
            else:
                obs_sketched = adata_a.obs.copy()
                obsm_dict = {}
                if spatial_key and spatial_key in adata_a.obsm:
                    obsm_dict[spatial_key] = adata_a.obsm[spatial_key]
            adata_a_transformed = ad.AnnData(X=a_hat_cpu, obs=obs_sketched, obsm=obsm_dict if obsm_dict else None)
            adata_a_transformed.obs["type"] = "source_transformed"  # Back to transformed since we're showing transferred features

            # Ensure target copy preserves spatial coordinates
            adata_b_copy = adata_b.copy()
            adata_b_copy.obs["type"] = "target"
            # Spatial coordinates should already be in adata_b.obsm if it's spatial data

            concat_adata_iter = ad.concat(
                [adata_a_transformed, adata_b_copy], axis=0,
                label="batch", keys=["source", "target"], index_unique="_",
            )
            # Set common_CT column correctly (if present)
            if cell_type_col in adata_a_transformed.obs.columns and cell_type_col in adata_b_copy.obs.columns:
                common_ct_values = pd.concat([adata_a_transformed.obs[cell_type_col], adata_b_copy.obs[cell_type_col]])
                concat_adata_iter.obs[cell_type_col] = common_ct_values.values
            # Compute UMAP for visualization (using global metric parameter)
            try:
                rsc.tl.pca(concat_adata_iter)
                rsc.pp.neighbors(concat_adata_iter, use_rep='X', metric=metric)
                rsc.tl.umap(concat_adata_iter)
                logging.info(f"UMAP computation successful for iteration {it+1}")
            except Exception as e:
                logging.error(f"UMAP computation failed for iteration {it+1}: {e}")
                # Skip plotting if UMAP fails
                continue

            # Create dual UMAP plots using plot_utils
            plot_dual_umap(
                concat_adata_iter,
                cell_type_col=cell_type_col,
                title_prefix=f'UMAP Iter {it+1}',
                save_path=os.path.join(debug_plots_path, 'umap', f"umap_iter_{it+1:04d}.png"),
                type_palette={'source_transformed': '#1f77b4', 'target': '#ff7f0e'}
            )

            # ============================================================
            # SPATIAL CHANNEL PLOTS (for spatial sampling strategy)
            # ============================================================
            if sampling_strategy == 'spatial' and spatial_key is not None:
                plot_spatial_channels(
                    adata_source=adata_a_transformed,
                    adata_target=adata_b_copy,
                    spatial_key=spatial_key,
                    iteration=it+1,
                    save_path=debug_plots_path,
                    channel_idx=2  # Using channel 2 as specified in the original code
                )

                # Also plot spatial mapping visualization
                # Extract hard mapping from current transport matrix
                try:
                    # Combine transport from all batches (similar to line 749-765)
                    T_full_for_mapping = torch.zeros(n_a, n_b_orig, device=device_t, dtype=torch.float32)
                    for result in batch_results:
                        T = result['T']
                        source_indices = result['source_indices']
                        target_indices = result['target_indices']

                        if sampling_strategy == 'spatial':
                            # For spatial, place batch transport directly
                            source_idx_tensor = torch.from_numpy(source_indices).long().to(device_t)
                            target_idx_tensor = torch.from_numpy(target_indices).long().to(device_t)
                            T_full_for_mapping[source_idx_tensor[:, None], target_idx_tensor] = T
                        elif use_stratified_pairing:
                            # For stratified pairing, place batch transport in full matrix
                            source_idx_tensor = torch.from_numpy(source_indices).long().to(device_t)
                            target_idx_tensor = torch.from_numpy(target_indices).long().to(device_t)
                            T_full_for_mapping[source_idx_tensor[:, None], target_idx_tensor] = T
                        else:
                            # For non-stratified, source covers all indices
                            target_idx_tensor = torch.from_numpy(target_indices).long().to(device_t)
                            T_full_for_mapping[:, target_idx_tensor] = T

                    # Extract hard mapping (argmax)
                    mapping_current = T_full_for_mapping.argmax(dim=1).cpu().numpy()

                    # Mark invalid mappings (where max transport is too small)
                    T_max_vals = T_full_for_mapping.max(dim=1)[0]
                    invalid_mask = (T_max_vals < 1e-6).cpu().numpy()
                    mapping_current[invalid_mask] = -1

                    # Get spatial coordinates
                    coords_source = adata_a.obsm[spatial_key].astype(np.float32)
                    coords_target = adata_b.obsm[spatial_key].astype(np.float32)

                    # Handle sketched source if applicable
                    if sketch_to_original is not None:
                        coords_source = coords_source[sketch_to_original]

                    # Plot spatial mapping
                    plot_spatial_mapping(
                        coords_source=coords_source,
                        coords_target=coords_target,
                        mapping=mapping_current,
                        iteration=it+1,
                        save_path=debug_plots_path,
                        n_samples=5000
                    )
                except Exception as e:
                    logging.warning(f"Failed to create spatial mapping plot: {e}")

            # ============================================================
            # ADDITIONAL DEBUG PLOT: Transfer-based UMAP (for all methods)
            # ============================================================
            # This shows what the alignment looks like using just T @ features_b
            # Helps separate "is T good?" from "is W good?"

            # Reconstruct full transport matrix from batch_results
            T_full_all_debug = torch.zeros(n_a, n_b_orig, device=device_t, dtype=torch.float32)
            for result in batch_results:
                T = result['T']
                source_indices = result['source_indices']
                target_indices = result['target_indices']

                if use_stratified_pairing:
                    # Convert numpy indices to torch tensors for CUDA compatibility
                    source_idx_tensor = torch.from_numpy(source_indices).long().to(device_t)
                    target_idx_tensor = torch.from_numpy(target_indices).long().to(device_t)
                    # For stratified pairing, place batch transport in full matrix
                    T_full_all_debug[source_idx_tensor[:, None], target_idx_tensor] = T
                else:
                    # Convert numpy indices to torch tensors for CUDA compatibility
                    target_idx_tensor = torch.from_numpy(target_indices).long().to(device_t)
                    # For non-stratified, source covers all indices
                    T_full_all_debug[:, target_idx_tensor] = T

            # ===== TARGET CELL CONVERGENCE TRACKING (for linear assignment only) =====
            # Track which target cells changed their matched source from previous iteration
            if use_linear_assignment and it > 0:
                # Compute inverse mapping: target -> source
                # T_full_all_debug is binary after linear assignment
                T_transpose = T_full_all_debug.t()  # (n_target, n_source)
                target_to_source_curr = T_transpose.argmax(dim=1).cpu().numpy()  # (n_target,) - which source maps to each target
                target_has_match_curr = (T_transpose.max(dim=1)[0] > 0.5).cpu().numpy()  # Boolean mask

                if 'target_to_source_prev' in locals():
                    # Compare with previous iteration
                    target_changed = (target_to_source_curr != target_to_source_prev) & target_has_match_curr & target_has_match_prev
                    n_target_changed = target_changed.sum()
                    n_target_matched = target_has_match_curr.sum()
                    pct_target_changed = 100.0 * n_target_changed / n_target_matched if n_target_matched > 0 else 0.0

                    convergence_data.append({
                        'iteration': it,
                        'batch_idx': 'ALL_TARGETS',  # Special marker for target cells
                        'n_changed': n_target_changed,
                        'n_total_valid': n_target_matched,
                        'pct_changed': pct_target_changed
                    })

                    logging.info(f"  Target convergence: {n_target_changed}/{n_target_matched} targets changed source match ({pct_target_changed:.1f}%)")

                # Store for next iteration
                target_to_source_prev = target_to_source_curr
                target_has_match_prev = target_has_match_curr

            # Get appropriate observations
            if sketch_to_original is not None:
                obs_sketched = adata_a.obs.iloc[sketch_to_original].copy()
            else:
                obs_sketched = adata_a.obs.copy()

            # Reconstruct full focused_mask from batch_results (for M-step training visualization)
            focused_mask_full = torch.zeros(n_a, device=device_t, dtype=torch.bool)
            for result in batch_results:
                focused_mask = result.get('focused_mask', None)
                if focused_mask is not None:
                    source_indices = result['source_indices']
                    if use_stratified_pairing:
                        # Convert numpy indices to torch tensors for CUDA compatibility
                        source_idx_tensor = torch.from_numpy(source_indices).long().to(device_t)
                        # Map batch-local focused_mask to full source indices
                        focused_mask_full[source_idx_tensor] = focused_mask
                    else:
                        # For non-stratified, source covers all indices
                        focused_mask_full[:] = focused_mask

            # Create transfer debug UMAP using plot_utils
            plot_transfer_debug_umap(
                T_full=T_full_all_debug,
                features_b=features_b,
                adata_a_obs=obs_sketched,
                adata_b=adata_b,
                cell_type_col=cell_type_col,
                iteration=it+1,
                save_dir=os.path.join(debug_plots_path, 'umap_transfer'),
                metric='euclidean',
                focused_mask_full=focused_mask_full,  # NEW: Pass M-step training mask
                use_linear_assignment=use_linear_assignment  # NEW: Pass flag for third panel
            )

            # Weight heatmap using plot_utils
            plot_weight_heatmap(
                model=model,
                title=f'Transformation Weights Iter {it+1}',
                save_path=os.path.join(debug_plots_path, 'heatmap', f"W_iter_{it+1:04d}.png")
            )

    # Extract mappings from final batch_results (for final output)
    logging.info("Extracting mappings from final transport plans...")
    batch_mappings = []
    for batch_idx, result in enumerate(batch_results):
        T = result['T']
        source_indices = result['source_indices']
        target_indices = result['target_indices']

        # Get hard assignments (argmax per source)
        mapping_batch = T.argmax(dim=1).cpu().numpy()  # Maps source -> target_batch_indices

        batch_mappings.append((batch_idx, mapping_batch, target_indices))
        logging.info(f"Batch {batch_idx}: extracted mapping for {len(mapping_batch)} source cells")

    # ============================================================
    # CONVERGENCE PLOT: Show how mappings converge over iterations
    # ============================================================
    if debug_plots_path:
        plot_convergence(
            convergence_data=convergence_data,
            save_dir=os.path.join(debug_plots_path, 'convergence')
        )

    # Apply transformation based on method
    if m_step_method == 'transfer':
        logging.info("Transfer method: Using features from final iteration...")

        # NOTE: features_a_transformed was already computed in the last iteration
        # (see transfer method section inside iteration loop)
        # No need to recompute E-step here - eliminates code duplication!

        if features_a_transformed is None:
            logging.error("features_a_transformed is None! This should not happen for transfer method.")
            raise RuntimeError("Transfer method failed: features_a_transformed not computed")

        logging.info(f"Transfer method: Using transformed features of shape {features_a_transformed.shape}")
        
        # Create transformed source data
        # For transfer method, use TRANSFERRED features (target expressions transferred to source)
        # This shows the actual result of the transfer method
        a_hat_cpu = features_a_transformed.detach().cpu().numpy()
        logging.info("Transfer method: Using TRANSFERRED source features for final UMAP (target expressions transferred to source)")
        # Use sketched observations if geosketch was applied
        if sketch_to_original is not None:
            obs_sketched = adata_a.obs.iloc[sketch_to_original].copy()
        else:
            obs_sketched = adata_a.obs.copy()
        adata_a_transformed = ad.AnnData(X=a_hat_cpu, obs=obs_sketched)
        adata_a_transformed.obs["type"] = "source_transformed"  # Back to transformed since we're showing transferred features
        
        # Create a dummy model for consistency (not used in transfer method)
        model = FeatureTransform(input_dim=d_a, output_dim=d_b, hidden_dim=hidden_dim, dropout=dropout).to(device_t)
        
        # Create mapping_final for transfer method (combine mappings from all batches)
        logging.info("Transfer method: Combining mappings from all batches...")
        mapping_final = np.full(n_a, -1, dtype=np.int64)  # Map from source to target
        
        for batch_idx, mapping_batch, target_batch_indices in batch_mappings:
            logging.info(f"Batch {batch_idx}: mapping_batch.shape={mapping_batch.shape}, target_batch_indices.shape={target_batch_indices.shape}")
            # mapping_batch maps source indices to target batch indices
            valid_mapping = mapping_batch >= 0
            if valid_mapping.any():
                # For valid mappings, map to original target indices
                valid_source_indices = np.where(valid_mapping)[0]  # Get source indices with valid mappings
                target_batch_indices_for_valid = mapping_batch[valid_mapping]  # Target batch indices for valid mappings
                mapped_targets = target_batch_indices[target_batch_indices_for_valid]  # Original target indices
                
                # Only update if we don't already have a mapping for this source index
                mask = mapping_final[valid_source_indices] == -1
                mapping_final[valid_source_indices[mask]] = mapped_targets[mask]
        
        # Create final concatenated data for transfer method
        adata_b_copy = adata_b.copy()
        adata_b_copy.obs["type"] = "target"
        
        concat_adata = ad.concat(
            [adata_a_transformed, adata_b_copy], axis=0,
            label="batch", keys=["source", "target"], index_unique="_",
        )
        
        # Set common_CT column correctly (if present)
        if cell_type_col in adata_a_transformed.obs.columns and cell_type_col in adata_b_copy.obs.columns:
            common_ct_values = pd.concat([adata_a_transformed.obs[cell_type_col], adata_b_copy.obs[cell_type_col]])
            concat_adata.obs[cell_type_col] = common_ct_values.values
        print(concat_adata.obs.head())
        
        # Compute UMAP for final visualization (using global metric parameter)
        try:
            rsc.tl.pca(concat_adata)
            rsc.pp.neighbors(concat_adata, use_rep='X', metric=metric)
            rsc.tl.umap(concat_adata)
            logging.info("Final UMAP computation successful")
        except Exception as e:
            logging.error(f"Final UMAP computation failed: {e}")
            # Continue without UMAP for final plots

    else:  # m_step_method == 'global'
        # Combine mappings from all batches
        logging.info("Combining mappings from all batches...")
        logging.info(f"n_a (sketched source size): {n_a}")
        logging.info(f"n_b_orig (original target size): {n_b_orig}")
        mapping_final = np.full(n_a, -1, dtype=np.int64)  # Map from source to target

        for batch_idx, mapping_batch, target_batch_indices in batch_mappings:
            logging.info(f"Batch {batch_idx}: mapping_batch.shape={mapping_batch.shape}, target_batch_indices.shape={target_batch_indices.shape}")
            # mapping_batch maps source indices to target batch indices
            valid_mapping = mapping_batch >= 0
            if valid_mapping.any():
                # For valid mappings, map to original target indices
                valid_source_indices = np.where(valid_mapping)[0]  # Get source indices with valid mappings
                target_batch_indices_for_valid = mapping_batch[valid_mapping]  # Target batch indices for valid mappings
                mapped_targets = target_batch_indices[target_batch_indices_for_valid]  # Original target indices

                # Only update if we don't already have a mapping for this source index
                mask = mapping_final[valid_source_indices] == -1
                mapping_final[valid_source_indices[mask]] = mapped_targets[mask]

        # Create final concatenated data for UMAP
        logging.info("Creating final concatenated data...")
        with torch.no_grad():
            model.eval()  # Set to eval mode for final inference

            # Variance diagnostics BEFORE final global transformation
            source_var_before = features_a.var(dim=0).mean().item()
            source_std_before = features_a.std(dim=0).mean().item()
            source_range_before = (features_a.max() - features_a.min()).item()
            logging.info(f"FINAL VARIANCE CHECK - Source BEFORE global transform: var={source_var_before:.6f}, std={source_std_before:.6f}, range={source_range_before:.6f}")

            a_hat_global = apply_model_with_scaling(features_a)

            # Variance diagnostics AFTER final global transformation
            trans_var_after = a_hat_global.var(dim=0).mean().item()
            trans_std_after = a_hat_global.std(dim=0).mean().item()
            trans_range_after = (a_hat_global.max() - a_hat_global.min()).item()
            logging.info(f"FINAL VARIANCE CHECK - Source AFTER global transform: var={trans_var_after:.6f}, std={trans_std_after:.6f}, range={trans_range_after:.6f}")
            logging.info(f"FINAL VARIANCE PRESERVATION RATIO (global): {trans_std_after/source_std_before:.4f}")

            a_hat_cpu = a_hat_global.detach().cpu().numpy()
        # Use sketched observations if geosketch was applied
        if sketch_to_original is not None:
            obs_sketched = adata_a.obs.iloc[sketch_to_original].copy()
        else:
            obs_sketched = adata_a.obs.copy()
        adata_a_transformed = ad.AnnData(X=a_hat_cpu, obs=obs_sketched)
        adata_a_transformed.obs["type"] = "source_transformed"  # Keep this for global method
        
        # Create final concatenated data
        adata_b_copy = adata_b.copy()
        adata_b_copy.obs["type"] = "target"
        
        concat_adata = ad.concat(
            [adata_a_transformed, adata_b_copy], axis=0,
            label="batch", keys=["source", "target"], index_unique="_",
        )
        
        # Set common_CT column correctly (if present)
        if cell_type_col in adata_a_transformed.obs.columns and cell_type_col in adata_b_copy.obs.columns:
            common_ct_values = pd.concat([adata_a_transformed.obs[cell_type_col], adata_b_copy.obs[cell_type_col]])
            concat_adata.obs[cell_type_col] = common_ct_values.values
        print(concat_adata.obs.head())
        
        # Compute UMAP for final visualization (using global metric parameter)
        try:
            rsc.tl.pca(concat_adata)
            rsc.pp.neighbors(concat_adata, use_rep='X', metric=metric)
            rsc.tl.umap(concat_adata)
            logging.info("Final UMAP computation successful")
        except Exception as e:
            logging.error(f"Final UMAP computation failed: {e}")
            # Continue without UMAP for final plots
    
    # Generate final debug plots using plot_utils
    if debug_plots_path:
        logging.info("Generating final debug plots...")

        # Final UMAP plot
        plot_dual_umap(
            concat_adata,
            cell_type_col=cell_type_col,
            title_prefix='Final UMAP',
            save_path=os.path.join(debug_plots_path, 'umap', 'umap_final.png'),
            type_palette={'source_transformed': '#1f77b4', 'target': '#ff7f0e'}
        )

        # Final weight heatmap
        plot_weight_heatmap(
            model=model,
            title='Final Transformation Weights',
            save_path=os.path.join(debug_plots_path, 'heatmap', 'W_final.png')
        )
    
    logging.info(f"Alignment completed. Final mapping: {np.sum(mapping_final >= 0)}/{len(mapping_final)} source cells mapped")

    # Reconstruct full transport matrix for output
    logging.info("Reconstructing full transport matrix for output...")
    T_full = torch.zeros(n_a, n_b_orig, device=device_t, dtype=torch.float32)

    for result in batch_results:
        T = result['T']
        source_indices = result['source_indices']
        target_indices = result['target_indices']

        if use_stratified_pairing:
            # Convert numpy indices to torch tensors for CUDA compatibility
            source_idx_tensor = torch.from_numpy(source_indices).long().to(device_t)
            target_idx_tensor = torch.from_numpy(target_indices).long().to(device_t)
            # For stratified pairing, place batch transport in full matrix
            T_full[source_idx_tensor[:, None], target_idx_tensor] = T
        else:
            # Convert numpy indices to torch tensors for CUDA compatibility
            target_idx_tensor = torch.from_numpy(target_indices).long().to(device_t)
            # For non-stratified, source covers all indices
            T_full[:, target_idx_tensor] = T

    logging.info(f"T_full reconstructed: shape {T_full.shape}, nnz={(T_full > 1e-6).sum().item()}")

    # Save model to disk (if debug_plots_path is provided and m_step_method is 'global')
    if debug_plots_path and m_step_method == 'global':
        from model_utils import save_alignment_model
        model_save_path = os.path.join(debug_plots_path, 'alignment_model.pt')
        save_alignment_model(
            model=model,
            save_path=model_save_path,
            feature_mean=global_feature_mean,
            feature_std=global_feature_std,
            gene_names=adata_a.var_names.tolist()
        )
        logging.info(f"Model saved to {model_save_path}")

    # Return scaler for applying model to new data (None if cosine metric was used)
    return model, mapping_final, concat_adata, T_full, global_feature_mean, global_feature_std
