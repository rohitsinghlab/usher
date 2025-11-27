"""
Geosketch utilities for spatial alignment.

This module provides geosketch-based sampling functions for source and target datasets.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import torch
import anndata as ad
from sklearn.decomposition import PCA


def geosketch_subsample(
    adata: ad.AnnData,
    sketch_size: int,
    obsm_key: str = 'X_umap',
    pca_components: int = 50,
    seed: int = 42
) -> np.ndarray:
    """
    Geosketch subsample large dataset.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing the dataset
    sketch_size : int
        Target number of cells to sketch
    obsm_key : str
        Key in .obsm for pre-computed embeddings (default: 'X_umap')
        If not found, will compute PCA using pca_components
    pca_components : int
        Number of PCA components to compute if obsm_key not found (default: 50)
    seed : int
        Random seed for geosketch

    Returns
    -------
    sketched_indices : np.ndarray
        Indices of sketched cells in original data
    """
    # Import gs function from standalone module
    from geosketch_standalone import gs

    # Get embeddings for geosketch
    if obsm_key in adata.obsm.keys():
        features_for_sketch = adata.obsm[obsm_key]
        logging.info(f"Using pre-computed embeddings from adata.obsm['{obsm_key}'] for geosketch: {features_for_sketch.shape}")
    else:
        # Fallback to PCA
        logging.warning(f"Key '{obsm_key}' not found in adata.obsm, computing PCA with {pca_components} components")
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        pca = PCA(n_components=min(pca_components, X.shape[1]))
        features_for_sketch = pca.fit_transform(X)
        logging.info(f"Computed PCA for geosketch: {X.shape} -> {features_for_sketch.shape}")

    # Geosketch on embedding space
    sketched_indices = gs(features_for_sketch, N=sketch_size, seed=seed)

    return np.array(sketched_indices)


def geosketch_target_batches(
    adata: ad.AnnData,
    batch_size: int,
    obsm_key: str = 'X_umap',
    pca_components: int = 50,
    seed: int = 42
) -> List[np.ndarray]:
    """
    Geosketch target into batches for processing.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing the target dataset
    batch_size : int
        Target batch size
    obsm_key : str
        Key in .obsm for pre-computed embeddings (default: 'X_umap')
        If not found, will compute PCA using pca_components
    pca_components : int
        Number of PCA components to compute if obsm_key not found (default: 50)
    seed : int
        Random seed for geosketch

    Returns
    -------
    batch_indices_list : List[np.ndarray]
        List of indices for each batch
    """
    # Import gs function from standalone module
    from geosketch_standalone import gs

    n_target = adata.n_obs

    if n_target <= batch_size:
        # Target is small enough, return single batch
        return [np.arange(n_target)]

    # Get embeddings for geosketch
    if obsm_key in adata.obsm.keys():
        features_for_sketch = adata.obsm[obsm_key]
        logging.info(f"Using pre-computed embeddings from adata.obsm['{obsm_key}'] for target batching: {features_for_sketch.shape}")
    else:
        # Fallback to PCA
        logging.warning(f"Key '{obsm_key}' not found in adata.obsm, computing PCA with {pca_components} components")
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        pca = PCA(n_components=min(pca_components, X.shape[1]))
        features_for_sketch = pca.fit_transform(X)
        logging.info(f"Computed PCA for target batching: {X.shape} -> {features_for_sketch.shape}")

    # Calculate number of batches needed
    n_batches = (n_target + batch_size - 1) // batch_size

    # Geosketch to get representative indices for each batch
    batch_indices_list = []
    remaining_indices = np.arange(n_target)

    for batch_idx in range(n_batches):
        if len(remaining_indices) <= batch_size:
            # Last batch: take all remaining
            batch_indices = remaining_indices
        else:
            # Geosketch from remaining indices
            remaining_features_for_sketch = features_for_sketch[remaining_indices]
            batch_size_actual = min(batch_size, len(remaining_indices))

            sketched_local = gs(remaining_features_for_sketch, N=batch_size_actual, seed=seed + batch_idx)
            batch_indices = remaining_indices[sketched_local]

        batch_indices_list.append(batch_indices)
        remaining_indices = np.setdiff1d(remaining_indices, batch_indices)

    return batch_indices_list


def geosketch_stratified_pairing(
    adata_a: ad.AnnData,
    adata_b: ad.AnnData,
    sketch_size: int,
    obsm_key: str = 'X_umap',
    pca_components: int = 50,
    seed: int = 42,
    fix: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Geosketch both source and target into k groups, then pair them randomly.

    Uses stratified sampling without replacement: iteratively geosketch k groups
    from source and target (removing sampled cells each iteration), then pair them
    as (source_group[i], target_group[i]) for i=0..k-1.

    Groups with fewer than sketch_size cells are dropped for BOTH source and target
    to maintain balanced pairing.

    Parameters
    ----------
    adata_a : ad.AnnData
        Source AnnData object
    adata_b : ad.AnnData
        Target AnnData object
    sketch_size : int
        Number of cells per group (same for source and target)
    obsm_key : str
        Key in .obsm containing pre-computed embeddings (e.g., 'X_umap', 'X_pca')
        If not found, will compute PCA using pca_components
    pca_components : int
        Number of PCA components to compute if obsm_key not found (default: 50)
    seed : int
        Random seed for geosketch
        - If fix=True: Use this seed directly (same groups across calls)
        - If fix=False: Use iteration-dependent seed (e.g., seed=base_seed+iteration)
    fix : bool
        If True, use fixed seed (same groups across calls with same seed)
        If False, caller should pass iteration-dependent seed for rotating groups

    Returns
    -------
    source_groups : List[np.ndarray]
        List of k source cell indices for each group (each has sketch_size cells)
    target_groups : List[np.ndarray]
        List of k target cell indices for each group (each has sketch_size cells)
        Paired with source_groups: align source_groups[i] with target_groups[i]

    Examples
    --------
    # Fixed groups (same across iterations):
    >>> source_groups, target_groups = geosketch_stratified_pairing(
    ...     adata_a, adata_b, sketch_size=1000, obsm_key='X_umap', seed=42, fix=True
    ... )

    # Rotating groups (different each iteration):
    >>> for iteration in range(n_iters):
    ...     source_groups, target_groups = geosketch_stratified_pairing(
    ...         adata_a, adata_b, sketch_size=1000, obsm_key='X_umap',
    ...         seed=42 + iteration, fix=False
    ...     )
    """
    from geosketch_standalone import gs

    n_source = adata_a.n_obs
    n_target = adata_b.n_obs

    # Calculate max possible k (number of complete groups)
    k_source_max = n_source // sketch_size
    k_target_max = n_target // sketch_size
    k = min(k_source_max, k_target_max)

    if k == 0:
        logging.warning(f"Cannot create any complete groups: n_source={n_source}, n_target={n_target}, sketch_size={sketch_size}")
        return [], []

    logging.info(f"Stratified pairing: Creating {k} paired groups (sketch_size={sketch_size}, fix={fix})")
    logging.info(f"  Source: {n_source} cells -> {k} groups of {sketch_size} cells ({k*sketch_size}/{n_source} used, {n_source - k*sketch_size} dropped)")
    logging.info(f"  Target: {n_target} cells -> {k} groups of {sketch_size} cells ({k*sketch_size}/{n_target} used, {n_target - k*sketch_size} dropped)")

    # Get embeddings for geosketch
    # Try to use pre-computed embeddings from .obsm
    if obsm_key in adata_a.obsm.keys():
        features_a_for_sketch = adata_a.obsm[obsm_key]
        logging.info(f"Using pre-computed embeddings from adata_a.obsm['{obsm_key}'] for source: {features_a_for_sketch.shape}")
    else:
        # Fallback to PCA
        logging.warning(f"Key '{obsm_key}' not found in adata_a.obsm, computing PCA with {pca_components} components")
        from sklearn.decomposition import PCA
        X_a = adata_a.X.toarray() if hasattr(adata_a.X, 'toarray') else adata_a.X
        pca_a = PCA(n_components=min(pca_components, X_a.shape[1]))
        features_a_for_sketch = pca_a.fit_transform(X_a)
        logging.info(f"Computed PCA for source: {X_a.shape} -> {features_a_for_sketch.shape}")

    if obsm_key in adata_b.obsm.keys():
        features_b_for_sketch = adata_b.obsm[obsm_key]
        logging.info(f"Using pre-computed embeddings from adata_b.obsm['{obsm_key}'] for target: {features_b_for_sketch.shape}")
    else:
        # Fallback to PCA
        logging.warning(f"Key '{obsm_key}' not found in adata_b.obsm, computing PCA with {pca_components} components")
        from sklearn.decomposition import PCA
        X_b = adata_b.X.toarray() if hasattr(adata_b.X, 'toarray') else adata_b.X
        pca_b = PCA(n_components=min(pca_components, X_b.shape[1]))
        features_b_for_sketch = pca_b.fit_transform(X_b)
        logging.info(f"Computed PCA for target: {X_b.shape} -> {features_b_for_sketch.shape}")

    # Stratified sampling: iteratively geosketch k groups from source
    source_groups = []
    remaining_source_indices = np.arange(n_source)

    for group_idx in range(k):
        # Geosketch from remaining source cells
        remaining_source_features = features_a_for_sketch[remaining_source_indices]

        # Use different seed for each group to ensure diversity
        group_seed = seed + group_idx if not fix else seed + group_idx
        sketched_local = gs(remaining_source_features, N=sketch_size, seed=group_seed)

        # Convert local indices to global indices
        group_indices = remaining_source_indices[sketched_local]
        source_groups.append(group_indices)

        # Remove sketched cells from remaining pool
        remaining_source_indices = np.setdiff1d(remaining_source_indices, group_indices)

        logging.info(f"  Source group {group_idx + 1}/{k}: sketched {len(group_indices)} cells, {len(remaining_source_indices)} remaining")

    # Stratified sampling: iteratively geosketch k groups from target
    target_groups = []
    remaining_target_indices = np.arange(n_target)

    for group_idx in range(k):
        # Geosketch from remaining target cells
        remaining_target_features = features_b_for_sketch[remaining_target_indices]

        # Use different seed offset for target to avoid correlated sampling
        group_seed = seed + k + group_idx if not fix else seed + k + group_idx
        sketched_local = gs(remaining_target_features, N=sketch_size, seed=group_seed)

        # Convert local indices to global indices
        group_indices = remaining_target_indices[sketched_local]
        target_groups.append(group_indices)

        # Remove sketched cells from remaining pool
        remaining_target_indices = np.setdiff1d(remaining_target_indices, group_indices)

        logging.info(f"  Target group {group_idx + 1}/{k}: sketched {len(group_indices)} cells, {len(remaining_target_indices)} remaining")

    # Verify all groups have exactly sketch_size cells
    for i, (src_group, tgt_group) in enumerate(zip(source_groups, target_groups)):
        assert len(src_group) == sketch_size, f"Source group {i} has {len(src_group)} cells, expected {sketch_size}"
        assert len(tgt_group) == sketch_size, f"Target group {i} has {len(tgt_group)} cells, expected {sketch_size}"

    logging.info(f"Stratified pairing complete: {k} paired groups created")

    return source_groups, target_groups
