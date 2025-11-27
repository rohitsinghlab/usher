"""
Celltype-based sampling utilities for optimal transport alignment.

This module handles celltype probability loading and geosketch-based batching
strategies for the alignment process.
"""

import logging
import numpy as np
import anndata as ad
from typing import List, Tuple, Optional, Dict, Any
from gs_utils import geosketch_subsample, geosketch_target_batches, geosketch_stratified_pairing

import pandas as pd

def _load_celltype_probs(
    adata: ad.AnnData,
    celltype_probs_layer: str,
    cell_type_col: str,
    subset_indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Load cell type probabilities from AnnData object.

    Tries to load from adata.obsm[celltype_probs_layer] first.
    If not present, creates one-hot encoding from adata.obs[cell_type_col].

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing cell type information
    celltype_probs_layer : str
        Key in .obsm to look for cell type probabilities
    cell_type_col : str
        Column in .obs to use for one-hot encoding if celltype_probs_layer not found
    subset_indices : np.ndarray, optional
        Indices to subset the cell type data (after loading)

    Returns
    -------
    celltype_probs : np.ndarray
        Cell type probabilities matrix (n_cells, n_celltypes)
    """
    if celltype_probs_layer in adata.obsm:
        # Load from obsm (DataFrame or array)
        celltype_data = adata.obsm[celltype_probs_layer]
        if isinstance(celltype_data, pd.DataFrame):
            celltype_probs = celltype_data.values
        else:
            celltype_probs = np.array(celltype_data)
        logging.info(f"Loaded cell type probabilities from .obsm['{celltype_probs_layer}']: shape {celltype_probs.shape}")
    else:
        # One-hot encode from categorical column
        logging.info(f"Cell type probabilities not found in .obsm['{celltype_probs_layer}'], one-hot encoding from .obs['{cell_type_col}']")
        if cell_type_col not in adata.obs:
            raise ValueError(f"Cell type column '{cell_type_col}' not found in .obs and '{celltype_probs_layer}' not found in .obsm")

        # Get cell types and create one-hot encoding
        cell_types = adata.obs[cell_type_col].values
        unique_types = np.unique(cell_types)
        n_types = len(unique_types)
        type_to_idx = {ct: i for i, ct in enumerate(unique_types)}

        celltype_probs = np.zeros((len(cell_types), n_types), dtype=np.float32)
        for i, ct in enumerate(cell_types):
            celltype_probs[i, type_to_idx[ct]] = 1.0

        logging.info(f"One-hot encoded cell types: {len(unique_types)} types, shape {celltype_probs.shape}")

    # Subset if requested
    if subset_indices is not None:
        celltype_probs = celltype_probs[subset_indices]
        logging.info(f"Subsetted cell type probabilities: shape {celltype_probs.shape}")

    return celltype_probs



def prepare_celltype_batches(
    adata_a: ad.AnnData,
    adata_b: ad.AnnData,
    sketch_size: int = 1000,
    use_stratified_pairing: bool = False,
    stratified_pairing_fix: bool = True,
    celltype_probs_layer: str = 'X_celltype_probs',
    cell_type_col: Optional[str] = None,
    sketch_obsm_key: str = 'X_umap',
    sketch_pca_components: int = 50,
    e_step_method: str = 'ot',
    seed: int = 2025,
    **kwargs
) -> Tuple[List[Tuple[Optional[np.ndarray], np.ndarray]], Dict[str, Any], Optional[np.ndarray]]:
    """
    Prepare batches for celltype-based sampling strategy.

    Parameters
    ----------
    adata_a : ad.AnnData
        Source dataset
    adata_b : ad.AnnData
        Target dataset
    sketch_size : int
        Group/batch size for sampling
    use_stratified_pairing : bool
        Whether to use stratified pairing (paired source-target groups)
    stratified_pairing_fix : bool
        Whether to use fixed groups across iterations
    celltype_probs_layer : str
        Key in .obsm for cell type probabilities
    cell_type_col : str, optional
        Column name containing cell type information
    sketch_obsm_key : str
        Key in .obsm for embeddings to use for geosketch
    sketch_pca_components : int
        Number of PCA components if sketch_obsm_key not found
    e_step_method : str
        E-step method ('ot', 'gw', or 'fgw') - determines if celltype probs are loaded
    seed : int
        Random seed for reproducibility
    **kwargs
        Additional parameters (ignored)

    Returns
    -------
    batches : List[Tuple[Optional[np.ndarray], np.ndarray]]
        List of (source_indices, target_indices) pairs
    auxiliary_data : Dict[str, Any]
        Dictionary containing:
        - celltype_probs_a: Source celltype probabilities (if applicable)
        - celltype_probs_b: Target celltype probabilities (if applicable)
        - source_groups: Source group indices (if stratified pairing)
        - target_groups: Target group indices (if stratified pairing)
        - n_source: Number of source cells after sampling
        - n_target: Number of target cells
    sketch_to_original : np.ndarray or None
        Mapping from sketched to original indices for source (if subsampled)
    """
    n_a_orig = len(adata_a)
    n_b_orig = len(adata_b)

    # Load celltype probabilities if needed for OT or FGW methods
    celltype_probs_a = None
    celltype_probs_b = None
    if e_step_method in ['ot', 'fgw']:
        logging.info(f"Loading cell type probabilities for {e_step_method} method...")
        celltype_probs_a = _load_celltype_probs(adata_a, celltype_probs_layer, cell_type_col)
        celltype_probs_b = _load_celltype_probs(adata_b, celltype_probs_layer, cell_type_col)

        if celltype_probs_a is not None:
            logging.info(f"Source cell type probs: {celltype_probs_a.shape}")
        if celltype_probs_b is not None:
            logging.info(f"Target cell type probs: {celltype_probs_b.shape}")

    batches = []
    sketch_to_original = None
    source_groups = None
    target_groups = None

    if use_stratified_pairing:
        # STRATIFIED PAIRING: Geosketch both source and target into k paired groups
        logging.info(f"=== Using STRATIFIED PAIRING sampling strategy ===")
        logging.info(f"Sketch size: {sketch_size}, Fix: {stratified_pairing_fix}")

        source_groups, target_groups = geosketch_stratified_pairing(
            adata_a, adata_b,
            sketch_size=sketch_size,
            obsm_key=sketch_obsm_key,
            pca_components=sketch_pca_components,
            seed=seed,
            fix=stratified_pairing_fix
        )

        logging.info(f"Stratified pairing: {len(source_groups)} paired groups created")

        # Create batch pairs
        batches = list(zip(source_groups, target_groups))
        n_source = n_a_orig  # Using full source dataset

    else:
        # ORIGINAL APPROACH: Sample source once, batch target
        logging.info(f"=== Using ORIGINAL sampling strategy ===")
        logging.info(f"Source sketch size: {sketch_size}, Target batch size: {sketch_size}")

        # 1. Sample source (reference) to reduce size
        if n_a_orig > sketch_size:
            logging.info(f"Geosketching source from {n_a_orig} to {sketch_size} cells...")
            sketch_to_original = geosketch_subsample(
                adata_a, sketch_size, obsm_key=sketch_obsm_key, pca_components=sketch_pca_components
            )
            n_source = len(sketch_to_original)

            # Subset celltype probabilities for source
            if celltype_probs_a is not None:
                celltype_probs_a = celltype_probs_a[sketch_to_original]
                logging.info(f"Subsetted source cell type probs to {celltype_probs_a.shape}")
        else:
            n_source = n_a_orig
            logging.info(f"No source sampling (size {n_a_orig} <= {sketch_size})")

        # 2. Sample target into batches
        target_batches = geosketch_target_batches(
            adata_b, sketch_size, obsm_key=sketch_obsm_key, pca_components=sketch_pca_components
        )

        logging.info(f"Created {len(target_batches)} target batches")

        # Create batch pairs (None for source means use all source cells)
        batches = [(None, target_batch) for target_batch in target_batches]

    # Prepare auxiliary data
    auxiliary_data = {
        'celltype_probs_a': celltype_probs_a,
        'celltype_probs_b': celltype_probs_b,
        'source_groups': source_groups,
        'target_groups': target_groups,
        'n_source': n_source,
        'n_target': n_b_orig
    }

    return batches, auxiliary_data, sketch_to_original