"""
Spatial windowing utilities for optimal transport alignment.

This module handles spatial coordinate-based windowing and k-NN preparation
for memory-efficient spatial alignment.
"""

import logging
import numpy as np
import torch
import anndata as ad
from typing import List, Tuple, Optional, Dict, Any
from sklearn.neighbors import NearestNeighbors


def _create_spatial_windows(
    coords: np.ndarray,
    window_height: float,
    window_width: float,
    overlap: float = 0.1
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Create spatial grid windows with overlap for memory-efficient processing.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates (n_cells, 2)
    window_height : float
        Height of each window
    window_width : float
        Width of each window
    overlap : float
        Overlap fraction between windows (0-1)

    Returns
    -------
    windows : List[np.ndarray]
        List of indices for each window
    window_info : List[Dict]
        Metadata for each window (bounds, n_spots)
    """
    min_x, min_y = coords[:, 0].min(), coords[:, 1].min()
    max_x, max_y = coords[:, 0].max(), coords[:, 1].max()

    # Compute step size with overlap
    step_height = window_height * (1 - overlap)
    step_width = window_width * (1 - overlap)

    windows = []
    window_info = []

    y = min_y
    while y < max_y:
        x = min_x
        while x < max_x:
            # Define window bounds
            x_min, x_max = x, x + window_width
            y_min, y_max = y, y + window_height

            # Find spots in this window
            mask = (
                (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
                (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max)
            )
            indices = np.where(mask)[0]

            if len(indices) > 0:
                windows.append(indices)
                window_info.append({
                    'x_min': x_min, 'x_max': x_max,
                    'y_min': y_min, 'y_max': y_max,
                    'n_spots': len(indices)
                })

            x += step_width
        y += step_height

    logging.info(f"Created {len(windows)} spatial windows with {overlap*100:.0f}% overlap")
    if windows:
        avg_spots = np.mean([len(w) for w in windows])
        logging.info(f"Average spots per window: {avg_spots:.0f}")

    return windows, window_info


def _compute_spatial_knn(
    coords_source: np.ndarray,
    coords_target: np.ndarray,
    k: int = 50
) -> np.ndarray:
    """
    Compute k-nearest spatial neighbors from source to target.

    Parameters
    ----------
    coords_source : np.ndarray
        Source spatial coordinates (n_source, 2)
    coords_target : np.ndarray
        Target spatial coordinates (n_target, 2)
    k : int
        Number of nearest neighbors

    Returns
    -------
    knn_indices : np.ndarray
        k-NN indices (n_source, k) - for each source cell, indices of k nearest target cells
    """
    # Ensure we don't request more neighbors than available
    k = min(k, coords_target.shape[0])

    # Fit k-NN on target coordinates
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
    nn.fit(coords_target)

    # Find k-NN for each source coordinate
    _, knn_indices = nn.kneighbors(coords_source)

    return knn_indices


def _auto_compute_window_size(
    coords: np.ndarray,
    n_windows_target: int = 20
) -> Tuple[float, float]:
    """
    Automatically compute window dimensions based on spatial extent.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates (n_cells, 2)
    n_windows_target : int
        Target number of windows to create

    Returns
    -------
    window_height : float
        Computed window height
    window_width : float
        Computed window width
    """
    min_x, min_y = coords[:, 0].min(), coords[:, 1].min()
    max_x, max_y = coords[:, 0].max(), coords[:, 1].max()

    x_range = max_x - min_x
    y_range = max_y - min_y

    # Assuming roughly square windows
    windows_per_side = np.sqrt(n_windows_target)
    window_width = x_range / windows_per_side
    window_height = y_range / windows_per_side

    return window_height, window_width


def prepare_spatial_batches(
    adata_a: ad.AnnData,
    adata_b: ad.AnnData,
    spatial_key: str = 'X_spatial',
    window_height: Optional[float] = None,
    window_width: Optional[float] = None,
    window_overlap: float = 0.1,
    n_windows_target: int = 20,
    spatial_knn: int = 50,  # Number of spatial nearest neighbors for constraints
    **kwargs
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any], Dict[str, Any]]:
    """
    Prepare batches using spatial windowing strategy.

    Parameters
    ----------
    adata_a : ad.AnnData
        Source dataset with spatial coordinates
    adata_b : ad.AnnData
        Target dataset with spatial coordinates
    spatial_key : str
        Key in .obsm for spatial coordinates
    window_height : float, optional
        Height of spatial windows (auto-computed if None)
    window_width : float, optional
        Width of spatial windows (auto-computed if None)
    window_overlap : float
        Overlap fraction between windows
    n_windows_target : int
        Target number of windows if auto-computing dimensions
    spatial_knn : int
        Number of spatial nearest neighbors for transport constraints
    **kwargs
        Additional parameters (ignored)

    Returns
    -------
    batches : List[Tuple[np.ndarray, np.ndarray]]
        List of (source_indices, target_indices) pairs for each window
    auxiliary_data : Dict[str, Any]
        Dictionary containing:
        - coords_a: Source spatial coordinates
        - coords_b: Target spatial coordinates
        - n_source: Number of source cells
        - n_target: Number of target cells
    window_info : Dict[str, Any]
        Metadata about spatial windows:
        - windows: List of window indices
        - window_metadata: List of window bounds and statistics
        - window_height: Used window height
        - window_width: Used window width
    """
    # Extract spatial coordinates
    if spatial_key not in adata_a.obsm:
        raise ValueError(f"Spatial coordinates not found in source .obsm['{spatial_key}']")
    if spatial_key not in adata_b.obsm:
        raise ValueError(f"Spatial coordinates not found in target .obsm['{spatial_key}']")

    coords_a = adata_a.obsm[spatial_key].astype(np.float32)
    coords_b = adata_b.obsm[spatial_key].astype(np.float32)

    if coords_a.shape[1] != 2 or coords_b.shape[1] != 2:
        raise ValueError(f"Spatial coordinates must be 2D, got shapes {coords_a.shape} and {coords_b.shape}")

    logging.info(f"Loaded spatial coordinates: source {coords_a.shape}, target {coords_b.shape}")

    # Auto-compute window dimensions if not provided
    if window_height is None or window_width is None:
        window_height, window_width = _auto_compute_window_size(coords_a, n_windows_target)
        logging.info(f"Auto-computed window dimensions: height={window_height:.2f}, width={window_width:.2f}")

    # Create spatial windows for source
    windows, window_metadata = _create_spatial_windows(
        coords_a, window_height, window_width, window_overlap
    )

    # Compute spatial k-NN from source to target
    logging.info(f"Computing spatial k-NN (k={spatial_knn}) from source to target...")
    knn_indices = _compute_spatial_knn(coords_a, coords_b, k=spatial_knn)
    logging.info(f"Computed spatial k-NN indices: shape {knn_indices.shape}")

    # Create batches from spatial windows
    # For affine-aligned data, use the SAME window bounds for both source and target
    batches = []
    for window_idx, source_window_indices in enumerate(windows):
        window_meta = window_metadata[window_idx]
        x_min, x_max = window_meta['x_min'], window_meta['x_max']
        y_min, y_max = window_meta['y_min'], window_meta['y_max']

        # Find target cells within the SAME bounds (NO padding for affine-aligned data!)
        target_mask = (
            (coords_b[:, 0] >= x_min) & (coords_b[:, 0] <= x_max) &
            (coords_b[:, 1] >= y_min) & (coords_b[:, 1] <= y_max)
        )
        target_indices = np.where(target_mask)[0]

        # Skip if either window is empty
        if len(source_window_indices) == 0 or len(target_indices) == 0:
            continue

        batches.append((source_window_indices, target_indices))

    logging.info(f"Created {len(batches)} spatial window batches")
    if batches:
        avg_source = np.mean([len(s) for s, _ in batches])
        avg_target = np.mean([len(t) for _, t in batches])
        logging.info(f"Average batch size: {avg_source:.0f} source, {avg_target:.0f} target cells")

        # Debug: Check if source and target have similar spatial distributions
        if abs(avg_source - avg_target) > 0.1 * avg_source:  # More than 10% difference
            logging.warning(f"Source and target windows have different cell counts!")
            logging.info(f"  Source spatial range: X[{coords_a[:, 0].min():.1f}, {coords_a[:, 0].max():.1f}], Y[{coords_a[:, 1].min():.1f}, {coords_a[:, 1].max():.1f}]")
            logging.info(f"  Target spatial range: X[{coords_b[:, 0].min():.1f}, {coords_b[:, 0].max():.1f}], Y[{coords_b[:, 1].min():.1f}, {coords_b[:, 1].max():.1f}]")

    # Prepare auxiliary data
    auxiliary_data = {
        'coords_a': coords_a,
        'coords_b': coords_b,
        'knn_indices': knn_indices,  # Spatial k-NN indices for transport constraints
        'n_source': len(coords_a),
        'n_target': len(coords_b)
    }

    # Window information
    window_info = {
        'windows': windows,
        'window_metadata': window_metadata,
        'window_height': window_height,
        'window_width': window_width,
        'overlap': window_overlap
    }

    return batches, auxiliary_data, window_info