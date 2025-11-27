"""
Graph utilities for spatial alignment.

This module provides graph-based distance computations for alignment algorithms.
"""

import logging
from typing import Optional
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors


def compute_knn_graph_distance(
    features: torch.Tensor,
    k: int = 5,
    metric: str = 'cosine',
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute kNN graph-based shortest path distances between features.

    Optionally applies cell-type aware distance penalty:
    d'(x,y) = d(x,y) · (1 + λ·𝟙[different_types])

    This encourages same-type cells to be closer in the structure matrix.

    Parameters
    ----------
    features : torch.Tensor
        Feature matrix (n_samples, n_features)
    k : int
        Number of nearest neighbors for kNN graph
    metric : str
        Distance metric for kNN ('cosine' or 'euclidean')
    device : str, optional
        Device to use for computation
    cell_types : np.ndarray, optional
        Cell type labels (n_samples,). If provided, applies penalty to cross-type distances.
    cell_type_penalty : float
        Penalty factor λ for different cell types. Default 0.5 means cross-type distances
        are 1.5x larger. Set to 0 to disable.

    Returns
    -------
    D : torch.Tensor
        Distance matrix (n_samples, n_samples) with shortest path distances
    """
    if device is None:
        device = features.device

    features_np = features.cpu().numpy()
    n_samples = features_np.shape[0]

    # Handle small datasets
    k = min(k, n_samples - 1)

    # Compute kNN graph
    if metric == 'cosine':
        # For cosine similarity, we need to normalize features
        features_norm = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute').fit(features_norm)
    else:
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm='auto').fit(features_np)

    # Get kNN graph (distances and indices)
    distances, indices = nbrs.kneighbors(features_np)

    # Create adjacency matrix for kNN graph
    # distances[i, j] is the distance from point i to its j-th nearest neighbor
    # indices[i, j] is the index of the j-th nearest neighbor of point i

    # Build sparse adjacency matrix
    row_indices = []
    col_indices = []
    edge_weights = []

    for i in range(n_samples):
        for j in range(1, k+1):  # Skip j=0 (self)
            neighbor_idx = indices[i, j]
            #distance = distances[i, j]

            # Add edge (i, neighbor_idx) with weight distance
            row_indices.append(i)
            col_indices.append(neighbor_idx)
            distance = 1
            edge_weights.append(distance)

            # Add symmetric edge (neighbor_idx, i) with same weight
            row_indices.append(neighbor_idx)
            col_indices.append(i)
            edge_weights.append(distance)

    # Create sparse adjacency matrix
    adjacency_matrix = csr_matrix(
        (edge_weights, (row_indices, col_indices)),
        shape=(n_samples, n_samples)
    )

    # Compute shortest path distances
    try:
        dist_matrix = shortest_path(
            adjacency_matrix,
            directed=False,
            unweighted=False,
            return_predecessors=False
        )
        print("Max. shortest path distance:", np.max(dist_matrix))
    except Exception as e:
        logging.warning(f"Shortest path computation failed: {e}. Using direct distances.")
        # Fallback: use direct cosine distances
        if metric == 'cosine':
            # Compute cosine similarity matrix
            cosine_sim = features_norm @ features_norm.T
            # Convert to cosine distance (1 - cosine_similarity)
            dist_matrix = 1.0 - cosine_sim
            # Ensure diagonal is 0
            np.fill_diagonal(dist_matrix, 0.0)
        else:
            # Compute Euclidean distance matrix
            dist_matrix = np.sqrt(((features_np[:, np.newaxis, :] - features_np[np.newaxis, :, :]) ** 2).sum(axis=2))

    # Handle disconnected components: replace infinities with max finite distance
    # This makes disconnected nodes "as far as possible" rather than adding penalty
    Max_dist = np.nanmax(dist_matrix[dist_matrix != np.inf])
    if np.isnan(Max_dist):
        Max_dist = 1.0

    # Replace infinities with max distance (disconnected = maximally far)
    n_infinite = np.isinf(dist_matrix).sum()
    if n_infinite > 0:
        dist_matrix[dist_matrix > Max_dist] = Max_dist
        logging.info(f"  Replaced {n_infinite} infinite distances with max finite distance {Max_dist:.4f}")

    # Normalize by maximum distance
    dist_matrix = dist_matrix / dist_matrix.max()

    # Convert back to torch tensor
    D = torch.from_numpy(dist_matrix).to(device).float()

    return D


def apply_cell_type_constraints(
    T: torch.Tensor,
    cell_types_source: np.ndarray,
    cell_types_target: np.ndarray,
    device: torch.device
) -> torch.Tensor:
    """
    Apply cell type constraints to transport plan by zeroing out cross-type matches.

    Parameters
    ----------
    T : torch.Tensor
        Transport plan (n_source x n_target)
    cell_types_source : np.ndarray
        Cell type labels for source cells
    cell_types_target : np.ndarray
        Cell type labels for target cells
    device : torch.device
        Device for computation

    Returns
    -------
    T_constrained : torch.Tensor
        Transport plan with cross-type matches zeroed out and re-normalized
    """
    n_source, n_target = T.shape

    # Create a mask for same-type matches
    # This is vectorized: for each (i,j) pair, check if cell_type_source[i] == cell_type_target[j]
    cell_types_source_expanded = np.tile(cell_types_source.reshape(-1, 1), (1, n_target))
    cell_types_target_expanded = np.tile(cell_types_target.reshape(1, -1), (n_source, 1))
    same_type_mask = (cell_types_source_expanded == cell_types_target_expanded)

    # Convert to torch tensor
    same_type_mask_tensor = torch.from_numpy(same_type_mask).to(device)

    # Zero out cross-type matches
    T_constrained = T * same_type_mask_tensor

    # Re-normalize: ensure rows sum to the same values as before (preserve marginals as much as possible)
    # Compute original row sums
    row_sums_original = T.sum(dim=1, keepdim=True)
    row_sums_constrained = T_constrained.sum(dim=1, keepdim=True)

    # Avoid division by zero
    row_sums_constrained = row_sums_constrained.clamp(min=1e-10)

    # Scale rows to match original sums
    T_constrained = T_constrained * (row_sums_original / row_sums_constrained)

    # Log statistics
    n_matches_before = (T > 1e-6).sum().item()
    n_matches_after = (T_constrained > 1e-6).sum().item()
    n_zeroed = n_matches_before - n_matches_after

    if n_matches_before > 0:
        logging.info(f"Cell type constraints: {n_zeroed}/{n_matches_before} matches zeroed out ({100*n_zeroed/n_matches_before:.1f}%)")
    else:
        logging.info(f"Cell type constraints: No matches in transport plan (all values below threshold)")

    return T_constrained
