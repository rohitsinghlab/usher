"""
E-step utilities for optimal transport computation.

This module provides functions to compute transport plans between source and target cells.
"""

import logging
from typing import Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
import ot
import ot.backend as otb

# Import graph utilities for GW/FGW
from graph_utils import compute_knn_graph_distance


def compute_transport_ot(
    features_source: torch.Tensor,
    features_target: torch.Tensor,
    aux_features_source: Optional[np.ndarray],
    aux_features_target: Optional[np.ndarray],
    gamma: float,
    epsilon: float,
    metric: str,
    balanced: bool,
    device: torch.device,
    iteration: int = 0,
    knn_mask: Optional[torch.Tensor] = None,
    knn_penalty: float = 5.0,
    warmstart_duals: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    verbose: bool = False
) -> torch.Tensor:
    """
    Compute optimal transport plan using Wasserstein distance.

    Parameters
    ----------
    features_source : torch.Tensor
        Source features (n_source, d), already normalized if needed
    features_target : torch.Tensor
        Target features (n_target, d), already normalized if needed
    aux_features_source : np.ndarray, optional
        Cell type probabilities for source (n_source, n_celltypes)
    aux_features_target : np.ndarray, optional
        Cell type probabilities for target (n_target, n_celltypes)
    gamma : float
        Weight for feature distance: M = (1-gamma)*M_aux + gamma*M_features
        If iteration=0, gamma is forced to 0.0
    epsilon : float
        Entropic regularization parameter for Sinkhorn
    metric : str
        Distance metric: 'cosine' or 'euclidean'
    balanced : bool
        Whether to use balanced OT (True) or unbalanced OT (False)
    device : torch.device
        Device for computation
    iteration : int
        Current iteration number (if 0, uses gamma=0 for pure cell type matching)

    Returns
    -------
    T : torch.Tensor
        Transport plan (n_source, n_target), row-normalized
    """
    n_source = features_source.shape[0]
    n_target = features_target.shape[0]

    # Force gamma=0 for iteration 0 (pure cell type matching)
    gamma_effective = 0.0 if iteration == 0 else gamma

    # ===== Compute Cost Matrix M =====
    if aux_features_source is not None and aux_features_target is not None:
        # Part 1: Auxiliary features distance (cell type probs OR spatial coords)
        aux_source_torch = torch.from_numpy(aux_features_source).to(device).float()
        aux_target_torch = torch.from_numpy(aux_features_target).to(device).float()
        M_aux = torch.cdist(aux_source_torch, aux_target_torch, p=2)

        # For spatial coordinates, normalize by max; for cell type probs, normalize by mean
        # Spatial coords have higher variance and need max normalization for stability
        # log if M_aux contains NaN values
        if torch.isnan(M_aux).any():
            logging.warning("M_aux contains NaN values.")
            
        M_aux = M_aux / (M_aux.max().clamp(min=1e-8))
        # log if M_aux contains NaN values
        if torch.isnan(features_source).any():
            logging.warning("features_source contains NaN values.")
        # Part 2: Feature distance
        if metric == 'cosine':
            logging.info(f"M_features metric: {metric}.")
            # features already normalized, compute cosine distance
            M_features = 1.0 - (features_source @ features_target.T)
        else:  # euclidean
            logging.info(f"M_features metric: {metric}.")
            M_features = torch.cdist(features_source, features_target, p=2)
        M_features = M_features / (M_features.max().clamp(min=1e-8))

        # Part 3: Combine
        M = (1.0 - gamma_effective) * M_aux + gamma_effective * M_features

        logging.info(f"  OT cost matrix: gamma={gamma_effective:.2f}, M_aux.mean={M_aux.mean():.4f}, M_features.mean={M_features.mean():.4f}")
    else:
        # Fallback: features only
        if metric == 'cosine':
            M = 1.0 - (features_source @ features_target.T)
        else:
            M = torch.cdist(features_source, features_target, p=2)
        M = M / (M.mean().clamp(min=1e-8))
        logging.warning("  OT: Cell type probabilities not available, using features only")

    # ===== Apply k-NN constraint for spatial windowing (if provided) =====
    if knn_mask is not None:
        # knn_mask is True for valid k-NN pairs, False for non-k-NN pairs
        # Apply penalty to non-k-NN matches
        penalty_value = M[knn_mask].median() * knn_penalty  # Scale penalty relative to valid costs
        M = M.clone()  # Avoid modifying the original
        M[~knn_mask] += penalty_value
        if verbose:
            print(f"Applied k-NN penalty: {penalty_value:.4f} to {(~knn_mask).sum()} non-k-NN pairs")

    # ===== Solve OT =====
    # Marginals (uniform)
    p = torch.ones(n_source, device=device, dtype=torch.float32) / n_source
    q = torch.ones(n_target, device=device, dtype=torch.float32) / n_target

    # Create TorchBackend for GPU acceleration
    backend = otb.TorchBackend()

    if balanced:
        # Balanced OT with TorchBackend (stays on GPU)
        if epsilon < 0.05:
            T = ot.bregman.sinkhorn_log(
                p, q, M,
                reg=epsilon, numItermax=1000,
                backend=backend
            )
        else:
            T = ot.bregman.sinkhorn(
                p, q, M,
                reg=epsilon, numItermax=1000,
                backend=backend
            )
        # T is already a torch tensor on the correct device
        if not isinstance(T, torch.Tensor):
            T = torch.as_tensor(T, device=device).float()
    else:
        # Unbalanced OT (no TorchBackend support - must use numpy)
        T = ot.unbalanced.sinkhorn_unbalanced(
            p.cpu().numpy(), q.cpu().numpy(), M.cpu().numpy(),
            reg=epsilon, reg_m=0.8, numItermax=1000
        )
        T = torch.from_numpy(T).to(device).float()

    # Row-normalize T (ensure each source sums to 1)
    if torch.isnan(T).any():
        logging.warning("T contains NaN values initially.")
    row_sums = T.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    if torch.isnan(row_sums).any():
        logging.warning("row_sums contains NaN values.")
    T = T / row_sums
    # Fill NaN values in T.
    if torch.isnan(T).any():
        num_nans = torch.isnan(T).sum().item()
        logging.warning(f"{num_nans} NaN values detected in OT transport matrix. Filling NaNs with zeros.")
        T = torch.nan_to_num(T, nan=0.0)
    logging.info(f"  OT: T stats - min={T.min():.8e}, max={T.max():.8e}, nnz={(T>1e-8).sum().item()}/{T.numel()}")

    return T


def compute_transport_gw(
    features_source: torch.Tensor,
    features_target: torch.Tensor,
    epsilon: float,
    knn_k: int,
    metric: str,
    use_knn_graph: bool,
    device: torch.device,
    verbose: bool = False
) -> torch.Tensor:
    """
    Compute optimal transport plan using Gromov-Wasserstein distance.

    Uses structure matrices C1, C2 computed from kNN graphs or direct distances.

    Parameters
    ----------
    features_source : torch.Tensor
        Source features (n_source, d), already normalized if needed
    features_target : torch.Tensor
        Target features (n_target, d), already normalized if needed
    epsilon : float
        Entropic regularization parameter
    knn_k : int
        Number of neighbors for kNN graph
    metric : str
        Distance metric: 'cosine' or 'euclidean'
    use_knn_graph : bool
        Whether to use kNN graph distances vs direct distances
    device : torch.device
        Device for computation


    Returns
    -------
    T : torch.Tensor
        Transport plan (n_source, n_target), row-normalized
    """
    n_source = features_source.shape[0]
    n_target = features_target.shape[0]

    # ===== Compute structure matrix C1 (source) =====
    if use_knn_graph:
        logging.info(f"  GW: Computing kNN graph structure for source (k={knn_k}, metric={metric})")
        C1 = compute_knn_graph_distance(
            features_source,
            k=knn_k,
            metric=metric,
            device=device,

        )
    else:
        # Direct pairwise distances
        C1 = torch.cdist(features_source, features_source, p=2)
        C1 = C1 / (C1.max() + 1e-8)

    # ===== Compute structure matrix C2 (target) =====
    if use_knn_graph:
        logging.info(f"  GW: Computing kNN graph structure for target (k={knn_k}, metric={metric})")
        C2 = compute_knn_graph_distance(
            features_target,
            k=knn_k,
            metric=metric,
            device=device,

        )
    else:
        # Direct pairwise distances
        C2 = torch.cdist(features_target, features_target, p=2)
        C2 = C2 / (C2.max() + 1e-8)

    # ===== Solve Gromov-Wasserstein =====
    # Marginals (uniform)
    p = torch.ones(n_source, device=device, dtype=torch.float32) / n_source
    q = torch.ones(n_target, device=device, dtype=torch.float32) / n_target

    logging.info(f"  GW: C1 shape={C1.shape}, C2 shape={C2.shape}, epsilon={epsilon}")
    logging.info(f"  GW: C1 stats - min={C1.min():.4f}, max={C1.max():.4f}, mean={C1.mean():.4f}")
    logging.info(f"  GW: C2 stats - min={C2.min():.4f}, max={C2.max():.4f}, mean={C2.mean():.4f}")

    # Solve entropic GW with TorchBackend (stays on GPU)
    T = ot.gromov.entropic_gromov_wasserstein(
        C1=C1, C2=C2,
        p=p, q=q,
        loss_fun='square_loss',
        epsilon=epsilon,
        max_iter=10000,
        tol=1e-6,
        verbose=verbose,
        log=False,
        backend='torch'  # Use PyTorch backend for GPU
    )

    # Ensure T is a torch tensor
    if not isinstance(T, torch.Tensor):
        T = torch.as_tensor(T, device=device).float()

    # Row-normalize T (ensure each source sums to 1)
    row_sums = T.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    T = T / row_sums

    logging.info(f"  GW: T stats - min={T.min():.8e}, max={T.max():.8e}, nnz={(T>1e-8).sum().item()}/{T.numel()}")

    return T


def compute_transport_fgw(
    features_source: torch.Tensor,
    features_target: torch.Tensor,
    aux_features_source: Optional[np.ndarray],
    aux_features_target: Optional[np.ndarray],
    gamma: float,
    epsilon: float,
    alpha: float,
    metric: str,
    knn_k: int,
    use_knn_graph: bool,
    device: torch.device,
    iteration: int = 0,
    verbose: bool = False
) -> torch.Tensor:
    """
    Compute optimal transport plan using Fused Gromov-Wasserstein distance.

    Combines structure preservation (GW) with feature alignment (Wasserstein).

    Parameters
    ----------
    features_source : torch.Tensor
        Source features (n_source, d), already normalized if needed
    features_target : torch.Tensor
        Target features (n_target, d), already normalized if needed
    aux_features_source : np.ndarray, optional
        Cell type probabilities for source (n_source, n_celltypes)
    aux_features_target : np.ndarray, optional
        Cell type probabilities for target (n_target, n_celltypes)
    gamma : float
        Weight for feature distance in M: M = (1-gamma)*M_aux + gamma*M_features
        If iteration=0, gamma is forced to 0.0
    epsilon : float
        Entropic regularization parameter
    alpha : float
        Balance between structure (GW) and features (Wasserstein)
        - alpha=1.0: pure GW (structure only)
        - alpha=0.0: pure Wasserstein (features only)
        - alpha=0.5: balanced FGW
    metric : str
        Distance metric: 'cosine' or 'euclidean'
    knn_k : int
        Number of neighbors for kNN graph
    use_knn_graph : bool
        Whether to use kNN graph distances vs direct distances
    device : torch.device
        Device for computation
    iteration : int
        Current iteration (if 0, uses gamma=0 for pure cell type matching)


    Returns
    -------
    T : torch.Tensor
        Transport plan (n_source, n_target), row-normalized
    """
    n_source = features_source.shape[0]
    n_target = features_target.shape[0]

    # Force gamma=0 for iteration 0 (pure cell type matching)
    gamma_effective = 0.0 if iteration == 0 else gamma

    # ===== PART 1: Compute structure matrices C1, C2 (same as GW) =====
    if use_knn_graph:
        logging.info(f"  FGW: Computing kNN graph structure for source (k={knn_k}, metric={metric})")
        C1 = compute_knn_graph_distance(
            features_source,
            k=knn_k,
            metric=metric,
            device=device,
        )
    else:
        C1 = torch.cdist(features_source, features_source, p=2)
        C1 = C1 / (C1.mean() + 1e-8)

    if use_knn_graph:
        logging.info(f"  FGW: Computing kNN graph structure for target (k={knn_k}, metric={metric})")
        C2 = compute_knn_graph_distance(
            features_target,
            k=knn_k,
            metric=metric,
            device=device,
        )
    else:
        C2 = torch.cdist(features_target, features_target, p=2)
        C2 = C2 / (C2.mean() + 1e-8)

    # ===== PART 2: Compute feature cost matrix M =====
    if aux_features_source is not None and aux_features_target is not None:
        # Part 2A: Cell type distance
        aux_source_torch = torch.from_numpy(aux_features_source).to(device).float()
        aux_target_torch = torch.from_numpy(aux_features_target).to(device).float()
        M_aux = torch.cdist(aux_source_torch, aux_target_torch, p=2)
        M_aux = M_aux / (M_aux.max().clamp(min=1e-8))
        # log if features_source contains NaN values
        if torch.isnan(features_source).any():
            logging.warning("features_source contains NaN values.")
        if M_aux.isnan().any():
            logging.warning("M_aux contains NaN values.")
        # Part 2: Feature distance
        if metric == 'cosine':
            logging.info(f"M_features metric: {metric}.")
            # features already normalized, compute cosine distance
            M_features = 1.0 - (features_source @ features_target.T)
        else:  # euclidean
            logging.info(f"M_features metric: {metric}.")
            M_features = torch.cdist(features_source, features_target, p=2)
        M_features = M_features / (M_features.max().clamp(min=1e-8))

        # Part 2C: Combine
        M = (1.0 - gamma_effective) * M_aux + gamma_effective * M_features

        logging.info(f"  FGW: gamma={gamma_effective:.2f}, M_aux.mean={M_aux.mean():.4f}, M_features.mean={M_features.mean():.4f}")
    else:
        # Fallback: features only
        if metric == 'cosine':
            M = 1.0 - (features_source @ features_target.T)
        else:
            M = torch.cdist(features_source, features_target, p=2)
        M = M / (M.max().clamp(min=1e-8))
        logging.warning("  FGW: Cell type probabilities not available, using features only for M")

    # ===== Solve Fused Gromov-Wasserstein =====
    # Marginals (uniform)
    p = torch.ones(n_source, device=device, dtype=torch.float32) / n_source
    q = torch.ones(n_target, device=device, dtype=torch.float32) / n_target

    logging.info(f"  FGW: alpha={alpha:.2f} (1.0=pure structure, 0.0=pure features), epsilon={epsilon}")
    logging.info(f"  FGW: C1 stats - min={C1.min():.4f}, max={C1.max():.4f}, mean={C1.mean():.4f}")
    logging.info(f"  FGW: C2 stats - min={C2.min():.4f}, max={C2.max():.4f}, mean={C2.mean():.4f}")
    logging.info(f"  FGW: M stats - min={M.min():.4f}, max={M.max():.4f}, mean={M.mean():.4f}")

    # Solve entropic FGW with TorchBackend (stays on GPU)
    T = ot.gromov.entropic_fused_gromov_wasserstein(
        M=M, C1=C1, C2=C2,
        p=p, q=q,
        loss_fun='square_loss',
        epsilon=epsilon,
        alpha=alpha,
        max_iter=10000,
        tol=1e-6,
        verbose=verbose,
        log=False,
        backend='torch'  # Use PyTorch backend for GPU
    )

    # Ensure T is a torch tensor
    if not isinstance(T, torch.Tensor):
        T = torch.as_tensor(T, device=device).float()

    # Row-normalize T (ensure each source sums to 1)
    row_sums = T.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    T = T / row_sums

    logging.info(f"  FGW: T stats - min={T.min():.8e}, max={T.max():.8e}, nnz={(T>1e-8).sum().item()}/{T.numel()}")

    return T


def apply_linear_assignment(
    T: torch.Tensor,
    gamma_effective: float,
    e_step_method: str,
    M: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Apply linear sum assignment (Hungarian algorithm) to sparsify transport plan.

    Parameters
    ----------
    T : torch.Tensor
        Soft transport plan (n_source, n_target)
    gamma_effective : float
        Effective gamma value (0 for pure cell type)
    e_step_method : str
        E-step method ('ot', 'gw', or 'fgw')
    M : torch.Tensor, optional
        Original cost matrix (if available, used when gamma=0)

    Returns
    -------
    T_sparse : torch.Tensor
        Sparsified transport plan (0 or 1 entries)
    row_indices : np.ndarray
        Source cell indices in matching
    col_indices : np.ndarray
        Target cell indices in matching
    """
    from scipy.optimize import linear_sum_assignment


    cost_matrix = -T.cpu().numpy()  # Negate because Hungarian minimizes

    # Solve linear sum assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Create sparse transport
    T_sparse = torch.zeros_like(T)
    T_sparse[row_indices, col_indices] = 1.0

    return T_sparse, row_indices, col_indices


def compute_transport_batch(
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    
    features_source_all: torch.Tensor,
    features_target_all: torch.Tensor,

    # New generic parameters (for spatial or other auxiliary features)
    auxiliary_features_source: Optional[np.ndarray] = None,
    auxiliary_features_target: Optional[np.ndarray] = None,

    gamma: float = 0.5,
    epsilon: float = 0.1,
    metric: str = 'cosine',
    balanced: bool = True,
    use_linear_assignment: bool = False,
    device: torch.device = None,
    iteration: int = 0,
    e_step_method: str = 'ot',
    entropy_percentile: float = 60.0,
    confidence_percentile: float = 40.0,
    # GW/FGW parameters
    knn_k: int = 15,
    use_knn_graph: bool = True,
    alpha: float = 0.5,
    verbose: bool = False,

    # Spatial-specific parameters
    knn_constraint_indices: Optional[np.ndarray] = None,  # k-NN indices for spatial windowing
    knn_penalty_weight: float = 5.0,  # Penalty for non-k-NN matches
    warmstart_duals: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # Dual potentials for warm-start
    sampling_strategy: str = 'celltype'  # Sampling strategy for transport constraints
) -> Dict:
    """
    Compute transport plan for a single batch/pair.

    This is a convenience wrapper that extracts the batch data and calls the appropriate
    transport solver (OT, GW, or FGW) based on e_step_method.

    Parameters
    ----------
    source_indices : np.ndarray
        Indices of source cells in this batch
    target_indices : np.ndarray
        Indices of target cells in this batch
    features_source_all : torch.Tensor
        Full source feature matrix
    features_target_all : torch.Tensor
        Full target feature matrix

    gamma : float
        Weight for feature distance in M: M = (1-gamma)*M_celltype + gamma*M_features (OT/FGW only)
    epsilon : float
        Entropic regularization parameter
    auxiliary_features_source : np.ndarray, optional
        Generic auxiliary features for source (e.g., cell type probs, spatial coords)
    auxiliary_features_target : np.ndarray, optional
        Generic auxiliary features for target (e.g., cell type probs, spatial coords)
    knn_constraint_indices : np.ndarray, optional
        k-NN indices for spatial windowing constraints (n_source_batch, k)
    knn_penalty_weight : float
        Penalty weight for non-k-NN matches in spatial windowing
    warmstart_duals : tuple of torch.Tensor, optional
        Dual potentials (u, v) for warm-starting Sinkhorn
    metric : str
        Distance metric ('cosine' or 'euclidean')
    balanced : bool
        Balanced OT flag (OT only, ignored for GW/FGW)
    use_linear_assignment : bool
        Whether to apply Hungarian algorithm to sparsify T
    device : torch.device
        Computation device
    iteration : int
        Current iteration (affects gamma: iteration 0 uses gamma=0)
    e_step_method : str
        E-step method: 'ot', 'gw', or 'fgw'
    entropy_percentile : float
        Percentile threshold for entropy (select bottom X% = low entropy)
    confidence_percentile : float
        Percentile threshold for confidence (select top X% = high confidence)
    knn_k : int
        Number of neighbors for kNN graph (GW/FGW only, default: 15)
    use_knn_graph : bool
        Whether to use kNN graph distances vs direct distances (GW/FGW only, default: True)
    alpha : float
        Balance between structure (GW) and features (Wasserstein) (FGW only, default: 0.5)
        - alpha=1.0: pure GW (structure only)
        - alpha=0.0: pure Wasserstein (features only)

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'T': transport plan (torch.Tensor)
        - 'source_indices': source indices (np.ndarray)
        - 'target_indices': target indices (np.ndarray)
        - 'row_indices': matched source indices within batch (if linear assignment)
        - 'col_indices': matched target indices within batch (if linear assignment)
        - 'focused_mask': boolean mask indicating focused cells (torch.Tensor)
    """


    # Handle device
    if device is None:
        device = features_source_all.device

    # Extract batch data
    # Convert numpy indices to torch tensors for CUDA compatibility
    source_idx_tensor = torch.from_numpy(source_indices).long().to(device)
    target_idx_tensor = torch.from_numpy(target_indices).long().to(device)

    features_source = features_source_all[source_idx_tensor]
    features_target = features_target_all[target_idx_tensor]

    # Extract auxiliary features (celltype probs or spatial coords) for this batch
    aux_features_source = None
    aux_features_target = None
    if auxiliary_features_source is not None:
        aux_features_source = auxiliary_features_source[source_indices]
    if auxiliary_features_target is not None:
        aux_features_target = auxiliary_features_target[target_indices]



    # Prepare k-NN mask for spatial windowing (if provided)
    knn_mask = None
    if knn_constraint_indices is not None and sampling_strategy == 'spatial':
        # Create boolean mask for k-NN constraints
        # knn_constraint_indices is (n_source_batch, k) with target indices
        n_source_batch = features_source.shape[0]
        n_target_batch = features_target.shape[0]
        knn_mask = torch.zeros(n_source_batch, n_target_batch, device=device, dtype=torch.bool)

        for i in range(n_source_batch):
            # Get k-NN targets for this source, map to batch-local indices
            knn_targets = knn_constraint_indices[i]
            # Find which of these are in the current target batch
            for target_idx in knn_targets:
                if target_idx in target_indices:
                    # Map global index to batch-local index
                    local_idx = np.where(target_indices == target_idx)[0]
                    if len(local_idx) > 0:
                        knn_mask[i, local_idx[0]] = True

    # Compute transport based on method
    if e_step_method == 'ot':
        T = compute_transport_ot(
            features_source, features_target,
            aux_features_source, aux_features_target,
            gamma, epsilon, metric, balanced, device, iteration,
            knn_mask=knn_mask,
            knn_penalty=knn_penalty_weight,
            warmstart_duals=warmstart_duals,
            verbose=verbose
        )
    elif e_step_method == 'gw':
        T = compute_transport_gw(
            features_source, features_target,
            epsilon=epsilon,
            knn_k=knn_k,
            metric=metric,
            use_knn_graph=use_knn_graph,
            device=device,
            verbose=verbose
        )
    elif e_step_method == 'fgw':
        T = compute_transport_fgw(
            features_source, features_target,
            aux_features_source, aux_features_target,
            gamma=gamma,
            epsilon=epsilon,
            alpha=alpha,
            metric=metric,
            knn_k=knn_k,
            use_knn_graph=use_knn_graph,
            device=device,
            iteration=iteration,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown e_step_method: {e_step_method}. Must be 'ot', 'gw', or 'fgw'.")

    # ===== Compute focused cells mask (BEFORE linear assignment) =====
    # This identifies source cells with low entropy (peaked T) and high confidence
    # These are the "good quality" matches we want to use for M-step training
    n_source = T.shape[0]

    # Compute entropy and confidence for each source cell
    T_safe = T.clamp(min=1e-10)
    entropy = -(T * torch.log(T_safe)).sum(dim=1)  # (n_source,) - HIGH entropy = flat/bad
    max_probs = T.max(dim=1)[0]  # (n_source,) - HIGH confidence = peaked/good

    # Adaptive thresholds based on percentiles
    entropy_threshold = torch.quantile(entropy, entropy_percentile / 100.0)
    confidence_threshold = torch.quantile(max_probs, confidence_percentile / 100.0)

    # Select cells meeting both criteria (LOW entropy AND HIGH confidence)
    low_entropy_mask = entropy <= entropy_threshold  # Bottom X% of entropy
    high_confidence_mask = max_probs >= confidence_threshold  # Top Y% of confidence
    focused_mask = low_entropy_mask & high_confidence_mask

    n_focused = focused_mask.sum().item()
    pct_focused = 100 * n_focused / n_source if n_source > 0 else 0.0

    logging.info(f"  Entropy filtering: {n_focused}/{n_source} cells focused ({pct_focused:.1f}%) " +
                 f"[entropy_th={entropy_threshold.item():.4f}, conf_th={confidence_threshold.item():.4f}]")

    # Check for NaN values in transport matrix
    if torch.isnan(T).any():
        logging.error(f"NaN detected in transport matrix at iteration {iteration}")
        logging.error(f"  Transport matrix shape: {T.shape}")
        logging.error(f"  Number of NaN values: {torch.isnan(T).sum().item()}")

    # Apply linear assignment if requested (happens AFTER we compute focused_mask)
    # Note: Linear assignment changes T, but we already identified focused cells from soft T
    row_indices = None
    col_indices = None
    if use_linear_assignment:
        gamma_effective = 0.0 if iteration == 0 else gamma
        T, row_indices, col_indices = apply_linear_assignment(
            T, gamma_effective, e_step_method, M=None
        )

        # ===== Cell Type Matching Accuracy (after linear assignment) =====
        if aux_features_source is not None and aux_features_target is not None and sampling_strategy == 'celltype':
            # Get predicted cell types (argmax of probabilities)
            source_celltypes = aux_features_source.argmax(axis=1)  # (n_source,)
            target_celltypes = aux_features_target.argmax(axis=1)  # (n_target,)

            # Accuracy for ALL matched pairs
            matched_source_types = source_celltypes[row_indices]
            matched_target_types = target_celltypes[col_indices]
            n_matches = len(row_indices)
            n_correct_all = (matched_source_types == matched_target_types).sum()
            pct_correct_all = 100 * n_correct_all / n_matches if n_matches > 0 else 0.0

            # Accuracy for FOCUSED cells only (used for M-step training)
            focused_mask_cpu = focused_mask.cpu().numpy()
            focused_row_indices = row_indices[focused_mask_cpu]
            focused_col_indices = col_indices[focused_mask_cpu]

            if len(focused_row_indices) > 0:
                focused_source_types = source_celltypes[focused_row_indices]
                focused_target_types = target_celltypes[focused_col_indices]
                n_matches_focused = len(focused_row_indices)
                n_correct_focused = (focused_source_types == focused_target_types).sum()
                pct_correct_focused = 100 * n_correct_focused / n_matches_focused

                logging.info(f"  Cell type accuracy (ALL): {n_correct_all}/{n_matches} correct ({pct_correct_all:.1f}%)")
                logging.info(f"  Cell type accuracy (FOCUSED/M-step): {n_correct_focused}/{n_matches_focused} correct ({pct_correct_focused:.1f}%)")
            else:
                logging.info(f"  Cell type accuracy (ALL): {n_correct_all}/{n_matches} correct ({pct_correct_all:.1f}%)")
                logging.info(f"  Cell type accuracy (FOCUSED/M-step): N/A (no focused cells)")

    return {
        'T': T,
        'source_indices': source_indices,
        'target_indices': target_indices,
        'row_indices': row_indices,
        'col_indices': col_indices,
        'focused_mask': focused_mask  # NEW: pass focused mask to M-step
    }
