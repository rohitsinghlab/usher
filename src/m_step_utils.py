"""
M-step utilities for learning feature transformations.

This module provides functions to train models or apply transfer methods
using transport plans from the E-step.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_feature_scaler(
    features_target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std for each feature column from target dataset.

    Parameters
    ----------
    features_target : torch.Tensor
        Target feature matrix (n_target, d)

    Returns
    -------
    feature_mean : torch.Tensor
        Mean for each feature column (1, d)
    feature_std : torch.Tensor
        Std for each feature column (1, d)
    """
    # Compute per-feature statistics along dim=0 (across cells)
    feature_mean = features_target.mean(dim=0, keepdim=True)  # (1, d)
    feature_std = features_target.std(dim=0, keepdim=True)    # (1, d)
    feature_std = torch.clamp(feature_std, min=1e-6)          # Avoid division by zero

    logging.info(f"  [Feature Scaler] Computed from target:")
    logging.info(f"    Mean per feature: min={feature_mean.min():.6f}, max={feature_mean.max():.6f}, avg={feature_mean.mean():.6f}")
    logging.info(f"    Std per feature: min={feature_std.min():.6f}, max={feature_std.max():.6f}, avg={feature_std.mean():.6f}")

    return feature_mean, feature_std


def standardize_features(
    features: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor
) -> torch.Tensor:
    """
    Standardize features using pre-computed mean and std (per feature column).

    Parameters
    ----------
    features : torch.Tensor
        Feature matrix (n, d)
    feature_mean : torch.Tensor
        Mean for each feature column (1, d)
    feature_std : torch.Tensor
        Std for each feature column (1, d)

    Returns
    -------
    features_standardized : torch.Tensor
        Standardized features (n, d) where each column has mean≈0, std≈1
    """
    return (features - feature_mean) / feature_std


def unstandardize_features(
    features_standardized: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor
) -> torch.Tensor:
    """
    Invert standardization to recover original scale.

    Parameters
    ----------
    features_standardized : torch.Tensor
        Standardized feature matrix (n, d)
    feature_mean : torch.Tensor
        Mean used for standardization (1, d)
    feature_std : torch.Tensor
        Std used for standardization (1, d)

    Returns
    -------
    features : torch.Tensor
        Features in original scale (n, d)
    """
    return features_standardized * feature_std + feature_mean


def sliced_wasserstein_distance(
    source: torch.Tensor,
    target: torch.Tensor,
    n_projections: int = 128
) -> torch.Tensor:
    """
    Compute Sliced Wasserstein Distance between source and target distributions.

    This encourages the overall distributions to match, helping with variance alignment.

    Parameters
    ----------
    source : torch.Tensor
        Source features (n_source, d)
    target : torch.Tensor
        Target features (n_target, d)
    n_projections : int
        Number of random projections to use

    Returns
    -------
    swd : torch.Tensor
        Scalar SWD loss
    """
    d = source.shape[1]
    device = source.device

    # Generate random projection directions
    theta = torch.randn(d, n_projections, device=device)
    theta = F.normalize(theta, p=2, dim=0)

    # Project both distributions
    source_proj = source @ theta  # (n_source, n_projections)
    target_proj = target @ theta  # (n_target, n_projections)

    # Sort projections
    source_sorted, _ = torch.sort(source_proj, dim=0)
    target_sorted, _ = torch.sort(target_proj, dim=0)

    # Handle different sample sizes by interpolation
    n_source = source.shape[0]
    n_target = target.shape[0]

    if n_source != n_target:
        # Interpolate to common size (use smaller size for efficiency)
        min_size = min(n_source, n_target)

        # Create indices for uniform sampling
        if n_source > min_size:
            indices_s = torch.linspace(0, n_source-1, min_size, device=device).long()
            source_sorted = source_sorted[indices_s]

        if n_target > min_size:
            indices_t = torch.linspace(0, n_target-1, min_size, device=device).long()
            target_sorted = target_sorted[indices_t]

    # Compute L2 distance between sorted projections
    swd = (source_sorted - target_sorted).pow(2).mean()

    return swd


def maximum_mean_discrepancy(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel: str = 'rbf',
    bandwidth: Optional[float] = None
) -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy (MMD) between source and target distributions.

    MMD measures the distance between distributions in kernel space, capturing
    all moments implicitly. Works well with SWD for comprehensive distribution matching.

    Parameters
    ----------
    source : torch.Tensor
        Source features (n_source, d)
    target : torch.Tensor
        Target features (n_target, d)
    kernel : str
        Kernel type ('rbf' for radial basis function)
    bandwidth : float, optional
        Kernel bandwidth. If None, uses median heuristic

    Returns
    -------
    mmd : torch.Tensor
        Scalar MMD loss
    """
    n_source = source.shape[0]
    n_target = target.shape[0]

    # For efficiency with large datasets, sample if needed
    max_samples = 1000
    if n_source > max_samples:
        idx = torch.randperm(n_source)[:max_samples]
        source = source[idx]
        n_source = max_samples
    if n_target > max_samples:
        idx = torch.randperm(n_target)[:max_samples]
        target = target[idx]
        n_target = max_samples

    # Compute pairwise distances
    XX = torch.cdist(source, source, p=2) ** 2
    YY = torch.cdist(target, target, p=2) ** 2
    XY = torch.cdist(source, target, p=2) ** 2

    # Median heuristic for bandwidth if not provided
    if bandwidth is None:
        # Use median of all pairwise distances
        with torch.no_grad():
            all_dists = torch.cat([XX.view(-1), YY.view(-1), XY.view(-1)])
            all_dists = all_dists[all_dists > 0]  # Remove zeros (diagonal)
            bandwidth = torch.sqrt(torch.median(all_dists) / 2.0)
            # Clamp to reasonable range
            bandwidth = torch.clamp(bandwidth, min=0.01, max=100.0)

    # RBF kernel
    if kernel == 'rbf':
        KXX = torch.exp(-XX / (2 * bandwidth ** 2))
        KYY = torch.exp(-YY / (2 * bandwidth ** 2))
        KXY = torch.exp(-XY / (2 * bandwidth ** 2))
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    # MMD computation
    # Unbiased estimator: remove diagonal terms
    KXX_sum = KXX.sum() - torch.diag(KXX).sum()
    KYY_sum = KYY.sum() - torch.diag(KYY).sum()
    KXY_sum = KXY.sum()

    mmd = KXX_sum / (n_source * (n_source - 1)) + \
          KYY_sum / (n_target * (n_target - 1)) - \
          2 * KXY_sum / (n_source * n_target)

    return mmd


def select_focused_cells(
    T: torch.Tensor,
    entropy_percentile: float = 30.0,
    confidence_percentile: float = 70.0
) -> Tuple[torch.Tensor, Dict]:
    """
    Select cells with focused (low entropy) and confident transport.

    Parameters
    ----------
    T : torch.Tensor
        Transport plan (n_source, n_target), row-normalized
    entropy_percentile : float
        Percentile threshold for entropy (select bottom X%)
    confidence_percentile : float
        Percentile threshold for confidence (select top X%)

    Returns
    -------
    focused_mask : torch.Tensor
        Boolean mask (n_source,) indicating focused cells
    stats : dict
        Statistics about selection
    """
    n_source, n_target = T.shape

    # Compute entropy and confidence for each source cell
    T_safe = T.clamp(min=1e-10)
    entropy = -(T * torch.log(T_safe)).sum(dim=1)  # (n_source,)
    max_probs = T.max(dim=1)[0]  # (n_source,)

    # Adaptive thresholds based on percentiles
    entropy_threshold = torch.quantile(entropy, entropy_percentile / 100.0)
    confidence_threshold = torch.quantile(max_probs, confidence_percentile / 100.0)

    # Select cells meeting both criteria
    low_entropy_mask = entropy <= entropy_threshold
    high_confidence_mask = max_probs >= confidence_threshold
    focused_mask = low_entropy_mask & high_confidence_mask

    # Statistics
    n_focused = focused_mask.sum().item()
    n_total = len(focused_mask)

    stats = {
        'n_focused': n_focused,
        'n_total': n_total,
        'pct_focused': 100 * n_focused / n_total if n_total > 0 else 0.0,
        'entropy_threshold': entropy_threshold.item(),
        'confidence_threshold': confidence_threshold.item(),
        'avg_entropy': entropy[focused_mask].mean().item() if n_focused > 0 else 0.0,
        'avg_confidence': max_probs[focused_mask].mean().item() if n_focused > 0 else 0.0
    }

    return focused_mask, stats


def aggregate_training_data_from_batches(
    batch_results: List[Dict],
    features_source_all: torch.Tensor,
    features_target_all: torch.Tensor,
    entropy_percentile: float = 30.0,
    confidence_percentile: float = 70.0
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Aggregate training data from all batch transport results.

    For each batch, selects focused cells and collects their source features
    and matched target features.

    Parameters
    ----------
    batch_results : List[Dict]
        List of E-step results, each containing:
        - 'T': transport plan
        - 'source_indices': source cell indices
        - 'target_indices': target cell indices
    features_source_all : torch.Tensor
        Full source feature matrix (n_source_total, d)
    features_target_all : torch.Tensor
        Full target feature matrix (n_target_total, d)
    entropy_percentile : float
        Percentile for entropy filtering
    confidence_percentile : float
        Percentile for confidence filtering

    Returns
    -------
    source_features_aggregated : torch.Tensor
        Aggregated source features for training (n_train, d_source)
    target_features_aggregated : torch.Tensor
        Aggregated matched target features (n_train, d_target)
    aggregation_stats : dict
        Statistics about aggregation
    """
    all_source_features = []
    all_target_features = []
    total_focused = 0
    total_cells = 0
    batch_stats = []

    for batch_idx, result in enumerate(batch_results):
        T = result['T']
        source_indices = result['source_indices']
        target_indices = result['target_indices']

        # Check if focused_mask was pre-computed in E-step
        focused_mask = result.get('focused_mask', None)

        if focused_mask is not None:
            # Use pre-computed mask from E-step (NEW behavior - avoids redundant computation)
            n_focused = focused_mask.sum().item()
            n_total = len(focused_mask)
            stats = {
                'n_focused': n_focused,
                'n_total': n_total,
                'pct_focused': 100 * n_focused / n_total if n_total > 0 else 0.0,
                'precomputed': True  # Indicates mask came from E-step
            }
            logging.info(f"  Batch {batch_idx + 1}: Using pre-computed focused_mask from E-step ({n_focused}/{n_total} cells)")
        else:
            # Fallback: compute it now (OLD behavior - backward compatible)
            focused_mask, stats = select_focused_cells(
                T, entropy_percentile, confidence_percentile
            )
            stats['precomputed'] = False
            n_focused = focused_mask.sum().item()
            logging.info(f"  Batch {batch_idx + 1}: Computing focused_mask in M-step (fallback, {n_focused}/{T.shape[0]} cells)")

        if n_focused == 0:
            logging.warning(f"  Batch {batch_idx + 1}: No focused cells found, skipping")
            batch_stats.append(stats)
            continue

        # Get source features for focused cells
        focused_local_indices = torch.where(focused_mask)[0]  # Local indices within batch
        focused_global_indices = source_indices[focused_local_indices.cpu().numpy()]  # Global indices (numpy)

        # Convert to torch tensor for CUDA compatibility
        device = features_source_all.device
        focused_global_idx_tensor = torch.from_numpy(focused_global_indices).long().to(device)
        source_features_batch = features_source_all[focused_global_idx_tensor]

        # Get best target matches for focused cells
        best_targets_local = T[focused_mask].argmax(dim=1)  # Local target indices within batch
        best_targets_global = target_indices[best_targets_local.cpu().numpy()]  # Global target indices (numpy)

        # Convert to torch tensor for CUDA compatibility
        best_targets_global_tensor = torch.from_numpy(best_targets_global).long().to(device)
        target_features_batch = features_target_all[best_targets_global_tensor]

        all_source_features.append(source_features_batch)
        all_target_features.append(target_features_batch)

        total_focused += n_focused
        total_cells += T.shape[0]
        batch_stats.append(stats)

        logging.info(f"  Batch {batch_idx + 1}: Selected {n_focused}/{T.shape[0]} cells ({stats['pct_focused']:.1f}%)")

    if len(all_source_features) == 0:
        raise ValueError("No focused cells found in any batch! Cannot train model.")

    # Concatenate all batches
    source_features_aggregated = torch.cat(all_source_features, dim=0)
    target_features_aggregated = torch.cat(all_target_features, dim=0)

    aggregation_stats = {
        'total_focused': total_focused,
        'total_cells': total_cells,
        'pct_focused': 100 * total_focused / total_cells if total_cells > 0 else 0.0,
        'n_batches_used': len(all_source_features),
        'n_batches_total': len(batch_results),
        'batch_stats': batch_stats
    }

    logging.info(f"Aggregated training data: {total_focused} cells from {len(all_source_features)} batches ({aggregation_stats['pct_focused']:.1f}% of total)")

    return source_features_aggregated, target_features_aggregated, aggregation_stats


def train_global_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    steps_per_iter: int,
    lambda_cross: float,
    lambda_struct: float,
    lambda_var: float,
    metric: str,
    structure_sample_size: Optional[int],
    device: torch.device,
    features_target_all: Optional[torch.Tensor] = None
) -> List[float]:
    """
    Train global transformation model on aggregated data.

    Parameters
    ----------
    model : nn.Module
        Feature transformation model
    optimizer : torch.optim.Optimizer
        Optimizer
    source_features : torch.Tensor
        Aggregated source features (n_train, d_source)
    target_features : torch.Tensor
        Aggregated target features (n_train, d_target)
    steps_per_iter : int
        Number of gradient steps
    lambda_cross : float
        Weight for cross-domain alignment loss
    lambda_struct : float
        Weight for structure preservation loss
    lambda_var : float
        Weight for variance preservation loss
    metric : str
        Distance metric ('cosine' or 'euclidean')
    structure_sample_size : int, optional
        Sample size for structure loss (for memory efficiency)
    device : torch.device
        Computation device
    features_target_all : torch.Tensor, optional
        FULL target dataset (n_target_total, d_target) for computing scaler statistics
        If None, scaler is computed from target_features (aggregated subset)
        IMPORTANT: Should always be provided to get correct scaler for applying to new data!

    Returns
    -------
    step_losses : List[float]
        Loss values for each step
    feature_mean : torch.Tensor or None
        Mean used for standardization (if metric=='euclidean'), otherwise None
    feature_std : torch.Tensor or None
        Std used for standardization (if metric=='euclidean'), otherwise None
    """
    model.train()
    torch.set_grad_enabled(True)

    step_losses = []

    # ===== FEATURE STANDARDIZATION (for euclidean metric only) =====
    feature_mean = None
    feature_std = None
    if metric == 'euclidean':
        logging.info(f"  [M-step] Euclidean metric: Standardizing features per-column using TARGET statistics")

        # Compute scaler from FULL TARGET dataset (not aggregated subset!)
        # This ensures scaler can be used to standardize new data correctly
        if features_target_all is not None:
            logging.info(f"    Computing scaler from FULL target dataset (n={features_target_all.shape[0]})")
            feature_mean, feature_std = compute_feature_scaler(features_target_all)
        else:
            logging.warning(f"    features_target_all not provided - computing scaler from aggregated subset (n={target_features.shape[0]})")
            logging.warning(f"    This scaler will NOT match the full target distribution!")
            feature_mean, feature_std = compute_feature_scaler(target_features)

        # Standardize both source and target TRAINING DATA using the scaler
        source_features = standardize_features(source_features, feature_mean, feature_std)
        target_features = standardize_features(target_features, feature_mean, feature_std)

        logging.info(f"    Standardized source (aggregated): mean={source_features.mean():.6f}, std={source_features.std():.6f}")
        logging.info(f"    Standardized target (aggregated): mean={target_features.mean():.6f}, std={target_features.std():.6f}")

    # ===== DIAGNOSTIC: Feature magnitudes BEFORE training =====
    with torch.no_grad():
        source_norms = torch.norm(source_features, dim=1)
        target_norms = torch.norm(target_features, dim=1)
        source_transformed_init = model(source_features)
        transformed_norms_init = torch.norm(source_transformed_init, dim=1)

        logging.info(f"  [M-step Diagnostics - BEFORE training]")
        logging.info(f"    Source feature norms: mean={source_norms.mean():.4f}, std={source_norms.std():.4f}, min={source_norms.min():.4f}, max={source_norms.max():.4f}")
        logging.info(f"    Target feature norms: mean={target_norms.mean():.4f}, std={target_norms.std():.4f}, min={target_norms.min():.4f}, max={target_norms.max():.4f}")
        logging.info(f"    Transformed norms (init): mean={transformed_norms_init.mean():.4f}, std={transformed_norms_init.std():.4f}")

        # Initial losses
        if metric == 'cosine':
            source_norm = F.normalize(source_transformed_init, p=2, dim=1)
            target_norm = F.normalize(target_features, p=2, dim=1)
            init_cross = (1.0 - (source_norm * target_norm).sum(dim=1)).mean()
        else:
            init_cross = F.mse_loss(source_transformed_init, target_features)

        S_source_init = torch.cdist(source_features[:1000], source_features[:1000], p=2) if source_features.shape[0] > 1000 else torch.cdist(source_features, source_features, p=2)
        S_target_init = torch.cdist(target_features[:1000], target_features[:1000], p=2) if target_features.shape[0] > 1000 else torch.cdist(target_features, target_features, p=2)
        init_struct = F.mse_loss(S_target_init, S_source_init)

        #init_swd = sliced_wasserstein_distance(source_transformed_init, target_features, 128)
        #init_mmd = maximum_mean_discrepancy(source_transformed_init, target_features)
        init_swd = 0
        init_mmd = 0

        logging.info(f"    Initial cross loss ({metric}): {init_cross.item():.6f}")
        logging.info(f"    Initial struct loss: {init_struct.item():.6f}")
       #logging.info(f"    Initial SWD loss: {init_swd.item():.6f}")
        #logging.info(f"    Initial MMD loss: {init_mmd.item():.6f}")
        logging.info(f"    Initial SWD loss: 0")
        logging.info(f"    Initial MMD loss: 0") 

    for step in range(steps_per_iter):
        optimizer.zero_grad()

        # Forward pass
        source_transformed = model(source_features)  # (n_train, d_target)
        # ===== Loss 1: Cross-domain alignment =====
        if metric == 'cosine':
            # Normalize and compute cosine distance
            source_norm = F.normalize(source_transformed, p=2, dim=1)
            target_norm = F.normalize(target_features, p=2, dim=1)
            loss_cross = (1.0 - (source_norm * target_norm).sum(dim=1)).mean()
        else:  # euclidean
            loss_cross = F.mse_loss(source_transformed, target_features)
        
        # ===== Loss 2: Distribution matching via SWD and MMD =====
        #loss_swd = sliced_wasserstein_distance(source_transformed, target_features, n_projections=128)
        #loss_mmd = maximum_mean_discrepancy(source_transformed, target_features)
        loss_swd = 0
        loss_mmd = 0
        loss_var = (source_transformed.std(dim=0) - target_features.std(dim=0)).abs().mean()
        if not model.use_residual:
            #logging.info(f"  [M-step] No hidden dimension - computing regularization")
            W = model.net.weight
            identity = torch.eye(W.shape[0], W.shape[1], device=W.device)
            #loss_reg = ((W - identity)**2).mean()
        else:
            #logging.info(f"  [M-step] Hidden dimension - no regularization")
            loss_reg = 0
        loss_reg = 0
        # ===== Loss 3: Structure preservation =====
        n_train = source_features.shape[0]
        if structure_sample_size is not None and n_train > structure_sample_size:
            # Sample for memory efficiency
            sample_indices = torch.randperm(n_train, device=device)[:structure_sample_size]
            source_sample = source_features[sample_indices]
            target_sample = target_features[sample_indices]

            S_source = torch.cdist(source_sample, source_sample, p=2)
            S_target = torch.cdist(target_sample, target_sample, p=2)
            S_source = S_source / (S_source.mean() + 1e-8)
            S_target = S_target / (S_target.mean() + 1e-8)
            loss_struct = F.mse_loss(S_target, S_source)
        else:
            # Use all data
            S_source = torch.cdist(source_features, source_features, p=2)
            S_target = torch.cdist(target_features, target_features, p=2)
            loss_struct = F.mse_loss(S_target, S_source)

        # ===== Combined loss =====
        # Split lambda_var between SWD and MMD (both encourage distribution matching)

        # loss = lambda_cross * loss_cross + lambda_struct * loss_struct + \
        #        lambda_var * (0.5 * loss_swd + 0.5 * loss_mmd) + lambda_var * loss_var + 0.1 * loss_reg
        loss = lambda_cross * loss_cross + lambda_var * loss_var

        # Check for nan/inf
        if not torch.isfinite(loss):
            logging.error(f"Non-finite loss at step {step}: loss={loss.item()}")
            break

        # Backprop
        loss.backward()
        optimizer.step()

        step_losses.append(loss.item())

        # Log progress
        if (step + 1) % 10 == 0 or step == 0:
            if model.use_residual:
                # logging.info(f"  Step {step+1}/{steps_per_iter}: loss={loss.item():.6f} "
                #             f"(cross={loss_cross.item():.4f}, struct={loss_struct.item():.4f}, "f"swd={loss_swd.item():.4f}, mmd={loss_mmd.item():.4f}), var={loss_var.item():.4f}, reg=0")
                logging.info(f"  Step {step+1}/{steps_per_iter}: loss={loss.item():.6f} "
                            f"(cross={loss_cross.item():.4f}, var={loss_var.item():.4f})")
            else:
                # logging.info(f"  Step {step+1}/{steps_per_iter}: loss={loss.item():.6f} "
                #             f"(cross={loss_cross.item():.4f}, struct={loss_struct.item():.4f}, "
                #             f"swd={loss_swd.item():.4f}, mmd={loss_mmd.item():.4f}), var={loss_var.item():.4f}, reg={loss_reg.item():.4f}")
                logging.info(f"  Step {step+1}/{steps_per_iter}: loss={loss.item():.6f} "
                            f"(cross={loss_cross.item():.4f}, var={loss_var.item():.4f})")

                # logging.info(f"  Step {step+1}/{steps_per_iter}: loss={loss.item():.6f} "
                #             f"(cross={loss_cross.item():.4f}, struct={loss_struct.item():.4f}, "
                #             f"swd={loss_swd.item():.4f}, mmd={loss_mmd.item():.4f}), var={loss_var.item():.4f}, reg={loss_reg.item():.4f}")
    # ===== DIAGNOSTIC: Feature magnitudes AFTER training =====
    with torch.no_grad():
        source_transformed_final = model(source_features)
        transformed_norms_final = torch.norm(source_transformed_final, dim=1)

        logging.info(f"  [M-step Diagnostics - AFTER training]")
        logging.info(f"    Transformed norms (final): mean={transformed_norms_final.mean():.4f}, std={transformed_norms_final.std():.4f}")
        logging.info(f"    Norm ratio (final/init): {transformed_norms_final.mean() / transformed_norms_init.mean():.4f}")
        logging.info(f"    Target norms: mean={target_norms.mean():.4f}")

        # Check alignment quality
        pairwise_diff = torch.norm(source_transformed_final - target_features, dim=1)
        logging.info(f"    Pairwise ||transformed - target||: mean={pairwise_diff.mean():.4f}, median={pairwise_diff.median():.4f}")

        # Angular similarity (regardless of metric used)
        source_norm_final = F.normalize(source_transformed_final, p=2, dim=1)
        target_norm_final = F.normalize(target_features, p=2, dim=1)
        cosine_sim = (source_norm_final * target_norm_final).sum(dim=1)
        logging.info(f"    Cosine similarity: mean={cosine_sim.mean():.4f}, median={cosine_sim.median():.4f}")

    if step_losses:
        logging.info(f"M-step completed: {len(step_losses)} steps, avg loss = {np.mean(step_losses):.6f}")
    else:
        logging.warning(f"M-step: No training steps completed!")

    return step_losses, feature_mean, feature_std


def apply_transfer_method(
    batch_results: List[Dict],
    features_target_all: torch.Tensor,
    n_source_total: int,
    device: torch.device
) -> torch.Tensor:
    """
    Apply transfer method by reconstructing full transport matrix and
    computing weighted average of target features.

    Parameters
    ----------
    batch_results : List[Dict]
        List of E-step results
    features_target_all : torch.Tensor
        Full target feature matrix (n_target, d)
    n_source_total : int
        Total number of source cells
    device : torch.device
        Computation device

    Returns
    -------
    features_source_transformed : torch.Tensor
        Transformed source features (n_source, d)
    """
    n_target = features_target_all.shape[0]

    # Reconstruct full transport matrix
    T_full = torch.zeros(n_source_total, n_target, device=device, dtype=torch.float32)

    for result in batch_results:
        T = result['T']
        source_indices = result['source_indices']
        target_indices = result['target_indices']

        # Convert numpy indices to torch tensors for CUDA compatibility
        source_idx_tensor = torch.from_numpy(source_indices).long().to(device)
        target_idx_tensor = torch.from_numpy(target_indices).long().to(device)

        # Place batch transport in full matrix
        T_full[source_idx_tensor[:, None], target_idx_tensor] = T

    # Apply transfer: weighted average of target features
    features_source_transformed = T_full @ features_target_all

    # Diagnostics
    nnz = (T_full > 1e-6).sum().item()
    nnz_pct = 100 * nnz / T_full.numel()
    logging.info(f"Transfer method: T_full nnz = {nnz}/{T_full.numel()} ({nnz_pct:.2f}%)")
    logging.info(f"Transfer method: Transformed {n_source_total} source cells")

    return features_source_transformed
