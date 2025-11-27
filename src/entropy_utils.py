"""
Entropy-based cell selection utilities for alignment training.

This module provides functions to select high-confidence cells from transport plans
based on entropy and maximum probability criteria.
"""

import logging
from typing import Tuple
import numpy as np
import torch


def select_focused_cells(
    T_det: torch.Tensor,
    entropy_percentile: float = 30.0,
    confidence_percentile: float = 70.0,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select focused cells for training based on transport plan entropy and confidence.

    Uses adaptive percentile-based selection to identify cells with:
    - Low entropy (focused transport to few targets)
    - High confidence (high maximum probability)

    This ensures we always get training samples even with flat transport plans.

    Parameters
    ----------
    T_det : torch.Tensor, shape (n_source, n_target)
        Row-stochastic transport plan (each row sums to 1)
    entropy_percentile : float
        Percentile threshold for entropy selection (lower = more focused)
        Default: 30.0 (select bottom 30% entropy cells)
    confidence_percentile : float
        Percentile threshold for confidence selection (higher = more confident)
        Default: 70.0 (select top 30% confidence cells)
    device : torch.device, optional
        Device for computation

    Returns
    -------
    focused_mask : torch.Tensor, shape (n_source,)
        Boolean mask indicating which cells were selected
    best_targets : torch.Tensor, shape (n_source,)
        Index of best target for each source cell (argmax of T_det)
    entropy : torch.Tensor, shape (n_source,)
        Entropy of each row in T_det

    Notes
    -----
    - Cells must pass BOTH entropy and confidence filters to be selected
    - Returns empty mask if no cells pass both filters (caller should handle this)
    - Logs detailed statistics about selection process
    """
    if device is None:
        device = T_det.device

    n_source, n_target = T_det.shape

    # Compute best targets and max probabilities
    best_targets = torch.argmax(T_det, dim=1)  # (n_source,)
    max_probs = T_det.max(dim=1)[0]  # (n_source,)

    # Compute entropy for each row: H = -Σ p*log(p)
    T_det_safe = T_det.clamp(min=1e-10)  # Avoid log(0)
    entropy = -(T_det * torch.log(T_det_safe)).sum(dim=1)  # (n_source,)

    # Compute adaptive thresholds based on distribution
    max_possible_entropy = np.log(n_target)
    entropy_threshold = torch.quantile(entropy, entropy_percentile / 100.0)
    confidence_threshold = torch.quantile(max_probs, confidence_percentile / 100.0)

    # Apply filters
    low_entropy_mask = entropy <= entropy_threshold
    high_confidence_mask = max_probs >= confidence_threshold
    focused_mask = low_entropy_mask & high_confidence_mask

    # Compute statistics for logging
    n_focused = focused_mask.sum().item()
    n_total = len(focused_mask)
    n_low_entropy = low_entropy_mask.sum().item()
    n_high_conf = high_confidence_mask.sum().item()

    if n_focused > 0:
        avg_entropy_used = entropy[focused_mask].mean().item()
        avg_max_prob_used = max_probs[focused_mask].mean().item()
    else:
        avg_entropy_used = 0.0
        avg_max_prob_used = 0.0
    avg_entropy_all = entropy.mean().item()
    avg_max_prob_all = max_probs.mean().item()

    # Log selection statistics
    logging.info(f"Adaptive cell selection (bottom {entropy_percentile:.0f}% entropy AND top {100-confidence_percentile:.0f}% confidence)")
    logging.info(f"  Entropy distribution: min={entropy.min().item():.4f}, median={entropy.median().item():.4f}, max={entropy.max().item():.4f}, max_possible={max_possible_entropy:.4f}")
    logging.info(f"  Max prob distribution: min={max_probs.min().item():.6f}, median={max_probs.median().item():.6f}, max={max_probs.max().item():.6f}")
    logging.info(f"  Entropy threshold ({entropy_percentile:.0f}th percentile): {entropy_threshold:.4f}")
    logging.info(f"  Confidence threshold ({confidence_percentile:.0f}th percentile): {confidence_threshold:.6f}")
    logging.info(f"  Cells passing entropy filter: {n_low_entropy}/{n_total} ({100*n_low_entropy/n_total:.1f}%)")
    logging.info(f"  Cells passing confidence filter: {n_high_conf}/{n_total} ({100*n_high_conf/n_total:.1f}%)")
    logging.info(f"  Final selected cells (both filters): {n_focused}/{n_total} ({100*n_focused/n_total:.1f}%)")

    if n_focused > 0:
        logging.info(f"  Selected cells - avg entropy: {avg_entropy_used:.4f}, avg max prob: {avg_max_prob_used:.6f}")
    logging.info(f"  All cells - avg entropy: {avg_entropy_all:.4f}, avg max prob: {avg_max_prob_all:.6f}")

    return focused_mask, best_targets, entropy
