"""
Plot utilities for spatial alignment visualization.

This module provides reusable plotting functions for alignment diagnostics.
"""

import logging
import os
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad


def plot_dual_umap(
    concat_adata: ad.AnnData,
    cell_type_col: str,
    title_prefix: str,
    save_path: str,
    type_palette: Optional[Dict[str, str]] = None
) -> None:
    """
    Create dual UMAP plots: one with 'type' coloring and one with cell type coloring.

    Parameters
    ----------
    concat_adata : ad.AnnData
        Concatenated AnnData with UMAP coordinates in .obsm['X_umap']
    cell_type_col : str
        Column name in .obs containing cell type annotations
    title_prefix : str
        Prefix for plot titles (e.g., 'Iter 5' or 'Final')
    save_path : str
        Full path to save the plot
    type_palette : dict, optional
        Color palette for 'type' column. If None, uses default colors.
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Default palette for source/target types
        if type_palette is None:
            type_palette = {'source_transformed': '#1f77b4', 'target': '#ff7f0e'}

        # Plot 1: Type coloring
        sc.pl.umap(
            concat_adata,
            color='type',
            palette=type_palette,
            ax=ax1,
            show=False,
            title=f'{title_prefix} - Type'
        )

        # Plot 2: Cell type coloring (if available)
        if cell_type_col in concat_adata.obs.columns:
            sc.pl.umap(
                concat_adata,
                color=cell_type_col,
                ax=ax2,
                show=False,
                title=f'{title_prefix} - {cell_type_col}'
            )
        else:
            ax2.text(0.5, 0.5, f'{cell_type_col} column not found',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'{title_prefix} - Cell Type (Not Available)')

        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        logging.info(f"UMAP plot saved: {save_path}")

    except Exception as e:
        logging.error(f"Failed to create dual UMAP plot: {e}")
        plt.close('all')


def plot_weight_heatmap(
    model,
    title: str,
    save_path: str
) -> None:
    """
    Plot transformation weight matrix as heatmap.

    Parameters
    ----------
    model : nn.Module
        Model with .net attribute containing weight matrix
    title : str
        Title for the heatmap
    save_path : str
        Full path to save the plot
    """
    try:
        import torch.nn as nn

        # Extract weight matrix from model
        W = None
        if hasattr(model, 'net'):
            if isinstance(model.net, nn.Linear):
                W = model.net.weight.detach().cpu().numpy()
            elif isinstance(model.net, nn.Sequential):
                # Find last linear layer
                for m in reversed(model.net):
                    if isinstance(m, nn.Linear):
                        W = m.weight.detach().cpu().numpy()
                        break

        if W is not None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            sns.heatmap(W, cmap='viridis', ax=ax)
            ax.set_title(title)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close('all')
            logging.info(f"Weight heatmap saved: {save_path}")
        else:
            logging.info("No linear layer found for weight heatmap")

    except Exception as e:
        logging.warning(f"Failed to create weight heatmap: {e}")
        plt.close('all')


def plot_convergence(
    convergence_data: List[Dict],
    save_dir: str
) -> None:
    """
    Plot convergence metrics over E-M iterations.

    Creates two subplots:
    1. Percentage of mappings that changed per iteration
    2. Absolute count of mappings that changed per iteration

    Also saves convergence data as CSV for later analysis.

    Parameters
    ----------
    convergence_data : list of dict
        List of convergence metrics with keys: iteration, batch_idx, n_changed, n_total_valid, pct_changed
    save_dir : str
        Directory to save the convergence plot and CSV data
    """
    if len(convergence_data) == 0:
        logging.info("No convergence data to plot (only 1 iteration or no mappings changed)")
        return

    try:
        # Convert to DataFrame
        conv_df = pd.DataFrame(convergence_data)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Percentage of mappings that changed per iteration (per batch)
        for batch_idx in conv_df['batch_idx'].unique():
            if batch_idx == 'ALL_TARGETS':
                continue
            batch_data = conv_df[conv_df['batch_idx'] == batch_idx]
            # Special label for target cell convergence
            label = 'All Targets' if batch_idx == 'ALL_TARGETS' else f'Batch {batch_idx}'
            ax1.plot(batch_data['iteration'], batch_data['pct_changed'],
                    marker='o', label=label, linewidth=2)

        ax1.set_xlabel('E-M Iteration', fontsize=12)
        ax1.set_ylabel('Percentage of Mappings Changed (%)', fontsize=12)
        ax1.set_title('Convergence: Percentage of Changed Mappings per Iteration', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Absolute number of mappings that changed per iteration (per batch)
        for batch_idx in conv_df['batch_idx'].unique():
            if batch_idx != 'ALL_TARGETS':
                continue
            batch_data = conv_df[conv_df['batch_idx'] == batch_idx]
            # Special label for target cell convergence
            label = 'All Targets' if batch_idx == 'ALL_TARGETS' else f'Batch {batch_idx}'
            ax2.plot(batch_data['iteration'], batch_data['n_changed'],
                    marker='o', label=label, linewidth=2)

        ax2.set_xlabel('E-M Iteration', fontsize=12)
        ax2.set_ylabel('Number of Mappings Changed', fontsize=12)
        ax2.set_title('Convergence: Absolute Count of Changed Mappings per Iteration', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # Save plot
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.join(save_dir, 'convergence_plot.png')
        plt.savefig(plot_filename, bbox_inches='tight', dpi=150)
        plt.close()

        logging.info(f"Convergence plot saved: {plot_filename}")

        # Save raw data as CSV
        csv_filename = os.path.join(save_dir, 'convergence_data.csv')
        conv_df.to_csv(csv_filename, index=False)
        logging.info(f"Convergence data saved: {csv_filename}")

    except Exception as e:
        logging.error(f"Failed to create convergence plot: {e}")
        plt.close('all')


def plot_transfer_debug_umap(
    T_full: 'torch.Tensor',
    features_b: 'torch.Tensor',
    adata_a_obs: pd.DataFrame,
    adata_b: ad.AnnData,
    cell_type_col: str,
    iteration: int,
    save_dir: str,
    metric: str = 'euclidean',
    focused_mask_full: Optional['torch.Tensor'] = None,
    use_linear_assignment: bool = False
) -> None:
    """
    Create transfer-based UMAP debug plot.

    This shows what the alignment looks like using just T @ features_b,
    helping separate "is T good?" from "is W good?".

    Parameters
    ----------
    T_full : torch.Tensor
        Full transport matrix (n_source, n_target)
    features_b : torch.Tensor
        Target features
    adata_a_obs : pd.DataFrame
        Source observations (after potential sketching)
    adata_b : ad.AnnData
        Target AnnData
    cell_type_col : str
        Column name for cell type annotations
    iteration : int
        Current iteration number
    save_dir : str
        Directory to save the plot
    metric : str
        Distance metric for UMAP neighbors ('euclidean' or 'cosine')
    focused_mask_full : torch.Tensor, optional
        Boolean mask indicating which source cells are used for M-step training
        (focused = low entropy + high confidence)
    """
    try:
        import torch
        import rapids_singlecell as rsc

        logging.info(f"Creating transfer-based debug plot for iteration {iteration}...")

        # Diagnostics: Check if T is binary (0 or 1) as expected after linear assignment
        T_unique_values = torch.unique(T_full)
        T_nnz = (T_full > 1e-6).sum().item()
        T_sparsity = 100 * T_nnz / T_full.numel()
        logging.info(f"  T_full diagnostics: shape={T_full.shape}, unique_values={T_unique_values.tolist()[:10]}, nnz={T_nnz}/{T_full.numel()} ({T_sparsity:.2f}%)")

        # Check row sums (should be 1.0 for each source cell)
        row_sums = T_full.sum(dim=1)
        logging.info(f"  T_full row sums: min={row_sums.min().item():.4f}, max={row_sums.max().item():.4f}, mean={row_sums.mean().item():.4f}")

        # Identify valid source cells (those with non-zero row sums = have transport assignments)
        valid_source_mask = row_sums > 1e-6
        n_valid = valid_source_mask.sum().item()
        n_total = len(row_sums)
        logging.info(f"  Valid source cells (with transport): {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")

        if n_valid == 0:
            logging.warning("No valid source cells with transport! Cannot create transfer debug plot.")
            return

        # Filter to only valid source cells
        T_full_valid = T_full[valid_source_mask]
        adata_a_obs_valid = adata_a_obs[valid_source_mask.cpu().numpy()].copy()

        # Compute transfer: T @ features_b (only for valid cells)
        with torch.no_grad():
            features_a_transfer = T_full_valid @ features_b
            a_transfer_cpu = features_a_transfer.cpu().numpy()

        # Create AnnData with transferred features (only valid cells)
        adata_a_transfer = ad.AnnData(X=a_transfer_cpu, obs=adata_a_obs_valid)
        adata_a_transfer.obs["type"] = "source_transfer_debug"

        # Add M-step training status if focused_mask provided (only for valid cells)
        if focused_mask_full is not None:
            focused_mask_valid = focused_mask_full[valid_source_mask]
            focused_mask_cpu = focused_mask_valid.cpu().numpy()
            adata_a_transfer.obs["used_for_m_step"] = pd.Categorical(
                ["Used for Training" if f else "Excluded (High Entropy)" for f in focused_mask_cpu]
            )
            logging.info(f"M-step usage (valid cells only): {focused_mask_cpu.sum()}/{len(focused_mask_cpu)} cells used for training " +
                        f"({100*focused_mask_cpu.sum()/len(focused_mask_cpu):.1f}%)")

        adata_b_copy = adata_b.copy()
        adata_b_copy.obs["type"] = "target"

        # IMPORTANT: Add placeholder 'used_for_m_step' column to target cells BEFORE concatenation
        # This ensures the column exists in concat_adata for all cells
        if focused_mask_full is not None:
            adata_b_copy.obs["used_for_m_step"] = pd.Categorical(
                ["Target (placeholder)"] * len(adata_b_copy),
                categories=["Used for Training", "Excluded (High Entropy)", "Target (placeholder)"]
            )

        concat_adata = ad.concat(
            [adata_a_transfer, adata_b_copy], axis=0,
            label="batch", keys=["source", "target"], index_unique="_",
        )

        # Set cell type column if available
        if cell_type_col in adata_a_transfer.obs.columns and cell_type_col in adata_b_copy.obs.columns:
            common_ct_values = pd.concat([adata_a_transfer.obs[cell_type_col], adata_b_copy.obs[cell_type_col]])
            concat_adata.obs[cell_type_col] = common_ct_values.values

        # Compute UMAP
        rsc.tl.pca(concat_adata)
        rsc.pp.neighbors(concat_adata, use_rep='X', metric=metric)
        rsc.tl.umap(concat_adata)

        # Create plot with 3 columns if focused_mask provided, otherwise 2
        n_cols = 3 if focused_mask_full is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(8*n_cols, 6))
        if n_cols == 2:
            ax1, ax2 = axes
        else:
            ax1, ax2, ax3 = axes

        palette = {'source_transfer_debug': '#9467bd', 'target': '#ff7f0e'}
        sc.pl.umap(
            concat_adata,
            color='type',
            palette=palette,
            ax=ax1,
            show=False,
            title=f'Transfer Debug Iter {iteration} - Type'
        )

        if cell_type_col in concat_adata.obs.columns:
            sc.pl.umap(
                concat_adata,
                color=cell_type_col,
                ax=ax2,
                show=False,
                title=f'Transfer Debug Iter {iteration} - {cell_type_col}'
            )
        else:
            ax2.text(0.5, 0.5, f'{cell_type_col} column not found',
                    ha='center', va='center', transform=ax2.transAxes)

        # Plot 3: M-step training status (if focused_mask provided)
        # Show only TARGET cells, colored by whether their matched source was used for M-step
        if focused_mask_full is not None and 'used_for_m_step' in concat_adata.obs.columns and use_linear_assignment:
            import torch

            # Compute inverse mapping: target -> source
            # T_full_valid is binary (0 or 1) after linear assignment
            # For each target, find which source maps to it (if any)
            T_full_valid_t = T_full_valid.t()  # Transpose to (n_target, n_source_valid)
            target_to_source = T_full_valid_t.argmax(dim=1).cpu().numpy()  # (n_target,)
            target_has_match = (T_full_valid_t.max(dim=1)[0] > 0.5).cpu().numpy()  # Boolean mask

            # Get M-step status for each source cell
            focused_mask_valid = focused_mask_full[valid_source_mask]
            focused_mask_cpu = focused_mask_valid.cpu().numpy()

            # Map target cells to their matched source's M-step status
            target_m_step_status = []
            for t_idx in range(len(target_has_match)):
                if target_has_match[t_idx]:
                    # Target has a match, get the matched source's M-step status
                    source_idx = target_to_source[t_idx]
                    if focused_mask_cpu[source_idx]:
                        target_m_step_status.append('Matched to Training Cell')
                    else:
                        target_m_step_status.append('Matched to Excluded Cell')
                else:
                    # Target has no match
                    target_m_step_status.append('No Source Match')

            # Filter to only target cells
            target_only = concat_adata[concat_adata.obs['type'] == 'target'].copy()

            target_only.obs['m_step_status'] = pd.Categorical(
                target_m_step_status,
                categories=['Matched to Training Cell', 'Matched to Excluded Cell', 'No Source Match'],
                ordered=True
            )

            m_step_palette = {
                'Matched to Training Cell': '#2ca02c',  # Green - good matches
                'Matched to Excluded Cell': '#d62728',  # Red - excluded matches
                'No Source Match': '#CCCCCC'  # Gray - unmatched
            }

            sc.pl.umap(
                target_only,  # Plot only target cells
                color='m_step_status',
                palette=m_step_palette,
                ax=ax3,
                show=False,
                title=f'Transfer Debug Iter {iteration} - Target M-step Status'
            )

        # Save plot
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.join(save_dir, f"umap_transfer_iter_{iteration:04d}.png")
        plt.savefig(plot_filename, bbox_inches='tight', dpi=150)
        plt.close()

        logging.info(f"Transfer debug UMAP saved: {plot_filename}")

    except Exception as e:
        logging.warning(f"Failed to create transfer debug plot for iteration {iteration}: {e}")


def plot_spatial_channels(
    adata_source: ad.AnnData,
    adata_target: ad.AnnData,
    spatial_key: str,
    iteration: int,
    save_path: str,
    channel_idx: int = 0,
    figsize: tuple = (15, 5),
    cmap: str = 'jet',
    point_size: float = 1,
    alpha: float = 0.8,
    dpi: int = 150
) -> None:
    """
    Create spatial channel plots showing gene expression on spatial coordinates.

    Parameters
    ----------
    adata_source : ad.AnnData
        Source (transformed) dataset with spatial coordinates
    adata_target : ad.AnnData
        Target dataset with spatial coordinates
    spatial_key : str
        Key in .obsm containing spatial coordinates
    iteration : int
        Current iteration number
    save_path : str
        Base path to save plots (will create 'channels' subdirectory)
    channel_idx : int
        Which channel/gene to plot (default: 0)
    figsize : tuple
        Figure size (default: (15, 5))
    cmap : str
        Colormap to use (default: 'jet')
    point_size : float
        Size of scatter points (default: 1)
    alpha : float
        Transparency of points (default: 0.8)
    dpi : int
        DPI for saved figure (default: 150)
    """
    try:
        # Check if spatial coordinates are available
        if spatial_key not in adata_source.obsm or spatial_key not in adata_target.obsm:
            logging.warning(f"Spatial coordinates not found in .obsm['{spatial_key}']")
            return

        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        # Extract spatial coordinates
        spatial_coords_source = adata_source.obsm[spatial_key]
        spatial_coords_target = adata_target.obsm[spatial_key]

        # Extract expression values for the specified channel
        if hasattr(adata_source.X, 'toarray'):  # Sparse matrix
            expression_source = adata_source.X[:, channel_idx].toarray().flatten()
        else:  # Dense array
            expression_source = adata_source.X[:, channel_idx]

        if hasattr(adata_target.X, 'toarray'):  # Sparse matrix
            expression_target = adata_target.X[:, channel_idx].toarray().flatten()
        else:  # Dense array
            expression_target = adata_target.X[:, channel_idx]

        # Source (transformed) spatial plot
        scatter_source = ax[0].scatter(
            spatial_coords_source[:, 1], spatial_coords_source[:, 0],
            c=expression_source, cmap=cmap, s=point_size, alpha=alpha
        )
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title('Source Transformed')
        ax[0].set_aspect('equal', 'box')
        plt.colorbar(scatter_source, ax=ax[0], fraction=0.046, pad=0.04)

        # Target spatial plot
        scatter_target = ax[1].scatter(
            spatial_coords_target[:, 1], spatial_coords_target[:, 0],
            c=expression_target, cmap=cmap, s=point_size, alpha=alpha
        )
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title('Target')
        ax[1].set_aspect('equal', 'box')
        plt.colorbar(scatter_target, ax=ax[1], fraction=0.046, pad=0.04)

        # Add title and save
        plt.suptitle(f'Channel {channel_idx} - Iteration {iteration}')

        # Create directory if needed
        channels_dir = os.path.join(save_path, 'channels')
        os.makedirs(channels_dir, exist_ok=True)

        plot_filename = os.path.join(channels_dir, f"channel{channel_idx}_iter_{iteration:04d}.png")
        plt.savefig(plot_filename, bbox_inches='tight', dpi=dpi)
        plt.close('all')

        logging.info(f"Spatial channel plot saved to {plot_filename}")

    except Exception as e:
        logging.warning(f"Failed to generate spatial channel plot: {e}")
        plt.close('all')


def plot_spatial_mapping(
    coords_source: np.ndarray,
    coords_target: np.ndarray,
    mapping: np.ndarray,
    iteration: int,
    save_path: str,
    n_samples: int = 5000,
    figsize: tuple = (10, 10),
    alpha_lines: float = 0.3,
    alpha_points: float = 0.5,
    linewidth: float = 0.5,
    point_size_source: float = 2,
    point_size_target: float = 2,
    seed: int = 42
) -> None:
    """
    Visualize spatial alignment by drawing lines from source to mapped target coordinates.

    Parameters
    ----------
    coords_source : np.ndarray
        Source spatial coordinates (n_source, 2)
    coords_target : np.ndarray
        Target spatial coordinates (n_target, 2)
    mapping : np.ndarray
        Mapping from source to target indices (n_source,)
        Values of -1 indicate no mapping
    iteration : int
        Current iteration number
    save_path : str
        Base path to save plots (will create 'mapping' subdirectory)
    n_samples : int
        Maximum number of mapped pairs to visualize (default: 5000)
    figsize : tuple
        Figure size (default: (10, 10))
    alpha_lines : float
        Transparency for mapping lines (default: 0.3)
    alpha_points : float
        Transparency for scatter points (default: 0.5)
    linewidth : float
        Width of mapping lines (default: 0.5)
    point_size_source : float
        Size of source points (default: 2)
    point_size_target : float
        Size of target points (default: 2)
    seed : int
        Random seed for reproducible sampling (default: 42)
    """
    try:
        np.random.seed(seed)

        # Filter valid mappings (exclude -1 values)
        valid_mask = mapping >= 0
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            logging.warning("No valid mappings to visualize")
            return

        # Sample if we have too many points
        if len(valid_indices) > n_samples:
            sampled_indices = np.random.choice(valid_indices, n_samples, replace=False)
        else:
            sampled_indices = valid_indices

        logging.info(f"Plotting {len(sampled_indices)} spatial mappings out of {len(valid_indices)} valid mappings")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot all source and target points as background
        ax.scatter(coords_source[:, 0], coords_source[:, 1],
                  c='lightblue', s=point_size_source, alpha=alpha_points * 0.5,
                  label='Source', zorder=1)
        ax.scatter(coords_target[:, 0], coords_target[:, 1],
                  c='lightgreen', s=point_size_target, alpha=alpha_points * 0.5,
                  label='Target', zorder=1)

        # Plot the sampled mappings
        for i in sampled_indices:
            target_idx = mapping[i]
            if 0 <= target_idx < len(coords_target):  # Double-check validity
                # Draw line from source to target
                ax.plot([coords_source[i, 0], coords_target[target_idx, 0]],
                       [coords_source[i, 1], coords_target[target_idx, 1]],
                       'gray', alpha=alpha_lines, linewidth=linewidth, zorder=2)

        # Highlight the sampled source points
        ax.scatter(coords_source[sampled_indices, 0], coords_source[sampled_indices, 1],
                  c='blue', s=point_size_source, alpha=alpha_points, zorder=3)

        # Highlight the corresponding target points
        target_indices = mapping[sampled_indices]
        valid_targets = target_indices[target_indices >= 0]
        if len(valid_targets) > 0:
            ax.scatter(coords_target[valid_targets, 0], coords_target[valid_targets, 1],
                      c='green', s=point_size_target, alpha=alpha_points, zorder=3)

        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'Spatial Alignment - Iteration {iteration}\n({len(sampled_indices)} mapped pairs shown)')
        ax.set_aspect('equal', 'box')
        ax.legend(loc='best')

        # Create directory if needed
        mapping_dir = os.path.join(save_path, 'mapping')
        os.makedirs(mapping_dir, exist_ok=True)

        plot_filename = os.path.join(mapping_dir, f"spatial_mapping_iter_{iteration:04d}.png")
        plt.savefig(plot_filename, bbox_inches='tight', dpi=150)
        plt.close()

        logging.info(f"Spatial mapping plot saved to {plot_filename}")

    except Exception as e:
        logging.warning(f"Failed to generate spatial mapping plot: {e}")
        plt.close('all')
