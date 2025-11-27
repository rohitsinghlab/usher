"""
Standalone geosketch implementation for spot alignment.
Copied essential functions from geosketch/sketch.py to avoid import issues.
"""

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances
import sys


def log(message):
    """Simple logging function."""
    print(message)


def gs(X, N, k='auto', seed=None, replace=False,
       alpha=0.1, max_iter=200, one_indexed=False, verbose=0):
    """Sample from a data set according to a geometric plaid covering.

    Parameters
    ----------
    X : `numpy.ndarray`
        Dense vector of low dimensional embeddings with rows corresponding
        to observations and columns corresponding to feature embeddings.
    N: `int`
        Desired sketch size.
    replace: `bool`, optional (default: False)
        When `True`, draws samples with replacement from covering boxes.
    k: `int` or `'auto'` (default: `'auto'`)
        Number of covering boxes.
        When `'auto'` and replace is `True`, draws sqrt(X.shape[0])
        covering boxes.
        When `'auto'` and replace is `False`, draws N covering boxes.
    alpha: `float`
        Binary search halts when it obtains between `k * (1 - alpha)` and
        `k * (1 + alpha)` covering boxes.
    seed: `int`, optional (default: None)
        Random seed passed to numpy.
    max_iter: `int`, optional (default: 200)
        Maximum iterations at which to terminate binary seach in rare
        case of non-monotonicity of covering boxes with box side length.
    one_indexed: `bool`, optional (default: False)
        Returns a 1-indexed result (e.g., R or Matlab indexing), instead
        of a 0-indexed result (e.g., Python or C indexing).
    verbose: `bool` or `int`, optional (default: 0)
        When `True` or not equal to 0, prints logging output.

    Returns
    -------
    samp_idx
        List of indices into X that make up the sketch.
    """
    n_samples, n_features = X.shape

    # Error checking and initialization.
    if not seed is None:
        np.random.seed(seed)
    if not replace and N > n_samples:
        raise ValueError('Cannot sample {} elements from {} elements '
                         'without replacement'.format(N, n_samples))
    if not replace and N == n_samples:
        if one_indexed:
            return list(np.array(range(N)) + 1)
        else:
            return list(range(N))
    if k == 'auto':
        if replace:
            k = int(np.sqrt(n_samples))
        else:
            k = N
    if k < 1:
        raise ValueError('Cannot draw {} covering boxes.'.format(k))

    # Tranlate to make data all positive.
    # Note: `-=' operator mutates variable outside of method.
    X = X - X.min(0)

    # Scale so that maximum value equals 1.
    X /= X.max()

    # Find max value along each dimension.
    X_ptp = np.ptp(X, 0)

    # Range for binary search.
    low_unit, high_unit = 0., max(X_ptp)

    # Initialize box length.
    unit = (low_unit + high_unit) / 4.

    d_to_argsort = {}

    n_iter = 0
    while True:

        if verbose > 1:
            log('n_iter = {}'.format(n_iter))

        grid_table = np.zeros((n_samples, n_features))

        # Assign points to intervals within each dimension.
        for d in range(n_features):
            if X_ptp[d] <= unit:
                continue

            points_d = X[:, d]
            if d not in d_to_argsort:
                d_to_argsort[d] = np.argsort(points_d)
            curr_start = None
            curr_interval = -1
            for sample_idx in d_to_argsort[d]:
                if curr_start is None or \
                   curr_start + unit < points_d[sample_idx]:
                    curr_start = points_d[sample_idx]
                    curr_interval += 1
                grid_table[sample_idx, d] = curr_interval

        # Store as map from grid cells to point indices.
        grid = {}
        for sample_idx in range(n_samples):
            grid_cell = tuple(grid_table[sample_idx, :])
            if grid_cell not in grid:
                grid[grid_cell] = []
            grid[grid_cell].append(sample_idx)
        del grid_table

        if verbose:
            log('Found {} non-empty grid cells'.format(len(grid)))

        if len(grid) > k * (1 + alpha):
            # Too many grid cells, increase unit.
            low_unit = unit
            if high_unit is None:
                unit *= 2.
            else:
                unit = (unit + high_unit) / 2.

            if verbose:
                log('Grid size {}, increase unit to {}'
                    .format(len(grid), unit))

        elif len(grid) < k * (1 - alpha):
            # Too few grid cells, decrease unit.
            high_unit = unit
            if low_unit is None:
                unit /= 2.
            else:
                unit = (unit + low_unit) / 2.

            if verbose:
                log('Grid size {}, decrease unit to {}'
                    .format(len(grid), unit))
        else:
            break

        if high_unit is not None and low_unit is not None and \
           high_unit - low_unit < 1e-20:
            break

        n_iter += 1
        if n_iter >= max_iter:
            # Should rarely get here.
            sys.stderr.write('WARNING: Max iterations reached, try increasing '
                             ' alpha parameter.\n')
            break

    if verbose:
        log('Found {} grid cells'.format(len(grid)))

    # Sample grid cell, then sample point within cell.
    valid_grids = set()
    gs_idx = []
    for n in range(N):
        if len(valid_grids) == 0:
            valid_grids = set(grid.keys())
        valid_grids_list = list(valid_grids)
        grid_cell = valid_grids_list[np.random.choice(len(valid_grids))]
        valid_grids.remove(grid_cell)
        sample = np.random.choice(list(grid[grid_cell]))
        if not replace:
            grid[grid_cell].remove(sample)
            if len(grid[grid_cell]) == 0:
                del grid[grid_cell]
        gs_idx.append(sample)

    if one_indexed:
        gs_idx = [ idx + 1 for idx in gs_idx ]

    return sorted(gs_idx)
