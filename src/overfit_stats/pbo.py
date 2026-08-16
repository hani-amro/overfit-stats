"""Probability of Backtest Overfitting, via Combinatorially Symmetric Cross-Validation.

The question PBO answers: *if I pick the configuration that looked best on half
the data, how often does it land below median on the other half?*

If that happens more than half the time, your selection procedure is worse than
choosing at random -- you are reliably picking the configuration that fits the
noise. The test is symmetric over every way of splitting the timeline into two
halves, so it does not depend on where you happened to cut.

Reference
---------
Bailey, D., Borwein, J., Lopez de Prado, M. & Zhu, Q. (2017). *The Probability of
Backtest Overfitting.* Journal of Computational Finance.
"""

from __future__ import annotations

from itertools import combinations
from typing import Tuple

import numpy as np

__all__ = ["pbo_cscv"]


def pbo_cscv(returns_matrix, n_splits: int = 14) -> Tuple[float, np.ndarray]:
    """Compute PBO over all symmetric in-sample/out-of-sample partitions.

    Parameters
    ----------
    returns_matrix : array-like, shape (T, N)
        Per-bar returns, one column per configuration you tried. All N columns
        must come from the same search -- that is what makes the selection bias
        measurable.
    n_splits : int
        Number of contiguous time blocks S (rounded down to even). Every one of
        ``C(S, S/2)`` partitions is evaluated. S=14 gives 3,432 partitions.

    Returns
    -------
    (pbo, logits)
        ``pbo`` is the fraction of partitions where the in-sample winner ranked
        at or below the out-of-sample median. Near 0 is healthy; above 0.5 is
        damning. NaN when the trial set is degenerate (see below).

    Notes
    -----
    Blocks are contiguous rather than shuffled so that autocorrelation and
    regime structure survive the split. Shuffling bars would leak information
    across the boundary and understate the overfitting.
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("returns_matrix must be 2-D, shape (T, N)")

    T, N = M.shape
    if N < 2:
        return float("nan"), np.array([])

    S = max(n_splits - (n_splits % 2), 2)
    block_len = T // S
    if block_len < 2:
        raise ValueError(
            f"series of length {T} is too short for {S} splits "
            f"(need at least {2 * S} rows)"
        )

    usable = block_len * S
    blocks = M[:usable].reshape(S, block_len, N)

    sum_r = blocks.sum(axis=1)                       # (S, N)
    sum_r2 = (blocks**2).sum(axis=1)                 # (S, N)
    cnt = np.full(S, block_len, dtype=float)

    combos = list(combinations(range(S), S // 2))
    C = len(combos)
    is_mask = np.zeros((C, S), dtype=float)
    for i, c in enumerate(combos):
        is_mask[i, list(c)] = 1.0
    oos_mask = 1.0 - is_mask

    def _sharpe(mask: np.ndarray) -> np.ndarray:
        """Sharpe of every column on every partition, from block sums only."""
        n = mask @ cnt                               # (C,)
        s1 = mask @ sum_r                            # (C, N)
        s2 = mask @ sum_r2                           # (C, N)
        mean = s1 / n[:, None]
        var = s2 / n[:, None] - mean**2

        # A near-constant column has no meaningful Sharpe. Zero it rather than
        # clipping the variance: a clipped 1e-18 would explode the ratio to ~1e9
        # and that column would win every in-sample argmax, corrupting PBO toward
        # a falsely healthy number.
        sh = np.zeros_like(mean)
        good = var > 1e-12
        sh[good] = mean[good] / np.sqrt(var[good])
        return sh

    is_sh = _sharpe(is_mask)
    oos_sh = _sharpe(oos_mask)

    # Degenerate trial set: if every column is effectively the same strategy there
    # was no search to overfit, and a rank-based statistic is undefined. Strict
    # ranking would report PBO=1.0 -- maximally damning -- for something as
    # innocent as a neighbourhood that collapsed to one configuration.
    full_sh = np.array(
        [M[:, j].mean() / (M[:, j].std() + 1e-18) for j in range(N)], dtype=float
    )
    if float(np.nanstd(full_sh)) < 1e-9:
        return float("nan"), np.array([])

    rows = np.arange(C)
    best_is = np.argmax(is_sh, axis=1)               # in-sample winner per partition
    oos_best = oos_sh[rows, best_is]                 # that winner's OOS Sharpe

    # Mid-rank tie handling: a tie contributes 0.5 rather than 0, so identical
    # performance reads as "median" (uninformative) instead of "worst".
    below = (oos_sh < oos_best[:, None]).sum(axis=1)
    tied = (oos_sh == oos_best[:, None]).sum(axis=1)  # includes the winner itself
    ranks = below + 0.5 * (tied - 1)

    omega = np.clip((ranks + 1.0) / (N + 1.0), 1e-6, 1 - 1e-6)
    logits = np.log(omega / (1.0 - omega))
    pbo = float(np.mean(logits <= 0.0))
    return pbo, logits
