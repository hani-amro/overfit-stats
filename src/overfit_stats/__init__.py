"""overfit-stats -- tell whether a backtest found an edge or found a coincidence.

Search 600 strategy configurations against pure random noise and one of them will
post a Sharpe of 1.5 and a triple-digit return. It is not a strategy. It is the
maximum of 600 draws, and the maximum of 600 draws is always impressive.

This package implements four peer-reviewed checks that price that in:

* :func:`deflated_sharpe_ratio` -- corrects the headline Sharpe for sample
  length, skew, kurtosis **and the number of trials it was selected from**.
* :func:`pbo_cscv` -- how often the in-sample winner lands below median
  out-of-sample, across every symmetric split of the timeline.
* :func:`walk_forward_from_matrix` -- how much Sharpe survives re-selecting on
  each expanding training window.
* :func:`monte_carlo_permutation` -- whether the timing beats a random phase
  shift of the same positions.

Every function returns **NaN rather than a confident number** when the input
cannot support a conclusion. A falsely reassuring 0.99 is the most dangerous
output a tool like this can produce.

Extracted from BacktestForge, a bias-free strategy backtester.
"""

from __future__ import annotations

from .pbo import pbo_cscv
from .sharpe import (
    annualized_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
)
from .validation import monte_carlo_permutation, walk_forward_from_matrix

__version__ = "0.1.0"

__all__ = [
    "per_period_sharpe",
    "annualized_sharpe",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "pbo_cscv",
    "walk_forward_from_matrix",
    "monte_carlo_permutation",
    "__version__",
]
