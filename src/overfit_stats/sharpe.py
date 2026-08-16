"""Sharpe ratios that account for how hard you looked.

A Sharpe ratio computed on the single best of 600 backtested configurations is
not the same statistic as a Sharpe ratio computed on the one strategy you thought
of first. The functions here correct for that difference.

References
----------
Bailey, D. & Lopez de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.*
Bailey, D. & Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting
for Selection Bias, Backtest Overfitting and Non-Normality.*
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
from scipy.stats import kurtosis, norm, skew

__all__ = [
    "per_period_sharpe",
    "annualized_sharpe",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
]

_EULER_GAMMA = 0.5772156649015329


def _clean(returns: Sequence[float]) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[~np.isnan(r)]


def per_period_sharpe(returns: Sequence[float]) -> float:
    """Sharpe at the data's native frequency. NOT annualised. 0 if degenerate.

    The literature below is written in per-period terms, so this is the form
    every other function here expects. Annualising early is a common way to get
    the DSR benchmark wrong by a factor of sqrt(252).
    """
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def annualized_sharpe(returns: Sequence[float], periods_per_year: int = 252) -> float:
    """Per-period Sharpe scaled by sqrt(periods_per_year). For reporting only."""
    return per_period_sharpe(returns) * math.sqrt(periods_per_year)


def probabilistic_sharpe_ratio(
    returns: Sequence[float], sr_benchmark: float = 0.0
) -> float:
    """P(true per-period Sharpe > ``sr_benchmark``), adjusted for skew and kurtosis.

    Returns a probability in [0, 1], or NaN when the sample is too short or
    degenerate to say anything. Non-normality matters: fat tails and negative
    skew both make a given Sharpe less trustworthy than the normal case implies,
    and this correction is what expresses that.
    """
    r = _clean(returns)
    T = len(r)
    if T < 3:
        return float("nan")

    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")

    sr = r.mean() / sd
    sk = float(skew(r, bias=False))
    ku = float(kurtosis(r, fisher=False, bias=False))   # non-excess: 3 for normal

    denom = 1.0 - sk * sr + ((ku - 1.0) / 4.0) * sr**2
    if denom <= 0 or math.isnan(denom):
        return float("nan")

    z = (sr - sr_benchmark) * math.sqrt(T - 1) / math.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(trial_sharpes: Sequence[float]) -> float:
    """The Sharpe you should expect from the luckiest of N worthless strategies.

    Under the null that every true Sharpe is zero, the maximum observed across N
    trials is still positive, and grows with N and with the spread of the trials.
    This is the bar a *selected* strategy has to clear before it has said
    anything at all.
    """
    s = _clean(trial_sharpes)
    N = len(s)
    if N < 2:
        return 0.0

    # Tolerance, not `<= 0`: the variance of N identical float64 values is not
    # exactly zero (catastrophic cancellation leaves ~1e-33), which would leak a
    # ~1e-17 benchmark instead of a clean 0.0.
    v = s.var(ddof=1)
    if v <= 1e-24:
        return 0.0

    sqrt_v = math.sqrt(v)
    z1 = norm.ppf(1.0 - 1.0 / N)
    z2 = norm.ppf(1.0 - 1.0 / (N * math.e))
    return float(sqrt_v * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    best_returns: Sequence[float], trial_sharpes: Sequence[float]
) -> Tuple[float, float]:
    """Deflated Sharpe Ratio of the selected strategy.

    Returns ``(dsr, sr0)`` where ``sr0`` is the expected-maximum-Sharpe benchmark
    and ``dsr`` is ``PSR(best_returns, benchmark=sr0)``.

    Read it as: the probability the winner is genuinely better than the luckiest
    of N coin flips. Below ~0.5 the impressive in-sample number is statistically
    indistinguishable from noise that was searched hard enough.

    **Returns NaN for a thin or near-identical trial set**, deliberately. With
    too few or too-similar trials the expected-max benchmark collapses toward
    zero, which would produce a near-perfect DSR and a falsely reassuring
    verdict. Reporting "unknown" is the honest answer; a confident 0.99 there
    would be the single most dangerous output this module could produce.
    """
    s = np.asarray(trial_sharpes, dtype=float)
    s = s[np.isfinite(s)]

    too_few = len(s) < 3
    too_similar = len(np.unique(np.round(s, 3))) < 3
    no_spread = len(s) >= 2 and s.var(ddof=1) < 1e-6
    if too_few or too_similar or no_spread:
        return float("nan"), float("nan")

    sr0 = expected_max_sharpe(s)
    dsr = probabilistic_sharpe_ratio(best_returns, sr_benchmark=sr0)
    return dsr, sr0
