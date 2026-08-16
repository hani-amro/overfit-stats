"""Degenerate inputs must return NaN, never a confident number.

This is the part of the package most likely to cause real harm if it were wrong.
A tool that reports "DSR 0.99, ship it" because the trial set was too thin to
deflate against is worse than no tool at all -- it converts uncertainty into
false confidence and puts a number next to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from overfit_stats import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    monte_carlo_permutation,
    pbo_cscv,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
)

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# DSR guards                                                                    #
# --------------------------------------------------------------------------- #
def test_dsr_is_nan_when_there_are_too_few_trials():
    returns = rng.normal(0.001, 0.01, 500)
    dsr, sr0 = deflated_sharpe_ratio(returns, [0.05, 0.06])
    assert np.isnan(dsr) and np.isnan(sr0)


def test_dsr_is_nan_when_trials_are_near_identical():
    """The dangerous case: many trials, but no real spread to deflate against."""
    returns = rng.normal(0.001, 0.01, 500)
    dsr, _ = deflated_sharpe_ratio(returns, [0.0500, 0.0501, 0.0500, 0.0501] * 25)
    assert np.isnan(dsr), "a collapsed trial set produced a confident DSR"


def test_dsr_is_finite_once_the_trial_set_has_real_spread():
    returns = rng.normal(0.001, 0.01, 500)
    dsr, sr0 = deflated_sharpe_ratio(returns, rng.normal(0.0, 0.05, 50))
    assert np.isfinite(dsr) and np.isfinite(sr0)


def test_expected_max_sharpe_grows_with_trial_count():
    """Looking harder raises the bar. That is the entire correction."""
    spread = rng.normal(0.0, 0.05, 2000)
    assert expected_max_sharpe(spread[:10]) < expected_max_sharpe(spread[:2000])


def test_expected_max_sharpe_is_zero_without_spread():
    assert expected_max_sharpe([0.05] * 100) == 0.0
    assert expected_max_sharpe([0.05]) == 0.0


# --------------------------------------------------------------------------- #
# PSR guards                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore:Precision loss occurred")
def test_psr_is_nan_on_degenerate_series():
    # scipy warns about catastrophic cancellation on constant input. That is the
    # case under test -- the point is that PSR returns NaN rather than a number.
    assert np.isnan(probabilistic_sharpe_ratio([0.01, 0.02]))       # too short
    assert np.isnan(probabilistic_sharpe_ratio([0.01] * 100))       # zero variance


def test_psr_is_a_probability():
    for series in (rng.normal(0.002, 0.01, 400), rng.normal(-0.002, 0.01, 400)):
        p = probabilistic_sharpe_ratio(series)
        assert 0.0 <= p <= 1.0


def test_psr_separates_a_winner_from_a_loser():
    assert probabilistic_sharpe_ratio(rng.normal(0.002, 0.01, 500)) > 0.95
    assert probabilistic_sharpe_ratio(rng.normal(-0.002, 0.01, 500)) < 0.05


def test_per_period_sharpe_is_zero_not_inf_on_constants():
    assert per_period_sharpe([0.01] * 50) == 0.0
    assert per_period_sharpe([0.01]) == 0.0


# --------------------------------------------------------------------------- #
# PBO guards                                                                    #
# --------------------------------------------------------------------------- #
def test_pbo_is_nan_when_every_column_is_the_same_strategy():
    """No search happened, so there is nothing to overfit.

    Strict ranking would score this a maximally-damning PBO of 1.0, which would
    be badly wrong for e.g. a parameter neighbourhood that collapsed to one point.
    """
    col = rng.normal(0.0005, 0.01, 400)
    pbo, logits = pbo_cscv(np.column_stack([col] * 8), n_splits=8)
    assert np.isnan(pbo) and logits.size == 0


def test_pbo_is_nan_with_a_single_column():
    pbo, _ = pbo_cscv(rng.normal(0, 0.01, (400, 1)))
    assert np.isnan(pbo)


def test_pbo_rejects_a_series_too_short_to_split():
    with pytest.raises(ValueError, match="too short"):
        pbo_cscv(rng.normal(0, 0.01, (10, 5)), n_splits=14)


def test_pbo_rejects_a_one_dimensional_input():
    with pytest.raises(ValueError, match="2-D"):
        pbo_cscv(rng.normal(0, 0.01, 400))


def test_pbo_is_a_probability():
    pbo, logits = pbo_cscv(rng.normal(0, 0.01, (600, 12)), n_splits=8)
    assert 0.0 <= pbo <= 1.0
    assert logits.size > 0


# --------------------------------------------------------------------------- #
# Permutation guards                                                            #
# --------------------------------------------------------------------------- #
def test_permutation_is_nan_for_buy_and_hold():
    """Enter once and hold has no timing. Every circular shift ties."""
    positions = np.ones(500)
    p, observed, null = monte_carlo_permutation(positions, rng.normal(0.001, 0.01, 500))
    assert np.isnan(p)
    assert np.isfinite(observed)
    assert null.size == 0


def test_permutation_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="align"):
        monte_carlo_permutation(np.ones(100), np.ones(50))


def test_permutation_p_value_can_never_be_exactly_zero():
    """The observed value is itself a draw from the null."""
    asset = rng.normal(0.0, 0.01, 400)
    positions = (asset > 0).astype(float)          # perfect hindsight timing
    p, _, null = monte_carlo_permutation(positions, asset, n_iter=200)
    assert p > 0.0
    assert p == pytest.approx(1.0 / 201.0, rel=1e-9)
    assert null.size == 200


def test_permutation_is_deterministic_for_a_given_seed():
    asset = rng.normal(0.0, 0.01, 300)
    positions = (rng.normal(0, 1, 300) > 0).astype(float)
    a = monte_carlo_permutation(positions, asset, seed=11, n_iter=100)[0]
    b = monte_carlo_permutation(positions, asset, seed=11, n_iter=100)[0]
    assert a == b


def test_permutation_detects_real_timing():
    """Hindsight timing must come out significant; random timing must not."""
    asset = rng.normal(0.0, 0.01, 600)
    hindsight = (asset > 0).astype(float)
    coinflip = (rng.normal(0, 1, 600) > 0).astype(float)

    p_hindsight, _, _ = monte_carlo_permutation(hindsight, asset, n_iter=500)
    p_coinflip, _, _ = monte_carlo_permutation(coinflip, asset, n_iter=500)

    assert p_hindsight < 0.01
    assert p_coinflip > 0.05
