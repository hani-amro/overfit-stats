"""The headline claim, tested: noise produces impressive Sharpes, and these
statistics refuse to be impressed.

A note on what "refuse" means, because it is easy to state this wrongly. The
Deflated Sharpe Ratio is a *probability*, and its benchmark is by construction
the expected maximum across the trials. So for a winner drawn from pure noise the
correct answer is **DSR near 0.5 -- uninformative** -- not DSR near 0. Asserting
"DSR < 0.5 on noise" would be asserting something the mathematics does not
promise, and it fails on roughly half of random seeds.

The real, defensible claim is the *contrast*: an annualised Sharpe of 1.8 that
carries a DSR of 0.5 is worthless, while the same headline number backed by a
genuine edge carries a DSR above 0.99. That is what these tests assert.
"""

from __future__ import annotations

import numpy as np
import pytest

from overfit_stats import (
    deflated_sharpe_ratio,
    pbo_cscv,
    per_period_sharpe,
    walk_forward_from_matrix,
)

T, N = 1000, 600


def _pure_noise(seed: int = 42) -> np.ndarray:
    """(T, N) of zero-mean Gaussian returns. Not one column has any edge."""
    return np.random.default_rng(seed).normal(0.0, 0.01, size=(T, N))


def _one_real_edge(seed: int = 7) -> np.ndarray:
    """Same noise, except column 0 carries a genuine, unambiguous drift.

    The drift is large enough (per-period Sharpe ~0.30 against a best-of-599
    noise maximum of ~0.11) that column 0 reliably wins the argmax. Otherwise the
    test would be measuring whether the edge happens to win the search, which is
    a different question from whether DSR endorses it once it has.
    """
    M = np.random.default_rng(seed).normal(0.0, 0.01, size=(T, N))
    M[:, 0] += 0.0030
    return M


def _best_column_dsr(M: np.ndarray) -> float:
    sharpes = np.array([per_period_sharpe(M[:, j]) for j in range(M.shape[1])])
    dsr, _ = deflated_sharpe_ratio(M[:, int(np.argmax(sharpes))], sharpes)
    return dsr


# --------------------------------------------------------------------------- #
# 1. Establish the problem before testing the cure                              #
# --------------------------------------------------------------------------- #
def test_searching_noise_produces_a_tempting_sharpe():
    """Without this, a low DSR below could just mean the data was obviously bad.

    Checked across eight independent draws rather than one. A single seed would
    make the demonstration itself depend on one lucky sample -- the same selection
    effect this package measures, applied to its own evidence.
    """
    annualised = []
    for seed in range(8):
        sharpes = np.array(
            [per_period_sharpe(_pure_noise(seed)[:, j]) for j in range(N)]
        )
        annualised.append(sharpes.max() * np.sqrt(252))

    assert min(annualised) > 1.0, (
        f"weakest of {len(annualised)} noise searches reached only "
        f"{min(annualised):.2f} annualised Sharpe"
    )
    assert float(np.mean(annualised)) > 1.4, (
        f"mean best-of-{N} annualised Sharpe on pure noise was "
        f"{np.mean(annualised):.2f}; the demonstration needs a tempting number"
    )


# --------------------------------------------------------------------------- #
# 2. The contrast                                                               #
# --------------------------------------------------------------------------- #
def test_dsr_endorses_a_real_edge_and_not_a_noise_winner():
    dsr_edge = _best_column_dsr(_one_real_edge())
    dsr_noise = _best_column_dsr(_pure_noise())

    assert dsr_edge > 0.95, f"DSR {dsr_edge:.3f} rejected a genuine edge"
    assert dsr_noise < 0.85, f"DSR {dsr_noise:.3f} confidently endorsed noise"
    assert dsr_edge - dsr_noise > 0.3


@pytest.mark.parametrize("seed", range(8))
def test_a_real_edge_is_endorsed_on_every_seed(seed):
    assert _best_column_dsr(_one_real_edge(seed)) > 0.95


@pytest.mark.parametrize("seed", range(8))
def test_noise_is_never_confidently_endorsed(seed):
    """The honest guarantee: no *confident* endorsement, on any seed.

    DSR on a noise winner scatters around 0.5. What must never happen is a
    number that reads as "ship it".
    """
    dsr = _best_column_dsr(_pure_noise(seed))
    assert dsr < 0.85, f"seed {seed}: DSR {dsr:.3f} would read as an endorsement"


def test_noise_dsr_centres_on_uninformative():
    """Across seeds the noise DSR should sit near 0.5, which is the whole point."""
    values = [_best_column_dsr(_pure_noise(s)) for s in range(12)]
    assert 0.35 < float(np.mean(values)) < 0.70, (
        f"mean DSR on noise was {np.mean(values):.3f}; expected ~0.5"
    )


# --------------------------------------------------------------------------- #
# 3. The other two checks must agree                                            #
# --------------------------------------------------------------------------- #
def test_pbo_is_high_on_noise_and_low_on_a_real_edge():
    pbo_noise, _ = pbo_cscv(_pure_noise()[:, :40], n_splits=10)
    pbo_edge, _ = pbo_cscv(_one_real_edge()[:, :40], n_splits=10)

    assert pbo_noise > 0.4, f"PBO {pbo_noise:.2f} called a noise search healthy"
    assert pbo_edge < 0.2, f"PBO {pbo_edge:.2f} condemned a genuine edge"


def test_walk_forward_retains_more_of_a_real_edge_than_of_noise():
    """Absolute retention depends on fold count; the contrast is the signal."""
    noise = walk_forward_from_matrix(_pure_noise()[:, :40], n_folds=6)
    edge = walk_forward_from_matrix(_one_real_edge()[:, :40], n_folds=6)

    assert noise["degradation"] < 0.1, (
        f"walk-forward retained {noise['degradation']:.0%} of a noise-fitted edge"
    )
    assert edge["degradation"] > noise["degradation"]
    assert edge["oos_sharpe"] > noise["oos_sharpe"]


def test_all_three_checks_agree_on_the_same_noise_search():
    """A verdict built on one statistic is a verdict built on one assumption."""
    M = _pure_noise()
    dsr = _best_column_dsr(M)
    pbo, _ = pbo_cscv(M[:, :40], n_splits=10)
    wf = walk_forward_from_matrix(M[:, :40], n_folds=6)

    assert dsr < 0.85 and pbo > 0.4 and wf["degradation"] < 0.1
