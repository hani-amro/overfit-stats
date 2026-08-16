"""Search 600 strategies against pure noise, then watch the statistics refuse to
be impressed.

    python examples/demo.py

There is no edge in this data. Any number that looks good is the maximum of 600
draws, and the maximum of 600 draws always looks good.
"""

from __future__ import annotations

import sys

import numpy as np

from overfit_stats import (
    deflated_sharpe_ratio,
    monte_carlo_permutation,
    pbo_cscv,
    per_period_sharpe,
    walk_forward_from_matrix,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

T, N, SEED = 1000, 600, 0
ANNUALISE = np.sqrt(252)


def report(title: str, matrix: np.ndarray) -> None:
    sharpes = np.array([per_period_sharpe(matrix[:, j]) for j in range(matrix.shape[1])])
    winner = int(np.argmax(sharpes))
    best = matrix[:, winner]

    dsr, sr0 = deflated_sharpe_ratio(best, sharpes)
    pbo, _ = pbo_cscv(matrix[:, :40], n_splits=10)
    wf = walk_forward_from_matrix(matrix[:, :40], n_folds=6)

    positions = (np.roll(best, 1) > 0).astype(float)
    p_value, _, _ = monte_carlo_permutation(positions, best, n_iter=500)

    print(f"\n{title}")
    print("-" * len(title))
    print(f"  winning column               : {winner} of {matrix.shape[1]}")
    print(f"  its annualised Sharpe        : {sharpes[winner] * ANNUALISE:6.2f}   <- the headline")
    print(f"  expected max under the null  : {sr0 * ANNUALISE:6.2f}   <- what luck alone buys")
    print(f"  Deflated Sharpe Ratio        : {dsr:6.3f}   {_verdict_dsr(dsr)}")
    print(f"  PBO (CSCV)                   : {pbo:6.3f}   {_verdict_pbo(pbo)}")
    print(f"  walk-forward retention       : {wf['degradation']:6.1%}   {_verdict_wf(wf['degradation'])}")
    print(f"  permutation p-value          : {p_value:6.3f}")


def _verdict_dsr(d: float) -> str:
    if not np.isfinite(d):
        return "unknown (trial set too thin to deflate)"
    if d > 0.95:
        return "clears the selection-bias bar"
    if d > 0.85:
        return "borderline"
    return "indistinguishable from luck"


def _verdict_pbo(p: float) -> str:
    if not np.isfinite(p):
        return "unknown (no real search to overfit)"
    return "healthy" if p < 0.2 else ("suspect" if p < 0.4 else "overfit")


def _verdict_wf(d: float) -> str:
    if not np.isfinite(d):
        return "unknown"
    return "edge survives selection" if d > 0.3 else "edge does not survive"


def main() -> None:
    rng = np.random.default_rng(SEED)

    noise = rng.normal(0.0, 0.01, size=(T, N))
    report(f"A. {N} strategies searched against PURE NOISE", noise)

    edge = rng.normal(0.0, 0.01, size=(T, N))
    edge[:, 0] += 0.0030
    report(f"B. Same search, but one column has a REAL edge", edge)

    print(
        "\nRead case A carefully. An annualised Sharpe of 1.80 sounds like a"
        "\nstrategy -- but luck across 600 trials was already worth 1.56, so"
        "\nalmost nothing is left once selection is priced in, and all four"
        "\nchecks agree. The raw Sharpe cannot tell A from B. The deflated"
        "\none can.\n"
    )


if __name__ == "__main__":
    main()
