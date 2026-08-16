# overfit-stats

**Tell whether a backtest found an edge or found a coincidence.**

![tests](https://img.shields.io/badge/tests-41%2F41-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![deps](https://img.shields.io/badge/dependencies-numpy%20%2B%20scipy-green)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

Search 600 strategy configurations against pure random noise and one of them will
post an annualised Sharpe of 1.8. It is not a strategy. It is the maximum of 600
draws, and the maximum of 600 draws is always impressive.

This package implements four peer-reviewed checks that price that in.

```
A. 600 strategies searched against PURE NOISE
  its annualised Sharpe        :   1.80   <- the headline
  expected max under the null  :   1.56   <- what luck alone buys
  Deflated Sharpe Ratio        :  0.681   indistinguishable from luck
  PBO (CSCV)                   :  0.532   overfit
  walk-forward retention       :   0.0%   edge does not survive

B. Same search, but one column has a REAL edge
  its annualised Sharpe        :   4.17   <- the headline
  expected max under the null  :   1.65   <- what luck alone buys
  Deflated Sharpe Ratio        :  1.000   clears the selection-bias bar
  PBO (CSCV)                   :  0.000   healthy
  walk-forward retention       : 100.0%   edge survives selection
```

Real output from [`examples/demo.py`](examples/demo.py).

---

## The four checks

| function | question it answers | reference |
|---|---|---|
| `deflated_sharpe_ratio` | Is this Sharpe better than the luckiest of N trials, after correcting for sample length, skew and kurtosis? | Bailey & López de Prado (2014) |
| `pbo_cscv` | How often does the in-sample winner land below median out-of-sample, across every symmetric split? | Bailey, Borwein, López de Prado & Zhu (2017) |
| `walk_forward_from_matrix` | How much Sharpe survives re-selecting the best configuration on each expanding training window? | — |
| `monte_carlo_permutation` | Does the *timing* beat a random phase shift of the same positions? | — |

```python
import numpy as np
from overfit_stats import deflated_sharpe_ratio, pbo_cscv, per_period_sharpe

# returns_matrix: (T, N) -- one column per configuration you tried
sharpes = np.array([per_period_sharpe(returns_matrix[:, j]) for j in range(N)])
winner  = int(np.argmax(sharpes))

dsr, benchmark = deflated_sharpe_ratio(returns_matrix[:, winner], sharpes)
pbo, _         = pbo_cscv(returns_matrix, n_splits=14)

print(f"DSR {dsr:.3f} against a luck benchmark of {benchmark:.3f}; PBO {pbo:.2f}")
```

---

## How to read the Deflated Sharpe Ratio

This is the part most people get wrong, including an earlier version of this
package's own test suite.

DSR is a **probability**, and its benchmark is by construction the *expected
maximum* across the trials. So for a winner drawn from pure noise the correct
answer is **DSR near 0.5 — uninformative** — not DSR near 0.

| DSR | meaning |
|---|---|
| > 0.95 | clears the selection-bias bar |
| 0.85 – 0.95 | borderline; more data or fewer trials needed |
| ~ 0.5 | indistinguishable from the best of N coin flips |
| NaN | the trial set was too thin or too uniform to deflate against |

Asserting "DSR must be below 0.5 on noise" would be claiming something the
mathematics does not promise — it fails on roughly half of random seeds. The
defensible claim is the *contrast*: a headline Sharpe of 1.80 carrying a DSR of
0.68 is worthless, while the same procedure on a real edge returns 1.000.

---

## NaN is a feature

Every function returns **NaN rather than a confident number** when the input
cannot support a conclusion. A falsely reassuring `0.99` is the most dangerous
output a tool like this can produce — it converts uncertainty into confidence and
puts a decimal point next to it.

- **Thin or near-identical trial set → `deflated_sharpe_ratio` returns NaN.**
  With too few or too-similar trials the expected-max benchmark collapses toward
  zero, which would manufacture a near-perfect DSR out of nothing.
- **Every column effectively the same strategy → `pbo_cscv` returns NaN.** There
  was no search to overfit. Strict ranking would report the maximally damning
  `PBO = 1.0` for something as innocent as a parameter neighbourhood that
  collapsed to a single point.
- **Position series that changes at most once → `monte_carlo_permutation`
  returns NaN.** Every circular shift of "enter once and hold" ties, so the test
  would report a damning `p ≈ 1` for what is simply buy-and-hold. There is no
  timing to test.
- **Constant or too-short return series → `probabilistic_sharpe_ratio` returns
  NaN**, and `per_period_sharpe` returns `0.0` rather than infinity.

---

## Implementation notes worth knowing

**Circular shifts, not shuffles.** The permutation null uses `np.roll` by a
random offset. That preserves both the run-length structure (autocorrelation) and
the turnover of the position series, so the only thing destroyed is the
*alignment* between positions and returns. An i.i.d. shuffle would shatter trend
runs and inflate turnover, making the null artificially weak and the p-value
artificially small.

**Contiguous blocks in CSCV.** Splits are contiguous rather than randomised so
autocorrelation and regime structure survive. Shuffling bars leaks information
across the boundary and understates the overfitting.

**Mid-rank tie handling in PBO.** A tie contributes 0.5 rather than 0, so
identical performance reads as "median" (uninformative) instead of "worst".

**Degenerate columns are zeroed, not clipped.** A near-constant column has no
meaningful Sharpe. Clipping its variance to `1e-18` would explode the ratio to
~`1e9`, and that column would win every in-sample `argmax` — corrupting PBO
toward a falsely healthy number.

**p-values can never be exactly zero.** The observed statistic is itself one draw
from the null, so the estimator is `(count + 1) / (n_iter + 1)`.

**Per-period, not annualised.** The literature is written in per-period terms.
Annualising before deflating is a common way to get the benchmark wrong by a
factor of `sqrt(252)`.

---

## Tests

`python -m pytest` → **41 passed**

| file | what it proves |
|---|---|
| `test_thesis.py` | noise really does produce tempting Sharpes (checked across 8 seeds, not one); DSR endorses a real edge on every seed and never confidently endorses noise; PBO and walk-forward agree with DSR on the same search |
| `test_guards.py` | every degenerate input returns NaN rather than a number; PSR is a probability; permutation is deterministic per seed and detects hindsight timing while rejecting coin-flip timing |

The thesis tests deliberately run across **multiple seeds**. A single seed would
make the demonstration itself depend on a lucky draw — precisely the error this
package exists to detect, and an embarrassing thing to leave in its own suite.

---

## Limitations

- **These are diagnostics, not a green light.** A high DSR means the number
  survived *this* selection-bias correction. It says nothing about regime change,
  liquidity, slippage beyond your assumed cost rate, or whether your data has
  survivorship bias baked in.
- **The trial set must be the real one.** DSR is only honest if `trial_sharpes`
  contains every configuration you actually looked at. Passing the five you
  remember instead of the six hundred you ran defeats the entire correction.
- **`pbo_cscv` is `C(S, S/2)` partitions.** S=14 is 3,432; S=20 is 184,756. Cost
  grows fast.
- **No transaction-cost model beyond a flat rate** in the permutation test.
- **Not benchmarked against reference implementations.** The formulas follow the
  cited papers, and the behavioural tests pin the expected qualitative outcomes,
  but no numeric cross-check against the authors' own code is published here.

---

## Install

```bash
pip install -e ".[test]"
python -m pytest
python examples/demo.py
```

Requires numpy and scipy. Nothing else.

---

## Provenance

Extracted from **BacktestForge**, a bias-free strategy backtester whose product is
an `OVERFIT RISK: HIGH / MEDIUM / LOW` verdict rather than an equity curve.

MIT — see [LICENSE](LICENSE).
