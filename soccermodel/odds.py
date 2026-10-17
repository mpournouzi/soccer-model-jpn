"""
Power-method margins and rounding to the price ladder (as in the cup workbooks).

  exponent = log(n / margin) / log(n)       n = number of outcomes
  odds     = (1 / p) ** exponent, then rounded to the nearest ladder price
"""

import numpy as np

LADDER = np.array([
    1.001, 1.002, 1.003, 1.004, 1.005, 1.006, 1.007, 1.008, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06,
    1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.2, 1.22, 1.25, 1.27,
    1.28, 1.3, 1.32, 1.33, 1.35, 1.36, 1.37, 1.38, 1.4, 1.42, 1.44, 1.45, 1.47, 1.5, 1.53, 1.55,
    1.57, 1.6, 1.61, 1.62, 1.65, 1.66, 1.67, 1.7, 1.72, 1.75, 1.8, 1.83, 1.85, 1.87, 1.9, 1.95,
    2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.37, 2.4, 2.45, 2.5, 2.55, 2.6, 2.62, 2.65, 2.75,
    2.85, 2.87, 2.9, 2.95, 3.0, 3.05, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5, 3.6, 3.75,
    3.8, 3.85, 4.0, 4.15, 4.2, 4.25, 4.3, 4.33, 4.35, 4.5, 4.6, 4.75, 4.85, 5.0, 5.25, 5.5, 5.75,
    6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0, 15.0, 17.0,
    19.0, 21.0, 23.0, 26.0, 29.0, 34.0, 41.0, 51.0, 67.0, 81.0, 101.0, 126.0, 151.0, 201.0,
    251.0, 301.0, 351.0, 501.0, 751.0, 1001.0, 1501.0, 2001.0, 2501.0, 3001.0, 3501.0, 4001.0,
    4501.0, 5001.0,
])


def power_exponent(n_outcomes: int, margin: float) -> float:
    return np.log(n_outcomes / margin) / np.log(n_outcomes)


def to_ladder(odds):
    """Nearest ladder price; NaN stays NaN."""
    odds = np.atleast_1d(np.asarray(odds, dtype=float))
    out = np.full(odds.shape, np.nan)
    ok = ~np.isnan(odds)
    idx = np.abs(LADDER[None, :] - odds[ok, None]).argmin(axis=1)
    out[ok] = LADDER[idx]
    return out


def price(probs, n_outcomes: int, margin: float, ladder: bool = True):
    """Margined odds for an array of probabilities. p == 0 gives NaN (no price)."""
    p = np.atleast_1d(np.asarray(probs, dtype=float))
    with np.errstate(divide='ignore'):
        raw = np.where(p > 0, (1 / p) ** power_exponent(n_outcomes, margin), np.nan)
    return to_ladder(raw) if ladder else raw


def yes_no(probs, margin: float):
    """Two-way yes/no prices for outright markets."""
    p = np.clip(np.asarray(probs, dtype=float), 0, 1)
    return price(p, 2, margin), price(1 - p, 2, margin)
