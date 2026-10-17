"""
Poisson and Elo-Davidson building blocks shared by the Elo, goals and simulation code.

Everything here is vectorised with numpy; the objective functions are the same ones
used in the original elo_upd_*.py / gr.py / jpn.py scripts, so results match them.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

MAX_GOALS = 12
_K = np.arange(MAX_GOALS + 1)
_LOG_FACT = gammaln(_K + 1)


def pmf(lam: float) -> np.ndarray:
    """Poisson pmf for 0..MAX_GOALS goals."""
    lam = max(lam, 1e-9)
    return np.exp(_K * np.log(lam) - lam - _LOG_FACT)


def score_matrix(lh: float, la: float) -> np.ndarray:
    """P(home=h, away=a) for h, a in 0..MAX_GOALS (independent Poisson)."""
    return np.outer(pmf(lh), pmf(la))


def win_prob(lh: float, la: float) -> float:
    """P(home goals > away goals) under independent Poisson."""
    return np.tril(score_matrix(lh, la), -1).sum()


def outcome_probs(lh: float, la: float):
    """(home win, draw, away win) under independent Poisson."""
    m = score_matrix(lh, la)
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()


def over_prob(total_expg: float, line: float = 2.5) -> float:
    """P(total goals > line); the sum of two Poissons is Poisson."""
    k = np.arange(int(np.floor(line)) + 1)
    lam = max(total_expg, 1e-9)
    return 1.0 - np.exp(k * np.log(lam) - lam - gammaln(k + 1)).sum()


# --------------------------
# Elo-Davidson
# --------------------------
def nu_from_draw_rate(draw_rate: float) -> float:
    """Davidson draw parameter from the draw rate between equal teams."""
    return 2 * draw_rate / (1 - draw_rate)


def davidson(elo_diff_adj, nu: float, scale: float = 400):
    """
    (p_home, p_draw, p_away) from an Elo difference that already includes HFA.
    Works on scalars or numpy arrays.
    """
    theta = 10 ** (np.asarray(elo_diff_adj, dtype=float) / scale)
    root = np.sqrt(theta)
    denom = theta + 1 + nu * root
    return theta / denom, nu * root / denom, 1 / denom


def logistic(elo_diff_adj: float, scale: float = 400) -> float:
    return 1 / (1 + 10 ** (-elo_diff_adj / scale))


# --------------------------
# Expected goals solver
# --------------------------
def solve_lambdas(elo_diff_adj: float, target_p_win: float, total_goals: float):
    """
    Find (lambda_home, lambda_away) whose Poisson home-win probability matches
    target_p_win while lambda_home + lambda_away stays close to total_goals.
    """
    def objective(params):
        lh, la = params
        if lh <= 0.01 or la <= 0.01:
            return 1e10
        return 100 * (win_prob(lh, la) - target_p_win) ** 2 + (lh + la - total_goals) ** 2

    if elo_diff_adj >= 0:
        ratio = 1 + elo_diff_adj / 400
    else:
        ratio = 1 / (1 - elo_diff_adj / 400)
    ratio = max(0.1, min(10, ratio))
    lh_init = total_goals * ratio / (1 + ratio)

    result = minimize(objective, [lh_init, total_goals - lh_init], method='Nelder-Mead',
                      options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
    return max(0.01, result.x[0]), max(0.01, result.x[1])


def margin_norms(lh: float, la: float):
    """
    ClubElo normalisers: E[sqrt(margin) | home win] and E[sqrt(margin) | away win].
    """
    m = score_matrix(lh, la)
    margins = np.arange(1, MAX_GOALS + 1)
    p_home = np.array([np.diagonal(m, -k).sum() for k in margins])
    p_away = np.array([np.diagonal(m, k).sum() for k in margins])
    roots = np.sqrt(margins)
    norm_home = (roots * p_home).sum() / p_home.sum() if p_home.sum() > 0 else 1.0
    norm_away = (roots * p_away).sum() / p_away.sum() if p_away.sum() > 0 else 1.0
    return norm_home, norm_away
