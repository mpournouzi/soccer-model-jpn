"""
League tables and Monte Carlo season simulation.

Each remaining match is simulated by drawing a scoreline whose outcome probabilities
are exactly the Davidson 1X2 probabilities, with the scoreline shape (and so goal
difference) taken from the Poisson expected goals.
Ranking: points, goal difference, goals scored, then random.
"""

import numpy as np
import pandas as pd

from .poisson import MAX_GOALS, score_matrix

_G = MAX_GOALS + 1
_H, _A = np.divmod(np.arange(_G * _G), _G)
_OUTCOME = np.sign(_H - _A)  # 1 home win, 0 draw, -1 away win


def scoreline_cdf(p1: float, px: float, p2: float, expg1: float, expg2: float) -> np.ndarray:
    m = score_matrix(expg1, expg2).ravel()
    w = np.zeros_like(m)
    for outcome, p in ((1, p1), (0, px), (-1, p2)):
        mask = _OUTCOME == outcome
        w[mask] = m[mask] / m[mask].sum() * p
    cdf = np.cumsum(w)
    return cdf / cdf[-1]


def sample_scores(cdf: np.ndarray, u: np.ndarray):
    idx = np.searchsorted(cdf, u, side='right').clip(max=len(cdf) - 1)
    return _H[idx], _A[idx]


def league_table(results: pd.DataFrame, teams) -> pd.DataFrame:
    rows = {t: dict(team=t, P=0, W=0, D=0, L=0, GF=0, GA=0) for t in teams}
    for home, away, hg, ag in results[['home', 'away', 'hg', 'ag']].itertuples(index=False):
        for team, gf, ga in ((home, hg, ag), (away, ag, hg)):
            r = rows[team]
            r['P'] += 1
            r['GF'] += gf
            r['GA'] += ga
            r['W' if gf > ga else 'D' if gf == ga else 'L'] += 1
    table = pd.DataFrame(rows.values())
    table['GD'] = table['GF'] - table['GA']
    table['Pts'] = 3 * table['W'] + table['D']
    table = table.sort_values(['Pts', 'GD', 'GF'], ascending=False, kind='stable').reset_index(drop=True)
    table.insert(0, 'pos', range(1, len(table) + 1))
    return table


def simulate_season(table: pd.DataFrame, fixtures: pd.DataFrame, n_sims: int, rng) -> dict:
    """
    fixtures needs home, away, prob1, probX, prob2, expg1, expg2.
    Returns position counts (teams x positions) and points samples.
    """
    teams = table['team'].tolist()
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    pts = np.tile(table['Pts'].to_numpy(float), (n_sims, 1))
    gd = np.tile(table['GD'].to_numpy(float), (n_sims, 1))
    gf = np.tile(table['GF'].to_numpy(float), (n_sims, 1))

    for f in fixtures.itertuples(index=False):
        cdf = scoreline_cdf(f.prob1, f.probX, f.prob2, f.expg1, f.expg2)
        hg, ag = sample_scores(cdf, rng.random(n_sims))
        h, a = idx[f.home], idx[f.away]
        pts[:, h] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        pts[:, a] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        gd[:, h] += hg - ag
        gd[:, a] += ag - hg
        gf[:, h] += hg
        gf[:, a] += ag

    key = pts * 1e7 + (gd + 1000) * 1e3 + gf + rng.random((n_sims, n))
    order = np.argsort(-key, axis=1)
    positions = np.empty_like(order)
    positions[np.arange(n_sims)[:, None], order] = np.arange(n)
    counts = np.zeros((n, n))
    for pos in range(n):
        counts[:, pos] = (positions == pos).sum(axis=0)
    return {'teams': teams, 'position_probs': counts / n_sims, 'points': pts}


def season_outputs(sim: dict, markets: dict):
    teams = sim['teams']
    n = len(teams)
    probs = sim['position_probs']
    positions = pd.DataFrame(probs, columns=[f'P{i}' for i in range(1, n + 1)])
    positions.insert(0, 'team', teams)
    positions.insert(1, 'exp_pts', sim['points'].mean(axis=0).round(1))
    positions.insert(2, 'exp_pos', (probs * np.arange(1, n + 1)).sum(axis=1).round(2))
    positions = positions.sort_values('exp_pos').reset_index(drop=True)

    market_probs = pd.DataFrame({'team': teams})
    for name, (lo, hi) in markets.items():
        lo = lo if lo > 0 else n + lo + 1  # negative = counted from the bottom
        hi = hi if hi > 0 else n + hi + 1
        market_probs[name] = probs[:, lo - 1:hi].sum(axis=1)
    market_probs = market_probs.set_index('team').loc[positions['team']].reset_index()
    return positions, market_probs
