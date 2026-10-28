"""
Elo ratings with tournament-specific dynamic ClubElo goal weighting
(port of elo_upd_jpn.py / elo_upd_gr.py).

  1. Map Elo difference -> (lambda_H, lambda_A) using the tournament's average total goals.
     lambda_match decides what the Poisson rates must reproduce:
       win_prob        P(home win) = Elo expected score (original scripts; counts draws twice)
       expected_score  P(home win) + P(draw)/2 = Elo expected score
  2. norm = E[sqrt(margin) | win] under those Poisson rates
  3. K_adjusted = K * sqrt(margin) / norm   (draws: K / mean(norm_home, norm_away))
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .poisson import logistic, margin_norms, solve_lambdas

ELO_RANGE = 600
TOTALS_GRID = np.round(np.arange(2.0, 3.51, 0.1), 1)


class ClubEloWeights:
    def __init__(self, step: int = 10, cache_dir: Path = None, lambda_match: str = 'win_prob'):
        if lambda_match not in ('win_prob', 'expected_score'):
            raise ValueError(f"elo.lambda_match must be win_prob or expected_score, not {lambda_match!r}")
        self.step = step
        self.metric = 'win' if lambda_match == 'win_prob' else 'score'
        suffix = '' if lambda_match == 'win_prob' else '_score'
        self.cache_path = Path(cache_dir) / f'clubelo_norms_step{step}{suffix}.pkl' if cache_dir else None
        self.tables = {}
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, 'rb') as f:
                self.tables = pickle.load(f)
        self._dirty = False

    def _table(self, total: float) -> dict:
        total = float(TOTALS_GRID[np.abs(TOTALS_GRID - total).argmin()])
        if total not in self.tables:
            table = {}
            for d in range(-ELO_RANGE, ELO_RANGE + 1, self.step):
                lh, la = solve_lambdas(d, logistic(d), total, self.metric)
                table[d] = margin_norms(lh, la)
            self.tables[total] = table
            self._dirty = True
        return self.tables[total]

    def multiplier(self, elo_diff: float, margin: int, home_win: bool, total: float) -> float:
        d = int(round(elo_diff / self.step) * self.step)
        norm_home, norm_away = self._table(total)[max(-ELO_RANGE, min(ELO_RANGE, d))]
        if margin == 0:
            return 1.0 / ((norm_home + norm_away) / 2)
        return np.sqrt(margin) / (norm_home if home_win else norm_away)

    def save(self):
        if self._dirty and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.tables, f)


def season_lengths(results: pd.DataFrame, tournaments) -> dict:
    """(team, tournament, season) -> matches the team played in that tournament-season."""
    lg = results[results['tournament'].isin(list(tournaments))]
    long = pd.concat([lg[['home', 'tournament', 'season']].set_axis(['team', 'tournament', 'season'], axis=1),
                      lg[['away', 'tournament', 'season']].set_axis(['team', 'tournament', 'season'], axis=1)])
    return long.groupby(['team', 'tournament', 'season']).size().to_dict()


def run_elo(results: pd.DataFrame, initial_elo: dict, params, weights: ClubEloWeights,
            scale: float = 400, lengths: dict = None):
    """
    Process all results chronologically.

    Returns (ratings dict, history DataFrame with pre-match ratings for every match).
    Uses the final score (hg/ag, including extra time) as the original scripts did.

    A tournament's `season_boost: {k_mult, share}` multiplies K for each team's matches in the
    first `share` of its season in that tournament (match n of N counts while (n + 0.5) / N < share),
    so ratings adapt quickly after the summer and settle later. lengths: (team, tournament, season)
    -> N, e.g. including this season's remaining fixtures; defaults to the matches in `results`.
    Each side gets its own multiplier, so a boosted update isn't zero-sum.
    """
    ratings = dict(initial_elo)
    names = results['tournament'].unique()
    k_by_t = {t: params.k(t) for t in names}
    hfa_by_t = {t: params.elo_hfa(t) for t in names}
    goals_by_t = {t: params.avg_goals(t) for t in names}
    boost = {t: params.season_boost(t) for t in names if params.season_boost(t)}
    if boost:
        lengths = {**season_lengths(results, boost), **(lengths or {})}
    played = {}

    def mult(team, t, season):
        if t not in boost:
            return 1.0
        key = (team, t, season)
        n = played.get(key, 0)
        played[key] = n + 1
        k_mult, share = boost[t]
        return k_mult if (n + 0.5) / lengths[key] < share else 1.0

    pre_home, pre_away = [], []
    for t, s, home, away, hg, ag in results[['tournament', 'season', 'home', 'away', 'hg', 'ag']].itertuples(index=False):
        rh, ra = ratings[home], ratings[away]
        pre_home.append(rh)
        pre_away.append(ra)

        elo_diff = rh - ra + hfa_by_t[t]
        expected = 1 / (1 + 10 ** (-elo_diff / scale))
        actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        g_mult = weights.multiplier(elo_diff, abs(hg - ag), hg > ag, goals_by_t[t])
        delta = k_by_t[t] * g_mult * (actual - expected)
        ratings[home] = rh + delta * mult(home, t, s)
        ratings[away] = ra - delta * mult(away, t, s)

    weights.save()
    history = results.assign(home_elo=pre_home, away_elo=pre_away)
    return ratings, history
