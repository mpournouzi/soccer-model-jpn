"""
Expected goals from Elo difference (Davidson model with team tilts) - port of jpn.py / gr.py.

The lambda lookup stores home_ratio = expg1 / total for each Elo difference (incl. HFA),
solved so that the Poisson home-win probability matches the Davidson home-win probability.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .poisson import davidson, nu_from_draw_rate, solve_lambdas

TILT_DECAY = 0.98
TILT_UPDATE = 0.02
LOOKUP_RANGE = 800


class LambdaLookup:
    def __init__(self, draw_rate: float, ref_total: float, step: int = 5, cache_dir: Path = None):
        self.draw_rate = round(draw_rate, 3)
        self.nu = nu_from_draw_rate(self.draw_rate)
        self.ref_total = round(ref_total, 2)
        self.step = step
        cache = None
        if cache_dir:
            cache = Path(cache_dir) / f'lambda_dr{self.draw_rate:.3f}_tg{self.ref_total:.2f}_s{step}.pkl'
        if cache and cache.exists():
            with open(cache, 'rb') as f:
                self.keys, self.ratios = pickle.load(f)
            return
        self.keys = np.arange(-LOOKUP_RANGE, LOOKUP_RANGE + 1, step)
        ratios = []
        for d in self.keys:
            target, _, _ = davidson(d, self.nu)
            lh, la = solve_lambdas(d, float(target), self.ref_total)
            ratios.append(lh / (lh + la))
        self.ratios = np.array(ratios)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, 'wb') as f:
                pickle.dump((self.keys, self.ratios), f)

    def expg(self, elo_diff_adj, total_goals):
        """(expg1, expg2) - linear interpolation between lookup points; works on arrays."""
        ratio = np.interp(elo_diff_adj, self.keys, self.ratios)
        return total_goals * ratio, total_goals * (1 - ratio)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({'elo_diff_adj': self.keys, 'home_ratio': self.ratios.round(6),
                             'away_ratio': (1 - self.ratios).round(6)})


def compute_tilts(results: pd.DataFrame, params) -> dict:
    """
    Team tilt = tendency for a team's matches to have more (>1) or fewer (<1) goals.
    Each tournament is measured against its own average total goals.
    """
    tilts = {}
    base_by_t = {t: params.avg_goals(t) for t in results['tournament'].unique()}
    for t, home, away, hg, ag in results[['tournament', 'home', 'away', 'hg_90', 'ag_90']].itertuples(index=False):
        base = base_by_t[t]
        th, ta = tilts.get(home, 1.0), tilts.get(away, 1.0)
        total = hg + ag
        tilts[home] = TILT_DECAY * th + TILT_UPDATE * (total / ta / base)
        tilts[away] = TILT_DECAY * ta + TILT_UPDATE * (total / th / base)
    return tilts


class MatchModel:
    """Everything needed to price a match in one competition."""

    def __init__(self, ratings: dict, tilts: dict, pred: dict, lookup: LambdaLookup):
        self.ratings = ratings
        self.tilts = tilts
        self.hfa = pred['hfa']
        self.total_goals = pred['total_goals']
        self.nu = nu_from_draw_rate(pred['draw_rate'])
        self.lookup = lookup

    def predict(self, home: str, away: str, neutral: bool = False) -> dict:
        diff = self.ratings[home] - self.ratings[away]
        diff_adj = diff + (0.0 if neutral else self.hfa)
        p1, px, p2 = davidson(diff_adj, self.nu)
        th, ta = self.tilts.get(home, 1.0), self.tilts.get(away, 1.0)
        expg1, expg2 = self.lookup.expg(diff_adj, self.total_goals * th * ta)
        return {'home_elo': self.ratings[home], 'away_elo': self.ratings[away],
                'elo_diff_adj': diff_adj, 'home_tilt': th, 'away_tilt': ta,
                'prob1': float(p1), 'probX': float(px), 'prob2': float(p2),
                'expg1': float(expg1), 'expg2': float(expg2)}
