"""
Per-tournament parameters, recomputed from the results on every run
(replaces running the HFA script and copying the numbers into the other scripts).

  hfa_elo     = 400 * log10(home wins / away wins), using 90-minute scores
  draw_rate   = draws / matches
  avg_goals   = mean of hg_90 + ag_90
"""

import numpy as np
import pandas as pd

MIN_MATCHES_FOR_AUTO = 20


def _stats(df: pd.DataFrame) -> dict:
    hw = int((df['hg_90'] > df['ag_90']).sum())
    aw = int((df['hg_90'] < df['ag_90']).sum())
    n = len(df)
    return {
        'matches': n,
        'hfa_elo': 400 * np.log10(hw / aw) if hw and aw else np.nan,
        'draw_rate': (df['hg_90'] == df['ag_90']).mean() if n else np.nan,
        'avg_goals': (df['hg_90'] + df['ag_90']).mean() if n else np.nan,
    }


def tournament_stats(results: pd.DataFrame, season: str) -> pd.DataFrame:
    rows = []
    for name, grp in results.groupby('tournament'):
        overall = _stats(grp)
        current = _stats(grp[grp['season'] == season])
        rows.append({'tournament': name, **overall,
                     **{f'season_{k}': v for k, v in current.items()}})
    return pd.DataFrame(rows).set_index('tournament')


class TournamentParams:
    """Resolves Elo (k, hfa) and prediction (hfa, draw_rate, total_goals) settings."""

    def __init__(self, cfg: dict, stats: pd.DataFrame):
        self.cfg = cfg
        self.stats = stats
        self.elo_cfg = cfg.get('elo', {})
        self.tournaments = cfg.get('tournaments', {}) or {}
        self.default_k = self.elo_cfg.get('default_k', 30)
        self.default_goals = float(stats['avg_goals'].mean()) if len(stats) else 2.5
        self.warnings = []

    def _auto(self, tournament: str, field: str, fallback: float, min_matches: int = MIN_MATCHES_FOR_AUTO) -> float:
        if tournament in self.stats.index:
            row = self.stats.loc[tournament]
            if row['matches'] >= min_matches and not np.isnan(row[field]):
                return float(row[field])
        warning = f"{tournament}: too few matches for auto {field}, using {fallback:.3f}"
        if warning not in self.warnings:
            self.warnings.append(warning)
        return fallback

    def k(self, tournament: str) -> float:
        return float(self.tournaments.get(tournament, {}).get('k', self.default_k))

    def elo_hfa(self, tournament: str) -> float:
        value = self.tournaments.get(tournament, {}).get('hfa', 'auto')
        return self._auto(tournament, 'hfa_elo', 0.0) if value == 'auto' else float(value)

    def avg_goals(self, tournament: str) -> float:
        # Only selects which ClubElo goal-weight table is used, so any sample will do.
        return self._auto(tournament, 'avg_goals', self.default_goals, min_matches=1)

    def prediction(self, tournament: str, overrides: dict = None) -> dict:
        """hfa / draw_rate / total_goals for predicting fixtures in one competition."""
        overrides = overrides or {}
        return {
            'hfa': float(overrides['hfa']) if 'hfa' in overrides
            else self._auto(tournament, 'hfa_elo', 0.0),
            'draw_rate': float(overrides['draw_rate']) if 'draw_rate' in overrides
            else self._auto(tournament, 'draw_rate', 0.25),
            'total_goals': float(overrides['total_goals']) if 'total_goals' in overrides
            else self.avg_goals(tournament),
        }

    def elo_table(self) -> pd.DataFrame:
        names = sorted(self.stats.index)
        return pd.DataFrame([{'tournament': t, 'k': self.k(t), 'elo_hfa': round(self.elo_hfa(t), 1),
                              'avg_goals': round(self.avg_goals(t), 3)} for t in names])
