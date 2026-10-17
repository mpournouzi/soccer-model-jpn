"""
Per-tournament parameters, recomputed from the results on every run
(replaces running the HFA script and copying the numbers into the other scripts).

  hfa_elo     = 400 * log10(home wins / away wins), using 90-minute scores
  draw_rate   = draws / matches
  avg_goals   = mean of hg_90 + ag_90

Prediction parameters can either use all history (method: history) or shrink this
season's rates towards earlier seasons (method: shrink):

  rate = (prior_matches * rate_before_this_season + n * rate_this_season) / (prior_matches + n)

applied to the home-win, away-win and draw shares and to goals per game, so a handful of
early-season matches moves the estimate only a little and a full season moves it a lot.
"""

import numpy as np
import pandas as pd

MIN_MATCHES_FOR_AUTO = 20


def _counts(df: pd.DataFrame) -> dict:
    return {
        'matches': len(df),
        'home_wins': int((df['hg_90'] > df['ag_90']).sum()),
        'away_wins': int((df['hg_90'] < df['ag_90']).sum()),
        'draws': int((df['hg_90'] == df['ag_90']).sum()),
        'goals': int((df['hg_90'] + df['ag_90']).sum()),
    }


def _derived(c: dict) -> dict:
    n, hw, aw = c['matches'], c['home_wins'], c['away_wins']
    return {
        'matches': n,
        'hfa_elo': 400 * np.log10(hw / aw) if hw and aw else np.nan,
        'draw_rate': c['draws'] / n if n else np.nan,
        'avg_goals': c['goals'] / n if n else np.nan,
    }


def tournament_stats(results: pd.DataFrame, season: str) -> pd.DataFrame:
    rows = []
    for name, grp in results.groupby('tournament'):
        current = grp['season'] == season
        overall, now, before = _counts(grp), _counts(grp[current]), _counts(grp[~current])
        rows.append({'tournament': name, **_derived(overall),
                     **{f'season_{k}': v for k, v in _derived(now).items()},
                     **{f'season_{k}': v for k, v in now.items() if k != 'matches'},
                     **{f'prior_{k}': v for k, v in before.items()}})
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
        pred_cfg = cfg.get('prediction_params', {}) or {}
        self.pred_method = pred_cfg.get('method', 'history')
        self.prior_matches = float(pred_cfg.get('prior_matches', 300))
        if self.pred_method not in ('history', 'shrink'):
            raise ValueError(f"prediction_params.method must be history or shrink, not {self.pred_method!r}")
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

    def _shrunk(self, tournament: str):
        """(hfa, draw_rate, total_goals) blending this season into earlier seasons, or None."""
        if tournament not in self.stats.index:
            return None
        r = self.stats.loc[tournament]
        n0, n = r['prior_matches'], r['season_matches']
        if n0 < MIN_MATCHES_FOR_AUTO:
            return None
        w0 = self.prior_matches / n0  # each earlier match counts w0, so the prior totals prior_matches
        denom = self.prior_matches + n

        def share(prior_count, season_count):
            return (w0 * prior_count + season_count) / denom

        hw = share(r['prior_home_wins'], r['season_home_wins'])
        aw = share(r['prior_away_wins'], r['season_away_wins'])
        return (400 * np.log10(hw / aw),
                share(r['prior_draws'], r['season_draws']),
                share(r['prior_goals'], r['season_goals']))

    def prediction(self, tournament: str, overrides: dict = None) -> dict:
        """hfa / draw_rate / total_goals for predicting fixtures in one competition."""
        overrides = overrides or {}
        shrunk = self._shrunk(tournament) if self.pred_method == 'shrink' else None
        if shrunk:
            auto = dict(zip(('hfa', 'draw_rate', 'total_goals'), shrunk))
        else:
            auto = {'hfa': lambda: self._auto(tournament, 'hfa_elo', 0.0),
                    'draw_rate': lambda: self._auto(tournament, 'draw_rate', 0.25),
                    'total_goals': lambda: self.avg_goals(tournament)}
        out = {}
        for key in ('hfa', 'draw_rate', 'total_goals'):
            if key in overrides:
                out[key] = float(overrides[key])
            else:
                out[key] = float(auto[key]() if callable(auto[key]) else auto[key])
        return out

    def elo_table(self) -> pd.DataFrame:
        names = sorted(self.stats.index)
        return pd.DataFrame([{'tournament': t, 'k': self.k(t), 'elo_hfa': round(self.elo_hfa(t), 1),
                              'avg_goals': round(self.avg_goals(t), 3)} for t in names])
